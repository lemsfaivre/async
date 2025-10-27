#!/usr/bin/env python3

import asyncio
import contextvars
import json
import signal
import sys
import tomllib
from collections.abc import Callable
from copy import deepcopy
from logging import Logger
from pathlib import Path
from typing import Annotated

from loguru import logger as log
from rich import print

import httpx
from tenacity import RetryCallState, retry, stop_after_attempt, wait_exponential
from typer import Exit, Option, Typer

SENTINEL: object = object()
TASK_NAME: contextvars.ContextVar  = contextvars.ContextVar("task_name", default="main")


# ======================================================================================================================
def setup_logging(cfg: dict):
    """Configure loguru with task name context."""
    log.remove()

    log_level: str = cfg.get("logging", {}).get("level", "INFO").upper()

    if cfg.get("logging", {}).get("console", True):
        log.add(
            sink=sys.stdout,
            level=log_level,
            format="<green>{time:HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{extra[task]: <8}</cyan> | "
                   "{message}",
            colorize=True,
            enqueue=True,
        )

    # Structured JSON logs
    json_log_file: str = cfg["logging"].get("json_file", "logs.jsonl")
    log.add(sink=json_log_file, level=log_level, serialize=True, enqueue=True)

    log.configure(extra={"task": "main"})


def get_logger() -> Logger:
    """Return a logger bound to the current task name."""
    current_task: asyncio.Task = asyncio.current_task()
    task_name: str = current_task.get_name() if current_task else TASK_NAME.get()
    return log.bind(task=task_name)


def load_settings(path: Path):
    with path.open("rb") as f:
        return tomllib.load(f)


def merge_overrides(base_cfg: dict, **overrides) -> dict:
    """Merge cLI overrides into TOML config."""
    cfg: dict = deepcopy(base_cfg)
    if overrides.get("timeout") is not None:
        cfg["general"]["global_timeout"] = overrides["timeout"]
    if overrides.get("poll_interval") is not None:
        cfg["general"]["pool_interval"] = overrides["poll_interval"]
    if overrides.get("json_file") is not None:
        cfg["logging"]["json_file"] = overrides["json_file"]
    if overrides.get("level") is not None:
        cfg["logging"]["level"] = overrides["level"].upper()
    return cfg

# ======================================================================================================================
# TENACITY RETRY LOGGING
# ======================================================================================================================
def log_retry_attempt(retry_state: RetryCallState) -> None:
    """Structured logging of retry attempts."""
    fn_name: str = retry_state.fn.__name__
    attempt: int = retry_state.attempt_number
    next_wait: int = getattr(retry_state.next_action, "sleep", None)
    exception: str = retry_state.outcome.exception() if retry_state.outcome else None

    log: Logger = get_logger()
    if exception:
        msg: str = f"Retrying {fn_name} after exception: {exception!r} (attempt {attempt})"
        if next_wait:
            msg += f", next retry in {next_wait:.1f}s"
        log.warning(msg)
    else:
        log.debug(f"Retrying {fn_name} (attempt {attempt})")


def build_retry_decorator(cfg: dict) -> Callable:
    """Create a tenacity retry decorator based on config values."""
    attempts: int = cfg.get("retry", {}).get("attempts", 1)
    min_wait: int = cfg.get("retry", {}).get("min_wait", 1)
    max_wait: int = cfg.get("retry", {}).get("max_wait", 10)

    return retry(
        stop=stop_after_attempt(max_attempt_number=attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        before_sleep=log_retry_attempt,
        reraise=True,
    )


# ======================================================================================================================
# PRODUCER / CONSUMER
# ======================================================================================================================
async def producer(
    name: str, url: str, queue: asyncio.Queue, retry_decorator: Callable, poll_interval: int, stop_event: asyncio.Event
) -> None:
    log: Logger = get_logger()

    @retry_decorator
    async def fetch_json(client: httpx.AsyncClient, url: str) -> dict:
        resp: httpx.Response = await client.get(url, timeout=5)
        resp.raise_for_status()
        return resp.json()

    log.info("Starting...")
    async with httpx.AsyncClient() as client:
        while not stop_event.is_set():
            try:
                data: dict = await fetch_json(client=client, url=url)
                await queue.put((name, data))
                log.info(f"Queued {len(data)} items from {url!r}")
            except asyncio.CancelledError:
                log.warning("Cancelled while fetching - exiting loop")
                break
            except Exception  as e:
                log.error(f"HTTP error: {repr(e)!r}")
            await asyncio.sleep(delay=poll_interval)

    # graceful exit
    await queue.put(SENTINEL)
    log.info("Producer finished and sent SENTINEL")


# ======================================================================================================================
async def consumer(queue: asyncio.Queue, output: Path, producers_count: int) -> None:
    log: Logger = get_logger()

    log.info("Starting...")
    done_count: int = 0

    with open(file=output, mode="a", encoding="utf-8") as f:
        while True:
            item: tuple[str, dict] = await queue.get()
            if item is SENTINEL:
                done_count += 1
                log.info(f"Received SENTINEL ({done_count}/{producers_count})")
                if done_count == producers_count:
                    log.info("All producers done - exiting consumer")
                    break
                continue

            name, data = item
            json_line: str = json.dumps({"source": name, "data": data})
            f.write(json_line + "\n")
            f.flush()
            queue.task_done()
            log.info(f"Wrote data from {name}")


# ======================================================================================================================
# MAIN RUNNER
# ======================================================================================================================
async def run_app(cfg: dict):
    stop_event: asyncio.Event = asyncio.Event()

    asyncio.current_task().set_name("main")
    log: Logger = get_logger()

    queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
    retry_decorator = build_retry_decorator(cfg)

    timeout: int = cfg.get("general", {}).get("global_timeout", 60)
    poll_interval: str = cfg.get("general", {}).get("poll_interval")
    output_file: str = cfg.get("general", {}).get("output_file")

    urls: dict[str, str] = cfg.get("apis")
    producer_tasks: list[asyncio.Task] = list()
    for name, url in urls.items():
        task: asyncio.Task = asyncio.create_task(
            coro=producer(
                name=name,
                url=url,
                queue=queue,
                retry_decorator=retry_decorator,
                poll_interval=poll_interval,
                stop_event=stop_event,
            ),
            name=name
        )
        producer_tasks.append(task)

    consumer_task: asyncio.Task = asyncio.create_task(
        consumer(queue=queue, output=output_file, producers_count=len(producer_tasks)), name="consumer"
    )

    all_tasks: list[asyncio.Task] = producer_tasks + [consumer_task]

    loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    try:
        log.info(f"Running with global timeout={timeout}")
        async with asyncio.timeout(timeout):
            # handle tasks as they complete
            async for completed in asyncio.as_completed(all_tasks):
                task_name: str = completed.get_name()
                try:
                    result = await completed
                    log.info(f"{task_name} completed successfully: {result}")
                except asyncio.CancelledError:
                    log.warning("Task cancelled")
                except Exception as e:
                    log.error(f"{task_name} raised an exception: {repr(e)!r}")
    except asyncio.TimeoutError:
        log.warning(f"Global timeout of ({timeout}s) reached - cancelling remaining tasks...")
        stop_event.set()
    finally:
        await asyncio.sleep(1)
        for t in all_tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*all_tasks, return_exceptions=True)
        log.info("Shutdown complete.")


# ======================================================================================================================
# TYPER CLI
# ======================================================================================================================
app: Typer = Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Async HTTP data fetcher with retry, queue, structured logging.",
)

@app.command()
def run(
    settings_file: Annotated[Path, Option("-s", "--settings", help="Path to TOML settings file.")] = "settings.toml",
    timeout: Annotated[int, Option("-t", "--timeout", show_default=False, help="Override global timeout.")] = None,
    poll_interval: Annotated[int, Option("-p", "--poll", show_default=False, help="Override polling interval.")] = None,
    json_file: Annotated[str, Option("-j", "--json-file", show_default=False, help="Override JSON log output.")] = None,
    level: Annotated[str, Option("-l", "--level", show_default=False, help="Override log level.")] = None,
    dry_run: Annotated[bool, Option("--dry-run", show_default=False, help="Load and validate configuration.")] = False
):
    """Run the async data pipeline."""
    base_config: dict = load_settings(settings_file)
    cfg: dict = merge_overrides(
        base_cfg=base_config, timeout=timeout, json_file=json_file, poll_interval=poll_interval,level=level
    )
    setup_logging(cfg=cfg)
    # log: Logger = get_logger()

    log.info("Configuration loaded")

    if dry_run:
        log.info("Configuration validated successfully")
        print(cfg)
        raise Exit(code=0)

    asyncio.run(run_app(cfg))


if __name__ == "__main__":
    asyncio.run(app())

