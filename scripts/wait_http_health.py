#!/usr/bin/env python3
"""Wait for loopback health without exceeding one monotonic deadline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable
import urllib.request


Json = dict[str, Any]
SCHEMA = "bi100-http-health-wait-v1"


def _process_starttime(pid: int) -> int | None:
    try:
        value = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except (FileNotFoundError, ProcessLookupError):
        return None
    fields = value[value.rfind(")") + 2:].split()
    return int(fields[19])


def wait_for_health(
    url: str,
    *,
    pid: int,
    starttime_ticks: int,
    timeout_s: float,
    opener: Callable[..., Any] = urllib.request.urlopen,
    process_starttime: Callable[[int], int | None] = _process_starttime,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Json:
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    started = monotonic()
    deadline = started + timeout_s
    attempts = 0
    last_error: str | None = None
    reason = "deadline_expired"

    while True:
        if process_starttime(pid) != starttime_ticks:
            reason = "service_identity_lost"
            break
        remaining = deadline - monotonic()
        if remaining <= 0:
            break
        attempts += 1
        try:
            with opener(
                    url, timeout=max(0.001, min(5.0, remaining))) as response:
                response.read()
            if process_starttime(pid) != starttime_ticks:
                reason = "service_identity_lost_after_health"
                break
            return {
                "schema": SCHEMA,
                "version": 1,
                "qualified": True,
                "attempts": attempts,
                "elapsed_s": monotonic() - started,
                "timeout_s": timeout_s,
                "reason": "healthy",
                "last_error": None,
            }
        except Exception as exc:
            last_error = type(exc).__name__
        remaining = deadline - monotonic()
        if remaining <= 0:
            break
        sleep(min(1.0, remaining))

    return {
        "schema": SCHEMA,
        "version": 1,
        "qualified": False,
        "attempts": attempts,
        "elapsed_s": monotonic() - started,
        "timeout_s": timeout_s,
        "reason": reason,
        "last_error": last_error,
    }


def _atomic_write(path: Path, value: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=True, indent=2,
                      sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--starttime-ticks", type=int, required=True)
    parser.add_argument("--timeout-s", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.pid <= 1 or args.starttime_ticks <= 0:
        parser.error("process identity must be positive")
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be positive")
    report = wait_for_health(
        f"http://127.0.0.1:{args.port}/health",
        pid=args.pid,
        starttime_ticks=args.starttime_ticks,
        timeout_s=args.timeout_s,
    )
    _atomic_write(args.out, report)
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
