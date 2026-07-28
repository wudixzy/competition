#!/usr/bin/env python3
"""Create a verified private session and reap all service descendants."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time


PR_SET_CHILD_SUBREAPER = 36


def _starttime_ticks(pid: int) -> int:
    value = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    closing = value.rfind(")")
    if closing < 0:
        raise RuntimeError("malformed process stat")
    fields_after_command = value[closing + 2:].split()
    return int(fields_after_command[19])


def _enable_child_subreaper() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _returncode(value: int) -> int:
    return value if value >= 0 else 128 - value


def _run_and_reap(command: list[str]) -> int:
    pending_signal: int | None = None
    forwarded_signal: int | None = None

    def remember_signal(signum, _frame) -> None:
        nonlocal pending_signal
        pending_signal = signum

    previous_handlers = {
        signum: signal.signal(signum, remember_signal)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    child = None
    try:
        if pending_signal is not None:
            return 128 + pending_signal
        child = subprocess.Popen(command, env=os.environ)
        while child.poll() is None:
            if (
                pending_signal is not None
                and forwarded_signal != pending_signal
            ):
                try:
                    os.kill(child.pid, pending_signal)
                except ProcessLookupError:
                    pass
                forwarded_signal = pending_signal
            time.sleep(0.05)
        child_returncode = child.returncode

        # The API server can exit before its multiprocessing workers. As a
        # subreaper, this session leader adopts those workers and waits until
        # every descendant has been reaped.
        while True:
            try:
                os.waitpid(-1, 0)
            except InterruptedError:
                continue
            except ChildProcessError:
                break
        return _returncode(child_returncode)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 3 or args[1] != "--":
        print(
            "usage: exec_bi100_session.py IDENTITY.json -- COMMAND [ARG ...]",
            file=sys.stderr,
        )
        return 2
    identity = Path(args[0])
    command = args[2:]
    if not identity.is_absolute() or not identity.parent.is_dir():
        print("identity path must have an existing absolute parent",
              file=sys.stderr)
        return 2
    if identity.exists():
        print(f"identity path already exists: {identity}", file=sys.stderr)
        return 2

    os.setsid()
    try:
        _enable_child_subreaper()
    except OSError as error:
        print(
            f"cannot enable child subreaper: {error.strerror}",
            file=sys.stderr,
        )
        return 2
    pid = os.getpid()
    session_token = secrets.token_hex(16)
    os.environ["BI100_PROCESS_SESSION_TOKEN"] = session_token
    report = {
        "schema": "bi100-process-session-v1",
        "version": 1,
        "pid": pid,
        "pgid": os.getpgrp(),
        "sid": os.getsid(0),
        "starttime_ticks": _starttime_ticks(pid),
        "session_token": session_token,
    }
    temporary = identity.with_name(f".{identity.name}.{pid}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        payload = (
            json.dumps(report, ensure_ascii=True, sort_keys=True) + "\n"
        ).encode("ascii")
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("identity write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, identity)
    directory = os.open(identity.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)

    print("[BI100 SESSION] child subreaper active", flush=True)
    try:
        return _run_and_reap(command)
    except OSError as error:
        print(
            f"cannot start session command: {error.strerror}",
            file=sys.stderr,
        )
        return 127


if __name__ == "__main__":
    raise SystemExit(main())
