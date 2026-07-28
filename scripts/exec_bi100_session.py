#!/usr/bin/env python3
"""Create a verified private session, record its identity, then exec."""

from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
import sys


def _starttime_ticks(pid: int) -> int:
    value = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    closing = value.rfind(")")
    if closing < 0:
        raise RuntimeError("malformed process stat")
    fields_after_command = value[closing + 2:].split()
    return int(fields_after_command[19])


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

    os.execvpe(command[0], command, os.environ)
    raise AssertionError("os.execvpe unexpectedly returned")


if __name__ == "__main__":
    raise SystemExit(main())
