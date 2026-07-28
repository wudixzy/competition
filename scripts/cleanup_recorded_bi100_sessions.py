#!/usr/bin/env python3
"""Recover only service process groups attested by this experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import tempfile
import time
from typing import Any, Callable


Json = dict[str, Any]
SCHEMA = "bi100-recorded-session-cleanup-v1"
IDENTITY_SCHEMA = "bi100-process-session-v1"


class IncompleteTokenScan(RuntimeError):
    pass


def _read_stat(path: Path) -> Json:
    value = path.read_text(encoding="ascii")
    closing = value.rfind(")")
    if closing < 0:
        raise ValueError("malformed process stat")
    pid = int(value[:value.index(" ")])
    fields = value[closing + 2:].split()
    return {
        "pid": pid,
        "state": fields[0],
        "pgid": int(fields[2]),
        "sid": int(fields[3]),
        "starttime_ticks": int(fields[19]),
    }


def _group_members(proc_root: Path, pgid: int) -> list[Json]:
    members: list[Json] = []
    for path in proc_root.iterdir():
        if not path.name.isdigit():
            continue
        try:
            row = _read_stat(path / "stat")
        except (FileNotFoundError, ProcessLookupError):
            continue
        if row["pgid"] == pgid:
            members.append(row)
    return sorted(members, key=lambda row: row["pid"])


def _live_members(proc_root: Path, pgid: int) -> list[Json]:
    return [
        row for row in _group_members(proc_root, pgid)
        if row["state"] != "Z"
    ]


def _has_token(proc_root: Path, pid: int, expected_token: bytes) -> bool:
    environment = (
        proc_root / str(pid) / "environ"
    ).read_bytes().split(b"\0")
    return expected_token in environment


def _token_members(
    proc_root: Path,
    expected_token: bytes,
) -> tuple[list[Json], int]:
    members: list[Json] = []
    scan_errors = 0
    for path in proc_root.iterdir():
        if not path.name.isdigit():
            continue
        try:
            row = _read_stat(path / "stat")
            if row["state"] == "Z":
                continue
            if _has_token(proc_root, row["pid"], expected_token):
                members.append(row)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError:
            scan_errors += 1
            continue
    return sorted(members, key=lambda row: row["pid"]), scan_errors


def _load_identity(path: Path) -> Json:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("identity must be an object")
    pid = value.get("pid")
    if (
        value.get("schema") != IDENTITY_SCHEMA
        or value.get("version") != 1
        or not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 1
        or value.get("pgid") != pid
        or value.get("sid") != pid
        or not isinstance(value.get("starttime_ticks"), int)
        or isinstance(value.get("starttime_ticks"), bool)
        or value.get("starttime_ticks") <= 0
        or not isinstance(value.get("session_token"), str)
        or len(value.get("session_token")) != 32
        or any(character not in "0123456789abcdef"
               for character in value.get("session_token"))
    ):
        raise ValueError("identity contract differs")
    return value


def _wait_token_quiescent(
    proc_root: Path,
    expected_token: bytes,
    timeout_s: float,
    *,
    require_complete_scan: bool,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Json]:
    deadline = monotonic() + timeout_s
    while True:
        live, scan_errors = _token_members(proc_root, expected_token)
        if require_complete_scan and scan_errors:
            raise IncompleteTokenScan(
                f"token process scan had {scan_errors} errors")
        if not live:
            return []
        remaining = deadline - monotonic()
        if remaining <= 0:
            return live
        sleep(min(0.25, remaining))


def recover(
    identity_paths: list[Path],
    *,
    proc_root: Path = Path("/proc"),
    term_grace_s: float = 60.0,
    kill_grace_s: float = 20.0,
    require_complete_token_scan: bool = True,
) -> Json:
    if term_grace_s < 0 or kill_grace_s < 0:
        raise ValueError("cleanup grace periods must be non-negative")
    reasons: list[str] = []
    actions: list[Json] = []
    seen_pgids: set[int] = set()
    own_pgid = os.getpgrp()
    own_sid = os.getsid(0)

    for identity_path in identity_paths:
        try:
            identity = _load_identity(identity_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            reasons.append(
                f"{identity_path}: invalid identity: {type(exc).__name__}")
            continue
        pid = identity["pid"]
        pgid = identity["pgid"]
        sid = identity["sid"]
        action: Json = {
            "identity": str(identity_path),
            "pid": pid,
            "pgid": pgid,
            "sid": sid,
            "term_sent": False,
            "kill_sent": False,
            "initial_live_count": 0,
            "initial_zombie_count": 0,
            "initial_escaped_count": 0,
            "token_scan_error_count": 0,
            "final_live_count": 0,
            "final_zombie_count": 0,
        }
        actions.append(action)
        if pgid in seen_pgids:
            reasons.append(f"{identity_path}: duplicate process group {pgid}")
            continue
        seen_pgids.add(pgid)
        if pgid == own_pgid or sid == own_sid:
            reasons.append(f"{identity_path}: refuses cleanup of own session")
            continue

        try:
            group_members = _group_members(proc_root, pgid)
            group_live = [
                row for row in group_members if row["state"] != "Z"]
        except OSError as exc:
            reasons.append(
                f"{identity_path}: process scan failed: {type(exc).__name__}")
            continue
        expected_token = (
            "BI100_PROCESS_SESSION_TOKEN="
            f"{identity['session_token']}"
        ).encode("ascii")
        try:
            token_live, token_scan_errors = _token_members(
                proc_root, expected_token)
        except OSError as exc:
            reasons.append(
                f"{identity_path}: token process scan failed: "
                f"{type(exc).__name__}")
            continue
        action["token_scan_error_count"] = token_scan_errors
        action["initial_zombie_count"] = sum(
            row["state"] == "Z" for row in group_members)
        if require_complete_token_scan and token_scan_errors:
            reasons.append(
                f"{identity_path}: token process scan was incomplete")
            continue
        action["initial_live_count"] = len(token_live)
        action["initial_escaped_count"] = sum(
            row["pgid"] != pgid for row in token_live)
        if any(row["sid"] != sid for row in group_live):
            reasons.append(f"{identity_path}: process session identity differs")
            continue
        token_mismatch = False
        token_scan_error = False
        for row in group_live:
            try:
                matches = _has_token(
                    proc_root, row["pid"], expected_token)
            except (FileNotFoundError, ProcessLookupError):
                token_mismatch = True
                break
            except OSError as exc:
                reasons.append(
                    f"{identity_path}: environment scan failed: "
                    f"{type(exc).__name__}")
                token_mismatch = True
                token_scan_error = True
                break
            if not matches:
                token_mismatch = True
                break
        if token_mismatch:
            if not token_scan_error:
                reasons.append(
                    f"{identity_path}: process session token differs")
            continue
        if not token_live:
            action["outcome"] = "already_quiescent"
            continue
        leader = next(
            (row for row in token_live if row["pid"] == pid), None)
        if (
            leader is not None
            and leader["starttime_ticks"] != identity["starttime_ticks"]
        ):
            reasons.append(f"{identity_path}: leader starttime differs")
            continue

        try:
            if group_live:
                os.killpg(pgid, signal.SIGTERM)
            for row in token_live:
                if row["pgid"] == pgid:
                    continue
                try:
                    current = _read_stat(
                        proc_root / str(row["pid"]) / "stat")
                    if (
                        current["starttime_ticks"] == row["starttime_ticks"]
                        and _has_token(
                            proc_root, row["pid"], expected_token)
                    ):
                        os.kill(row["pid"], signal.SIGTERM)
                except (FileNotFoundError, ProcessLookupError):
                    continue
            action["term_sent"] = True
        except ProcessLookupError:
            pass
        except OSError as exc:
            reasons.append(
                f"{identity_path}: SIGTERM failed: {type(exc).__name__}")
            continue
        try:
            live = _wait_token_quiescent(
                proc_root,
                expected_token,
                term_grace_s,
                require_complete_scan=require_complete_token_scan,
            )
        except IncompleteTokenScan:
            reasons.append(
                f"{identity_path}: token process rescan was incomplete")
            continue
        if live:
            try:
                for row in live:
                    try:
                        current = _read_stat(
                            proc_root / str(row["pid"]) / "stat")
                        if (
                            current["starttime_ticks"]
                            == row["starttime_ticks"]
                            and _has_token(
                                proc_root, row["pid"], expected_token)
                        ):
                            os.kill(row["pid"], signal.SIGKILL)
                    except (FileNotFoundError, ProcessLookupError):
                        continue
                action["kill_sent"] = True
            except ProcessLookupError:
                pass
            except OSError as exc:
                reasons.append(
                    f"{identity_path}: SIGKILL failed: {type(exc).__name__}")
                continue
            try:
                live = _wait_token_quiescent(
                    proc_root,
                    expected_token,
                    kill_grace_s,
                    require_complete_scan=require_complete_token_scan,
                )
            except IncompleteTokenScan:
                reasons.append(
                    f"{identity_path}: token process rescan was incomplete")
                continue
        action["final_live_count"] = len(live)
        try:
            action["final_zombie_count"] = sum(
                row["state"] == "Z"
                for row in _group_members(proc_root, pgid)
            )
        except OSError as exc:
            reasons.append(
                f"{identity_path}: final process scan failed: "
                f"{type(exc).__name__}")
            continue
        action["outcome"] = "quiescent" if not live else "survived"
        if live:
            reasons.append(f"{identity_path}: live process group survived")

    return {
        "schema": SCHEMA,
        "version": 1,
        "qualified": not reasons,
        "reasons": reasons,
        "identity_count": len(identity_paths),
        "actions": actions,
        "term_grace_s": term_grace_s,
        "kill_grace_s": kill_grace_s,
        "complete_token_scan_required": require_complete_token_scan,
        "privacy": {
            "command_lines_recorded": False,
            "environment_recorded": False,
        },
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
    parser.add_argument("--identity", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = recover(args.identity)
    _atomic_write(args.out, report)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
