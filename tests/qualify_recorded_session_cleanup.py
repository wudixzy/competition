#!/usr/bin/env python3
"""Require recorded-session recovery to have found no live experiment child."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


Json = dict[str, Any]
SCHEMA = "bi100-recorded-session-cleanup-qualification-v1"
RECOVERY_SCHEMA = "bi100-recorded-session-cleanup-v1"


def _load(path: Path) -> tuple[Json, bytes]:
    payload = path.read_bytes()
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("recovery report must be an object")
    return value, payload


def _positive_integer(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
    )


def qualify(recovery: Json, expected_identities: list[Path]) -> Json:
    reasons: list[str] = []
    expected = [str(path.resolve()) for path in expected_identities]
    if len(expected) != len(set(expected)):
        reasons.append("expected identity paths are not unique")
    actions = recovery.get("actions")
    privacy = recovery.get("privacy")
    if (
        recovery.get("schema") != RECOVERY_SCHEMA
        or recovery.get("version") != 1
        or recovery.get("qualified") is not True
        or recovery.get("reasons") != []
        or recovery.get("identity_count") != len(expected)
        or recovery.get("term_grace_s") != 60.0
        or recovery.get("kill_grace_s") != 20.0
        or recovery.get("complete_token_scan_required") is not True
        or not isinstance(actions, list)
        or len(actions) != len(expected)
        or not isinstance(privacy, dict)
        or privacy.get("command_lines_recorded") is not False
        or privacy.get("environment_recorded") is not False
    ):
        reasons.append("recorded-session recovery contract differs")

    if isinstance(actions, list):
        observed_identities: list[str] = []
        for index, action in enumerate(actions):
            label = f"action {index + 1}"
            if not isinstance(action, dict):
                reasons.append(f"{label} is malformed")
                continue
            identity = action.get("identity")
            if isinstance(identity, str):
                try:
                    observed_identities.append(
                        str(Path(identity).resolve()))
                except (OSError, RuntimeError, ValueError):
                    observed_identities.append("<invalid>")
                    reasons.append(f"{label} identity path is malformed")
            pid = action.get("pid")
            if (
                not isinstance(identity, str)
                or not _positive_integer(pid)
                or action.get("pgid") != pid
                or action.get("sid") != pid
                or action.get("term_sent") is not False
                or action.get("kill_sent") is not False
                or action.get("initial_live_count") != 0
                or action.get("initial_escaped_count") != 0
                or action.get("token_scan_error_count") != 0
                or action.get("final_live_count") != 0
                or action.get("outcome") != "already_quiescent"
            ):
                reasons.append(f"{label} required recovery")
        if observed_identities != expected:
            reasons.append("recorded-session identity order differs")

    return {
        "schema": SCHEMA,
        "version": 1,
        "qualified": not reasons,
        "reasons": reasons,
        "expected_identity_count": len(expected),
        "observed_identity_count": (
            len(actions) if isinstance(actions, list) else None
        ),
        "term_grace_s": recovery.get("term_grace_s"),
        "kill_grace_s": recovery.get("kill_grace_s"),
        "emergency_recovery_used": any(
            isinstance(action, dict)
            and (
                action.get("term_sent") is True
                or action.get("kill_sent") is True
            )
            for action in actions
        ) if isinstance(actions, list) else None,
        "privacy": {
            "contains_session_token": False,
            "contains_environment": False,
            "contains_command_line": False,
        },
        "production_promotion_authorized": False,
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
    parser.add_argument("recovery", type=Path)
    parser.add_argument(
        "--expected-identity",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        recovery, payload = _load(args.recovery)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        recovery = {}
        report = qualify(recovery, args.expected_identity)
        report["reasons"].insert(
            0, f"recovery report is invalid: {type(exc).__name__}")
        report["qualified"] = False
        report["input_sha256"] = None
    else:
        report = qualify(recovery, args.expected_identity)
        report["input_sha256"] = hashlib.sha256(
            payload).hexdigest()
    _atomic_write(args.out, report)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
