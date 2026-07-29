#!/usr/bin/env python3
"""Append and summarize privacy-safe experiment stage timing records."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any


EVENT_SCHEMA = "bi100-experiment-timeline-event-v1"
REPORT_SCHEMA = "bi100-experiment-timeline-report-v1"
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}")


def _validate_name(value: str, label: str) -> str:
    if NAME_RE.fullmatch(value) is None:
        raise ValueError(f"{label} has an invalid format")
    return value


def append_event(
    path: Path,
    *,
    run_id: str,
    stage: str,
    event: str,
    status: str | None = None,
    wall_time_ns: int | None = None,
    monotonic_ns: int | None = None,
    pid: int | None = None,
) -> dict[str, Any]:
    _validate_name(run_id, "run_id")
    _validate_name(stage, "stage")
    if event not in {"start", "end"}:
        raise ValueError("event must be start or end")
    if event == "start" and status is not None:
        raise ValueError("start events cannot carry a status")
    if event == "end" and status not in {"pass", "fail", "skip"}:
        raise ValueError("end events require pass, fail or skip status")
    path = path.resolve()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_APPEND | os.O_CREAT,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "a", encoding="ascii") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            record = {
                "schema": EVENT_SCHEMA,
                "version": 1,
                "run_id": run_id,
                "stage": stage,
                "event": event,
                "status": status,
                "wall_time_ns": (
                    time.time_ns() if wall_time_ns is None else wall_time_ns),
                "monotonic_ns": (
                    time.monotonic_ns()
                    if monotonic_ns is None else monotonic_ns),
                "pid": os.getpid() if pid is None else pid,
            }
            stream.write(
                json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        if not path.exists():
            try:
                os.close(descriptor)
            except OSError:
                pass
    return record


def _load_events(path: Path) -> list[dict[str, Any]]:
    events = []
    for line_number, line in enumerate(
            path.read_text(encoding="ascii").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"timeline line {line_number} is not JSON") from exc
        if (
            not isinstance(value, dict)
            or value.get("schema") != EVENT_SCHEMA
            or value.get("version") != 1
            or value.get("event") not in {"start", "end"}
            or not isinstance(value.get("monotonic_ns"), int)
            or not isinstance(value.get("wall_time_ns"), int)
            or not isinstance(value.get("pid"), int)
        ):
            raise ValueError(f"timeline line {line_number} is malformed")
        _validate_name(value.get("run_id", ""), "run_id")
        _validate_name(value.get("stage", ""), "stage")
        events.append(value)
    return events


def summarize(path: Path, *, expected_run_id: str | None = None) -> dict[str, Any]:
    events = _load_events(path)
    reasons: list[str] = []
    if not events:
        reasons.append("timeline contains no events")
    run_ids = sorted({event["run_id"] for event in events})
    if len(run_ids) != 1:
        reasons.append("timeline must contain exactly one run_id")
    run_id = run_ids[0] if len(run_ids) == 1 else None
    if expected_run_id is not None and run_id != expected_run_id:
        reasons.append("timeline run_id differs from expected identity")

    active: dict[tuple[str, int], dict[str, Any]] = {}
    occurrence_by_stage: dict[str, int] = {}
    completed = []
    for event in events:
        stage = event["stage"]
        if event["event"] == "start":
            occurrence = occurrence_by_stage.get(stage, 0)
            occurrence_by_stage[stage] = occurrence + 1
            key = (stage, occurrence)
            active[key] = event
            continue
        candidates = [
            key for key in active
            if key[0] == stage and active[key] is not None
        ]
        if not candidates:
            reasons.append(f"stage {stage} ended without a start")
            continue
        key = min(candidates, key=lambda item: item[1])
        start = active.pop(key)
        elapsed_ns = event["monotonic_ns"] - start["monotonic_ns"]
        if elapsed_ns < 0:
            reasons.append(f"stage {stage} has negative elapsed time")
            continue
        completed.append({
            "stage": stage,
            "occurrence": key[1],
            "status": event.get("status"),
            "started_wall_time_ns": start["wall_time_ns"],
            "finished_wall_time_ns": event["wall_time_ns"],
            "started_monotonic_ns": start["monotonic_ns"],
            "finished_monotonic_ns": event["monotonic_ns"],
            "elapsed_s": elapsed_ns / 1_000_000_000,
        })
    for stage, _ in sorted(active):
        reasons.append(f"stage {stage} has no end event")
    if any(row["status"] == "fail" for row in completed):
        reasons.append("one or more stages failed")

    if completed:
        first = min(row["started_monotonic_ns"] for row in completed)
        last = max(row["finished_monotonic_ns"] for row in completed)
        wall_span_s = max(0.0, (last - first) / 1_000_000_000)
    else:
        wall_span_s = 0.0
    summed_stage_s = sum(row["elapsed_s"] for row in completed)
    parallelism = summed_stage_s / wall_span_s if wall_span_s > 0 else None
    if parallelism is not None and not math.isfinite(parallelism):
        reasons.append("timeline parallelism is non-finite")

    return {
        "schema": REPORT_SCHEMA,
        "version": 1,
        "qualified": not reasons,
        "reasons": reasons,
        "run_id": run_id,
        "event_count": len(events),
        "completed_stage_count": len(completed),
        "wall_span_s": wall_span_s,
        "summed_stage_s": summed_stage_s,
        "effective_parallelism": parallelism,
        "stages": completed,
        "privacy": {
            "contains_prompts": False,
            "contains_model_outputs": False,
            "contains_tensor_values": False,
            "contains_credentials": False
        },
    }


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    mark = subparsers.add_parser("mark")
    mark.add_argument("--timeline", type=Path, required=True)
    mark.add_argument("--run-id", required=True)
    mark.add_argument("--stage", required=True)
    mark.add_argument("--event", choices=("start", "end"), required=True)
    mark.add_argument("--status", choices=("pass", "fail", "skip"))
    report = subparsers.add_parser("summarize")
    report.add_argument("--timeline", type=Path, required=True)
    report.add_argument("--run-id")
    report.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "mark":
        value = append_event(
            args.timeline,
            run_id=args.run_id,
            stage=args.stage,
            event=args.event,
            status=args.status,
        )
        print(json.dumps(value, ensure_ascii=True, sort_keys=True))
        return 0
    value = summarize(args.timeline, expected_run_id=args.run_id)
    _atomic_write(args.out, value)
    print(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if value["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
