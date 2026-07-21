#!/usr/bin/env python3
"""Compare two CPU KV pressure harness reports without exposing messages."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any


Json = dict[str, Any]
INPUT_SCHEMA = "bi100-cpu-kv-offload-pressure-api-v1"
SCHEMA = "bi100-cpu-kv-pressure-ab-v1"
VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IGNORED_PARAMS = frozenset(("mode", "json_out"))
_TARGET_COLD = "target_cold"
_AFTER_PRESSURE = "target_after_pressure"


class ComparisonError(ValueError):
    """Raised for an unreadable input; valid reports use qualified=false."""


def _finite(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _int(value: Any, minimum: int | None = None) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)
            and (minimum is None or value >= minimum))


def _load(path: Path) -> Json:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ComparisonError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ComparisonError(f"{path} must contain a JSON object")
    return value


def _request_map(report: Json, label: str, reasons: list[str]) -> list[Json]:
    requests = report.get("requests")
    if not isinstance(requests, list) or not requests:
        reasons.append(f"{label} requests must be a non-empty list")
        return []
    result: list[Json] = []
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            reasons.append(f"{label} request {index} is not an object")
        else:
            result.append(request)
    return result


def _validate_report(report: Any, label: str, reasons: list[str]) -> tuple[list[Json], Json]:
    if not isinstance(report, dict):
        reasons.append(f"{label} report is not an object")
        return [], {}
    if report.get("schema") != INPUT_SCHEMA:
        reasons.append(f"{label} schema is invalid")
    if report.get("version") != VERSION:
        reasons.append(f"{label} version is invalid")
    if report.get("qualified") is not True:
        reasons.append(f"{label} is not qualified")
    validation = report.get("validation")
    if (not isinstance(validation, dict)
            or validation.get("qualified") is not True
            or validation.get("reasons") not in ([], None)):
        reasons.append(f"{label} validation is not qualified")
    params = report.get("params")
    if not isinstance(params, dict):
        reasons.append(f"{label} params are missing")
        params = {}
    if params.get("mode") not in ("control", "candidate"):
        reasons.append(f"{label} mode is invalid")
    expected = "control" if label == "control" else "candidate"
    if params.get("mode") != expected:
        reasons.append(f"{label} mode must be {expected}")
    for field in ("max_control_cached", "min_candidate_cached"):
        if not _int(params.get(field), 0):
            reasons.append(f"{label} {field} is invalid")
    for field in ("target_prompt_tokens", "pressure_prompt_tokens",
                  "pressure_count", "max_tokens", "block_size"):
        if not _int(params.get(field), 1):
            reasons.append(f"{label} {field} is invalid")
    if not isinstance(params.get("run_id"), str) or not params["run_id"]:
        reasons.append(f"{label} run_id is invalid")
    if not _finite(params.get("timeout_s")) or params["timeout_s"] <= 0:
        reasons.append(f"{label} timeout_s is invalid")
    return _request_map(report, label, reasons), params


def _validate_request(request: Json, label: str, reasons: list[str]) -> bool:
    name = request.get("name", "<unnamed>")
    if request.get("status") != "ok":
        reasons.append(f"{label} request {name} is not ok")
        return False
    expected = request.get("expected_prompt_tokens")
    if not _int(expected, 0):
        reasons.append(f"{label} request {name} expected tokens are invalid")
    summary = request.get("summary")
    if not isinstance(summary, dict):
        reasons.append(f"{label} request {name} summary is missing")
        return False
    for field in ("prompt_tokens", "cached_tokens", "completion_tokens"):
        if not _int(summary.get(field), 0):
            reasons.append(f"{label} request {name} {field} is invalid")
    if _int(expected, 0) and summary.get("prompt_tokens") != expected:
        reasons.append(f"{label} request {name} prompt tokens mismatch")
    if not _finite(summary.get("elapsed_s")) or summary.get("elapsed_s") <= 0:
        reasons.append(f"{label} request {name} elapsed_s is not positive finite")
    if summary.get("finish_reason") not in ("stop", "length"):
        reasons.append(f"{label} request {name} finish_reason is invalid")
    if not isinstance(summary.get("message_sha256"), str) or not _SHA256.fullmatch(
            summary["message_sha256"]):
        reasons.append(f"{label} request {name} message_sha256 is invalid")
    return True


def _usable_summary(request: Json) -> bool:
    summary = request.get("summary")
    if not isinstance(summary, dict):
        return False
    return (_int(summary.get("cached_tokens"), 0)
            and _int(summary.get("completion_tokens"), 0)
            and _finite(summary.get("elapsed_s")))


def _params_match(control: Json, candidate: Json, reasons: list[str]) -> None:
    left = {key: value for key, value in control.items() if key not in _IGNORED_PARAMS}
    right = {key: value for key, value in candidate.items() if key not in _IGNORED_PARAMS}
    if left != right:
        reasons.append("params differ outside mode/json_out")


def _summary(request: Json) -> Json:
    value = request["summary"]
    return {
        "cached_tokens": value["cached_tokens"],
        "completion_tokens": value["completion_tokens"],
        "finish_reason": value["finish_reason"],
        "message_sha256": value["message_sha256"],
        "elapsed_s": value["elapsed_s"],
    }


def _comparison_row(control: Json, candidate: Json) -> Json:
    left = _summary(control)
    right = _summary(candidate)
    control_elapsed = left["elapsed_s"]
    ratio = (right["elapsed_s"] / control_elapsed
             if control_elapsed > 0 else None)
    return {
        "name": control["name"],
        "expected_prompt_tokens": control["expected_prompt_tokens"],
        "control": left,
        "candidate": right,
        "cached_tokens_delta": right["cached_tokens"] - left["cached_tokens"],
        "elapsed_delta_s": right["elapsed_s"] - control_elapsed,
        "elapsed_ratio": ratio,
    }


def compare(control: Any, candidate: Any) -> Json:
    reasons: list[str] = []
    control_requests, control_params = _validate_report(control, "control", reasons)
    candidate_requests, candidate_params = _validate_report(candidate, "candidate", reasons)
    _params_match(control_params, candidate_params, reasons)

    if len(control_requests) != len(candidate_requests):
        reasons.append("request counts differ")
    rows: list[Json] = []
    for index, (left, right) in enumerate(zip(control_requests, candidate_requests)):
        _validate_request(left, "control", reasons)
        _validate_request(right, "candidate", reasons)
        if left.get("name") != right.get("name"):
            reasons.append(f"request {index} names differ")
        if left.get("expected_prompt_tokens") != right.get("expected_prompt_tokens"):
            reasons.append(f"request {index} expected tokens differ")
        if _usable_summary(left) and _usable_summary(right):
            for field in ("completion_tokens", "finish_reason", "message_sha256"):
                if left["summary"].get(field) != right["summary"].get(field):
                    reasons.append(f"request {index} {field} differs")
            if left.get("name") == right.get("name"):
                rows.append(_comparison_row(left, right))

    names = [request.get("name") for request in control_requests]
    if any(not isinstance(name, str) for name in names):
        reasons.append("control request names are invalid")
    if len(names) != len({name for name in names if isinstance(name, str)}):
        reasons.append("control request names are not unique")
    candidate_names = [request.get("name") for request in candidate_requests]
    if any(not isinstance(name, str) for name in candidate_names):
        reasons.append("candidate request names are invalid")
    if len(candidate_names) != len({name for name in candidate_names if isinstance(name, str)}):
        reasons.append("candidate request names are not unique")

    pressure_count = control_params.get("pressure_count")
    target_tokens = control_params.get("target_prompt_tokens")
    pressure_tokens = control_params.get("pressure_prompt_tokens")
    if _int(pressure_count, 1) and _int(target_tokens, 1) and _int(pressure_tokens, 1):
        expected_requests = [
            (_TARGET_COLD, target_tokens),
            ("target_immediate_warm", target_tokens),
            *[(f"pressure_cold_{index:04d}", pressure_tokens)
              for index in range(pressure_count)],
            (_AFTER_PRESSURE, target_tokens),
            ("target_refreshed", target_tokens),
        ]
        actual_requests = [(item.get("name"), item.get("expected_prompt_tokens"))
                           for item in control_requests]
        if actual_requests != expected_requests:
            reasons.append("control request sequence is incomplete or invalid")
        actual_candidate = [(item.get("name"), item.get("expected_prompt_tokens"))
                            for item in candidate_requests]
        if actual_candidate != expected_requests:
            reasons.append("candidate request sequence is incomplete or invalid")

    for label, requests in (("control", control_requests), ("candidate", candidate_requests)):
        cold = next((item for item in requests if item.get("name") == _TARGET_COLD), None)
        if isinstance(cold, dict) and isinstance(cold.get("summary"), dict):
            if cold["summary"].get("cached_tokens") != 0:
                reasons.append(f"{label} target_cold cached_tokens is not zero")

    control_after = next((item for item in control_requests
                          if item.get("name") == _AFTER_PRESSURE), None)
    candidate_after = next((item for item in candidate_requests
                            if item.get("name") == _AFTER_PRESSURE), None)
    if isinstance(control_after, dict) and _usable_summary(control_after):
        if control_after["summary"]["cached_tokens"] > control_params.get("max_control_cached", -1):
            reasons.append("control target_after_pressure exceeds threshold")
    if isinstance(candidate_after, dict) and _usable_summary(candidate_after):
        if candidate_after["summary"]["cached_tokens"] < candidate_params.get("min_candidate_cached", 0):
            reasons.append("candidate target_after_pressure is below threshold")

    after = next((row for row in rows if row["name"] == _AFTER_PRESSURE), None)
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "qualified": not reasons,
        "reasons": reasons,
        "requests": rows,
        "after_pressure": after,
    }


def atomic_write(path: Path, report: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                     dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
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
    parser = argparse.ArgumentParser(description="Compare CPU KV pressure A/B reports")
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = compare(_load(args.control), _load(args.candidate))
    except ComparisonError as error:
        report = {"schema": SCHEMA, "version": VERSION, "qualified": False,
                  "reasons": [str(error)], "requests": [], "after_pressure": None}
    atomic_write(args.out, report)
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
