#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


INPUT_SCHEMA = "bi100-cpu-kv-offload-pressure-api-v1"
SCHEMA = "bi100-cpu-kv-layout-pressure-ab-v1"
MAX_AFTER_PRESSURE_RATIO = 0.8
MAX_WARM_REGRESSION_RATIO = 1.02
IGNORED_PARAMS = frozenset(("json_out",))
TARGET_TIMINGS = (
    "target_immediate_warm",
    "target_after_pressure",
    "target_refreshed",
)


def _finite_positive(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def _validate_report(value: Any, label: str,
                     reasons: list[str]) -> tuple[dict[str, Any], list[Any]]:
    if not isinstance(value, dict):
        reasons.append(f"{label} must be an object")
        return {}, []
    if value.get("schema") != INPUT_SCHEMA or value.get("version") != 1:
        reasons.append(f"{label} schema/version is invalid")
    if value.get("qualified") is not True:
        reasons.append(f"{label} retention gate is not qualified")
    validation = value.get("validation")
    if (not isinstance(validation, dict)
            or validation.get("qualified") is not True
            or validation.get("reasons") not in ([], None)):
        reasons.append(f"{label} validation is not qualified")
    params = value.get("params")
    if not isinstance(params, dict):
        reasons.append(f"{label} params are missing")
        params = {}
    if params.get("mode") != "candidate":
        reasons.append(f"{label} must use the retention candidate contract")
    requests = value.get("requests")
    if not isinstance(requests, list) or not requests:
        reasons.append(f"{label} requests are missing")
        requests = []
    return params, requests


def _request_map(requests: list[Any], label: str,
                 reasons: list[str]) -> dict[str, dict[str, Any]]:
    result = {}
    for index, value in enumerate(requests):
        if not isinstance(value, dict):
            reasons.append(f"{label} request {index} is not an object")
            continue
        name = value.get("name")
        summary = value.get("summary")
        if not isinstance(name, str) or not name or name in result:
            reasons.append(f"{label} request {index} has an invalid name")
            continue
        if value.get("status") != "ok" or not isinstance(summary, dict):
            reasons.append(f"{label} request {name} is not usable")
            continue
        result[name] = value
    return result


def _safe_summary(value: dict[str, Any]) -> dict[str, Any]:
    summary = value["summary"]
    return {
        "cached_tokens": summary.get("cached_tokens"),
        "completion_tokens": summary.get("completion_tokens"),
        "elapsed_s": summary.get("elapsed_s"),
        "finish_reason": summary.get("finish_reason"),
        "message_sha256": summary.get("message_sha256"),
        "prompt_tokens": summary.get("prompt_tokens"),
    }


def compare(paged: Any, block_major: Any) -> dict[str, Any]:
    reasons: list[str] = []
    paged_params, paged_requests = _validate_report(
        paged, "paged", reasons)
    candidate_params, candidate_requests = _validate_report(
        block_major, "block_major", reasons)
    left_params = {
        key: value for key, value in paged_params.items()
        if key not in IGNORED_PARAMS
    }
    right_params = {
        key: value for key, value in candidate_params.items()
        if key not in IGNORED_PARAMS
    }
    if left_params != right_params:
        reasons.append("request parameters differ outside json_out")

    left = _request_map(paged_requests, "paged", reasons)
    right = _request_map(candidate_requests, "block_major", reasons)
    left_order = [item.get("name") for item in paged_requests
                  if isinstance(item, dict)]
    right_order = [item.get("name") for item in candidate_requests
                   if isinstance(item, dict)]
    if left_order != right_order:
        reasons.append("request order differs")
    if set(left) != set(right):
        reasons.append("request sets differ")

    rows = []
    for name in left_order:
        if name not in left or name not in right:
            continue
        paged_summary = _safe_summary(left[name])
        candidate_summary = _safe_summary(right[name])
        for field in (
                "cached_tokens", "completion_tokens", "finish_reason",
                "message_sha256", "prompt_tokens"):
            if paged_summary.get(field) != candidate_summary.get(field):
                reasons.append(f"request {name} differs in {field}")
        paged_elapsed = paged_summary.get("elapsed_s")
        candidate_elapsed = candidate_summary.get("elapsed_s")
        ratio = None
        if (not _finite_positive(paged_elapsed)
                or not _finite_positive(candidate_elapsed)):
            reasons.append(f"request {name} has invalid elapsed_s")
        else:
            ratio = candidate_elapsed / paged_elapsed
        rows.append({
            "block_major": candidate_summary,
            "elapsed_ratio": ratio,
            "name": name,
            "paged": paged_summary,
        })

    by_name = {row["name"]: row for row in rows}
    for name in TARGET_TIMINGS:
        if name not in by_name:
            reasons.append(f"missing timing row {name}")
    after = by_name.get("target_after_pressure")
    if (after is not None and _finite_positive(after.get("elapsed_ratio"))
            and after["elapsed_ratio"] > MAX_AFTER_PRESSURE_RATIO + 1e-12):
        reasons.append(
            "target_after_pressure regression ratio "
            f"{after['elapsed_ratio']:.6f} exceeds "
            f"{MAX_AFTER_PRESSURE_RATIO:.2f}")
    for name in ("target_immediate_warm", "target_refreshed"):
        row = by_name.get(name)
        if (row is not None and _finite_positive(row.get("elapsed_ratio"))
                and row["elapsed_ratio"] > MAX_WARM_REGRESSION_RATIO + 1e-12):
            reasons.append(
                f"{name} regression ratio {row['elapsed_ratio']:.6f} "
                f"exceeds {MAX_WARM_REGRESSION_RATIO:.2f}")

    return {
        "qualified": not reasons,
        "reasons": reasons,
        "requests": rows,
        "schema": SCHEMA,
        "thresholds": {
            "maximum_after_pressure_ratio": MAX_AFTER_PRESSURE_RATIO,
            "maximum_warm_regression_ratio": MAX_WARM_REGRESSION_RATIO,
        },
        "version": 1,
    }


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_atomic(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
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
    parser.add_argument("--paged", type=Path, required=True)
    parser.add_argument("--block-major", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = compare(_load(args.paged), _load(args.block_major))
    _write_atomic(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
