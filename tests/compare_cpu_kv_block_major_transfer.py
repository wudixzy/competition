#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PAGED_SCHEMA = "bi100-cpu-kv-offload-capability-v1"
BLOCK_MAJOR_SCHEMA = "bi100-cpu-kv-block-major-transfer-v1"
SCHEMA = "bi100-cpu-kv-block-major-comparison-v1"
TOKEN_COUNTS = (65_536, 131_072)
MIN_DIRECTION_SPEEDUP = 4.0


def _finite_positive(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def compare(paged: Any, candidate: Any) -> dict[str, Any]:
    reasons = []
    if not isinstance(paged, dict) or paged.get("schema") != PAGED_SCHEMA:
        reasons.append("paged evidence schema is invalid")
        paged = {}
    if (not isinstance(candidate, dict)
            or candidate.get("schema") != BLOCK_MAJOR_SCHEMA
            or candidate.get("version") != 1):
        reasons.append("block-major evidence schema/version is invalid")
        candidate = {}
    if paged.get("mode") != "gate":
        reasons.append("paged evidence is not a gate run")
    if paged.get("decision", {}).get("qualified") is not True:
        reasons.append("paged transfer gate is not qualified")
    if candidate.get("decision", {}).get("diagnostic_passed") is not True:
        reasons.append("block-major transfer diagnostic did not pass")
    if candidate.get("reordered_mapping_exact") is not True:
        reasons.append("block-major reordered mapping is not exact")
    if candidate.get("reordered_mapping_blocks") != 513:
        reasons.append("block-major probe did not cover the 512/513 boundary")
    if candidate.get("worker_d2h_before_h2d") is not True:
        reasons.append("installed worker transfer order is invalid")
    for field in ("shape", "device_name", "torch_version"):
        if paged.get(field) != candidate.get(field):
            reasons.append(f"paired evidence differs in {field}")

    rows = []
    paged_results = paged.get("results", {})
    candidate_results = candidate.get("results", {})
    for token_count in TOKEN_COUNTS:
        key = str(token_count)
        control = paged_results.get(key) if isinstance(paged_results, dict) else None
        result = (
            candidate_results.get(key)
            if isinstance(candidate_results, dict) else None)
        if not isinstance(control, dict) or not isinstance(result, dict):
            reasons.append(f"missing paired case {token_count}")
            continue
        if control.get("exact") is not True or result.get("exact") is not True:
            reasons.append(f"case {token_count} is not exact in both paths")
        if control.get("bytes_per_direction") != result.get("bytes_per_direction"):
            reasons.append(f"case {token_count} byte counts differ")
        row = {
            "bytes_per_direction": control.get("bytes_per_direction"),
            "candidate": {},
            "paged": {},
            "speedup": {},
            "token_count": token_count,
        }
        for direction in ("d2h", "h2d"):
            field = f"{direction}_median_ms"
            control_ms = control.get(field)
            candidate_ms = result.get(field)
            if not _finite_positive(control_ms) or not _finite_positive(candidate_ms):
                reasons.append(f"case {token_count} has invalid {field}")
                continue
            speedup = control_ms / candidate_ms
            row["paged"][field] = control_ms
            row["candidate"][field] = candidate_ms
            row["speedup"][direction] = speedup
            if speedup + 1e-12 < MIN_DIRECTION_SPEEDUP:
                reasons.append(
                    f"case {token_count} {direction} speedup {speedup:.6f} "
                    f"is below {MIN_DIRECTION_SPEEDUP:.1f}x")
        rows.append(row)

    return {
        "cases": rows,
        "qualified": not reasons and len(rows) == len(TOKEN_COUNTS),
        "reasons": reasons,
        "schema": SCHEMA,
        "thresholds": {
            "minimum_direction_speedup": MIN_DIRECTION_SPEEDUP,
            "token_counts": list(TOKEN_COUNTS),
        },
        "version": 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paged", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = compare(
        json.loads(args.paged.read_text(encoding="utf-8")),
        json.loads(args.candidate.read_text(encoding="utf-8")),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
