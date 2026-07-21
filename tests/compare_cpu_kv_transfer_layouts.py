#!/usr/bin/env python3
"""Compare fixed paged and contiguous BI100 KV transfer evidence."""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


PAGED_SCHEMA = "bi100-cpu-kv-offload-capability-v1"
CONTIGUOUS_SCHEMA = "bi100-cpu-kv-contiguous-transfer-v1"
SCHEMA = "bi100-cpu-kv-transfer-layout-comparison-v1"
VERSION = 1
TOKEN_COUNTS = (65_536, 131_072)
MIN_DIRECTION_SPEEDUP = 4.0


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _finite_positive(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def compare(paged: Any, contiguous: Any) -> dict[str, Any]:
    reasons: list[str] = []
    if not isinstance(paged, dict) or paged.get("schema") != PAGED_SCHEMA:
        reasons.append("paged evidence schema is invalid")
        paged = {}
    if (not isinstance(contiguous, dict)
            or contiguous.get("schema") != CONTIGUOUS_SCHEMA
            or contiguous.get("version") != VERSION):
        reasons.append("contiguous evidence schema/version is invalid")
        contiguous = {}
    if paged.get("mode") != "gate":
        reasons.append("paged evidence is not a gate run")
    if paged.get("decision", {}).get("qualified") is not True:
        reasons.append("paged transfer gate is not qualified")
    if contiguous.get("decision", {}).get("diagnostic_passed") is not True:
        reasons.append("contiguous transfer diagnostic did not pass")
    if paged.get("shape") != contiguous.get("shape"):
        reasons.append("transfer shapes differ")
    if paged.get("device_name") != contiguous.get("device_name"):
        reasons.append("device names differ")
    if paged.get("torch_version") != contiguous.get("torch_version"):
        reasons.append("torch versions differ")

    rows: list[dict[str, Any]] = []
    paged_results = paged.get("results", {})
    contiguous_results = contiguous.get("results", {})
    for token_count in TOKEN_COUNTS:
        key = str(token_count)
        left = paged_results.get(key) if isinstance(paged_results, dict) else None
        right = (contiguous_results.get(key)
                 if isinstance(contiguous_results, dict) else None)
        if not isinstance(left, dict) or not isinstance(right, dict):
            reasons.append(f"missing paired case {token_count}")
            continue
        if left.get("exact") is not True or right.get("exact") is not True:
            reasons.append(f"case {token_count} is not exact in both layouts")
        if left.get("bytes_per_direction") != right.get("bytes_per_direction"):
            reasons.append(f"case {token_count} byte counts differ")

        row: dict[str, Any] = {
            "token_count": token_count,
            "bytes_per_direction": left.get("bytes_per_direction"),
            "paged": {},
            "contiguous": {},
            "speedup": {},
        }
        for direction in ("d2h", "h2d"):
            field = f"{direction}_median_ms"
            left_ms = left.get(field)
            right_ms = right.get(field)
            if not _finite_positive(left_ms) or not _finite_positive(right_ms):
                reasons.append(f"case {token_count} has invalid {field}")
                continue
            speedup = left_ms / right_ms
            row["paged"][field] = left_ms
            row["contiguous"][field] = right_ms
            row["speedup"][direction] = speedup
            if speedup + 1e-12 < MIN_DIRECTION_SPEEDUP:
                reasons.append(
                    f"case {token_count} {direction} speedup {speedup:.6f} "
                    f"is below {MIN_DIRECTION_SPEEDUP:.1f}x")
        rows.append(row)

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "thresholds": {
            "token_counts": list(TOKEN_COUNTS),
            "minimum_direction_speedup": MIN_DIRECTION_SPEEDUP,
        },
        "cases": rows,
        "qualified": not reasons and len(rows) == len(TOKEN_COUNTS),
        "reasons": reasons,
    }


def atomic_write(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                     dir=path.parent)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare paged and contiguous CPU KV transfer evidence")
    parser.add_argument("--paged", type=Path, required=True)
    parser.add_argument("--contiguous", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare(_load(args.paged), _load(args.contiguous))
    atomic_write(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
