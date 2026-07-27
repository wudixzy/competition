#!/usr/bin/env python3
"""Qualify one direct-W13 plus vendor-W2 exact-tail MoE design."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


Json = dict[str, Any]
SCHEMA = "bi100-moe-hybrid-exact-tail-qualification-v1"
VERSION = 1
BENCHMARK_SCHEMA = "bi100-moe-direct-routed-v2"
RELATIVE_L2_LIMIT = 1.0e-5
SPEEDUP_MIN = 1.25
SAVING_MS_MIN = 0.02
SEQUENCE_STEPS = 500
EXPECTED_SHAPE = {
    "experts": 256,
    "top_k": 8,
    "hidden": 2048,
    "intermediate": 128,
    "dtype": "torch.float16",
}


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def qualify(report: Json) -> Json:
    reasons: list[str] = []
    if report.get("schema") != BENCHMARK_SCHEMA:
        reasons.append("benchmark schema mismatch")
    if report.get("shape") != EXPECTED_SHAPE:
        reasons.append("benchmark did not use TP4 rank-local target shape")
    config = report.get("config") or {}
    if config.get("sequence_steps") != SEQUENCE_STEPS:
        reasons.append("sequence length differs from fixed 500-step gate")
    capabilities = report.get("extension_capabilities") or {}
    if capabilities.get("w13") is not True:
        reasons.append("direct W13 extension is unavailable")

    numerics = report.get("numerics") or {}
    direct_w13 = numerics.get("direct_w13") or {}
    fixed = numerics.get("hybrid_exact_tail") or {}
    sequence = (report.get("sequence") or {}).get(
        "hybrid_exact_tail") or {}
    for label, row in (
            ("direct W13", direct_w13),
            ("hybrid fixed endpoint", fixed)):
        if row.get("finite") is not True:
            reasons.append(f"{label} produced non-finite output")
        value = row.get("relative_l2")
        if not _finite(value) or value > RELATIVE_L2_LIMIT:
            reasons.append(f"{label} relative L2 exceeds 1e-5")

    if sequence.get("steps") != SEQUENCE_STEPS:
        reasons.append("hybrid sequence does not contain 500 steps")
    if sequence.get("finite_steps") != SEQUENCE_STEPS:
        reasons.append("hybrid sequence contains non-finite output")
    for field in ("relative_l2", "max_step_relative_l2"):
        value = sequence.get(field)
        if not _finite(value) or value > RELATIVE_L2_LIMIT:
            reasons.append(f"hybrid sequence {field} exceeds 1e-5")

    timings = report.get("timings") or {}
    baseline_fixed = (timings.get("baseline_fixed") or {}).get("median_ms")
    candidate_fixed = (
        timings.get("hybrid_exact_tail_fixed") or {}).get("median_ms")
    baseline_routed = (timings.get("baseline_routed") or {}).get("median_ms")
    candidate_routed = (
        timings.get("hybrid_exact_tail_routed") or {}).get("median_ms")
    fixed_speedup = (
        baseline_fixed / candidate_fixed
        if _finite(baseline_fixed) and _finite(candidate_fixed)
        and candidate_fixed > 0 else None
    )
    routed_speedup = (
        baseline_routed / candidate_routed
        if _finite(baseline_routed) and _finite(candidate_routed)
        and candidate_routed > 0 else None
    )
    routed_saving_ms = (
        baseline_routed - candidate_routed
        if _finite(baseline_routed) and _finite(candidate_routed) else None
    )
    for label, value in (
            ("fixed speedup", fixed_speedup),
            ("routed speedup", routed_speedup)):
        if not _finite(value) or value < SPEEDUP_MIN:
            reasons.append(f"{label} is below 1.25x")
    if (not _finite(routed_saving_ms)
            or routed_saving_ms < SAVING_MS_MIN):
        reasons.append("routed saving is below 0.02 ms")

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "component_qualified": not reasons,
        "reasons": reasons,
        "limits": {
            "relative_l2": RELATIVE_L2_LIMIT,
            "speedup": SPEEDUP_MIN,
            "saving_ms": SAVING_MS_MIN,
            "sequence_steps": SEQUENCE_STEPS,
        },
        "observed": {
            "direct_w13_relative_l2": direct_w13.get("relative_l2"),
            "fixed_relative_l2": fixed.get("relative_l2"),
            "sequence_relative_l2": sequence.get("relative_l2"),
            "sequence_max_step_relative_l2": sequence.get(
                "max_step_relative_l2"),
            "sequence_exact_steps": sequence.get("exact_steps"),
            "fixed_speedup": fixed_speedup,
            "routed_speedup": routed_speedup,
            "routed_saving_ms": routed_saving_ms,
        },
        "design": {
            "w13": "direct_selected_expert",
            "activation": "reference_silu_and_mul",
            "w2": "vendor_bmm",
            "routed_reduction": "serial_float_exact_reduce",
        },
        "semantic_quality_evaluated": False,
        "full_model_evaluated": False,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = qualify(json.loads(args.report.read_text(encoding="utf-8")))
    _atomic_write(args.out, report)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report["component_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
