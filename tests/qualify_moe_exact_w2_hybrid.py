#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


RELATIVE_L2_LIMIT = 1.0e-5
FIXED_SPEEDUP_MIN = 1.5
ROUTED_SPEEDUP_MIN = 1.25
SEQUENCE_STEPS_MIN = 500
EXPECTED_SHAPE = {
    "experts": 256,
    "top_k": 8,
    "hidden": 2048,
    "intermediate": 128,
    "dtype": "torch.float16",
}


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def qualify(value: Any) -> dict[str, Any]:
    reasons: list[str] = []
    if not isinstance(value, dict):
        return {
            "schema": "bi100-moe-exact-w2-hybrid-qualification-v1",
            "qualified": False,
            "reasons": ["benchmark report is not an object"],
        }
    if value.get("schema") != "bi100-moe-exact-w2-hybrid-v1":
        reasons.append("benchmark schema is invalid")
    if value.get("shape") != EXPECTED_SHAPE:
        reasons.append("benchmark did not use the TP4 rank-local target shape")

    checks = value.get("checks")
    if not isinstance(checks, dict):
        checks = {}
        reasons.append("numerical checks are missing")
    if checks.get("selected_w2_exact") is not True:
        reasons.append("selected W2 gather is not byte-exact")
    for name in ("direct_w13", "hybrid"):
        row = checks.get(name)
        if not isinstance(row, dict):
            reasons.append(f"{name} numerical row is missing")
            continue
        if row.get("finite") is not True:
            reasons.append(f"{name} produced non-finite output")
        relative_l2 = row.get("relative_l2")
        if (
            not finite_number(relative_l2)
            or float(relative_l2) > RELATIVE_L2_LIMIT
        ):
            reasons.append(
                f"{name} relative L2 exceeds {RELATIVE_L2_LIMIT:g}")

    sequence = value.get("sequence")
    if not isinstance(sequence, dict):
        sequence = {}
        reasons.append("sequence report is missing")
    steps = sequence.get("steps")
    if not isinstance(steps, int) or steps < SEQUENCE_STEPS_MIN:
        reasons.append(
            f"sequence must contain at least {SEQUENCE_STEPS_MIN} steps")
    if sequence.get("finite_steps") != steps:
        reasons.append("sequence contains a non-finite step")
    for field in ("relative_l2", "max_step_relative_l2"):
        metric = sequence.get(field)
        if (
            not finite_number(metric)
            or float(metric) > RELATIVE_L2_LIMIT
        ):
            reasons.append(
                f"sequence {field} exceeds {RELATIVE_L2_LIMIT:g}")

    timings = value.get("timings")
    if not isinstance(timings, dict):
        timings = {}
        reasons.append("timing report is missing")
    fixed_speedup = (
        timings.get("hybrid_fixed", {}).get("speedup_vs_baseline")
        if isinstance(timings.get("hybrid_fixed"), dict)
        else None
    )
    routed_speedup = (
        timings.get("hybrid_routed", {}).get("speedup_vs_baseline")
        if isinstance(timings.get("hybrid_routed"), dict)
        else None
    )
    if (
        not finite_number(fixed_speedup)
        or float(fixed_speedup) < FIXED_SPEEDUP_MIN
    ):
        reasons.append(
            f"fixed speedup is below {FIXED_SPEEDUP_MIN:g}x")
    if (
        not finite_number(routed_speedup)
        or float(routed_speedup) < ROUTED_SPEEDUP_MIN
    ):
        reasons.append(
            f"routed speedup is below {ROUTED_SPEEDUP_MIN:g}x")

    return {
        "schema": "bi100-moe-exact-w2-hybrid-qualification-v1",
        "qualified": not reasons,
        "limits": {
            "relative_l2": RELATIVE_L2_LIMIT,
            "fixed_speedup": FIXED_SPEEDUP_MIN,
            "routed_speedup": ROUTED_SPEEDUP_MIN,
            "sequence_steps": SEQUENCE_STEPS_MIN,
        },
        "observed": {
            "direct_w13_relative_l2": (
                checks.get("direct_w13", {}).get("relative_l2")
                if isinstance(checks.get("direct_w13"), dict)
                else None
            ),
            "hybrid_relative_l2": (
                checks.get("hybrid", {}).get("relative_l2")
                if isinstance(checks.get("hybrid"), dict)
                else None
            ),
            "sequence_relative_l2": sequence.get("relative_l2"),
            "sequence_max_step_relative_l2": sequence.get(
                "max_step_relative_l2"),
            "fixed_speedup": fixed_speedup,
            "routed_speedup": routed_speedup,
        },
        "reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    value = json.loads(args.report.read_text(encoding="utf-8"))
    result = qualify(value)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
