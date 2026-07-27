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
EXPECTED_SEEDS = (20260716, 20260727)


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
            "schema": "bi100-moe-pairwise-w13-qualification-v1",
            "qualified": False,
            "reasons": ["benchmark report is not an object"],
        }
    if value.get("schema") != "bi100-moe-pairwise-w13-v1":
        reasons.append("benchmark schema is invalid")
    config = value.get("config")
    if not isinstance(config, dict):
        config = {}
        reasons.append("benchmark config is missing")
    if config.get("fixed_seeds") != list(EXPECTED_SEEDS):
        reasons.append("fixed seed set differs from the gate")

    fixed = value.get("fixed")
    if not isinstance(fixed, dict):
        fixed = {}
        reasons.append("fixed numerical rows are missing")
    fixed_relative_l2 = {}
    for seed in EXPECTED_SEEDS:
        row = fixed.get(str(seed), {})
        pairwise = row.get("pairwise") if isinstance(row, dict) else None
        if not isinstance(pairwise, dict):
            reasons.append(f"pairwise fixed seed {seed} row is missing")
            continue
        if pairwise.get("finite") is not True:
            reasons.append(f"pairwise fixed seed {seed} is non-finite")
        metric = pairwise.get("relative_l2")
        fixed_relative_l2[str(seed)] = metric
        if not finite_number(metric) or float(metric) > RELATIVE_L2_LIMIT:
            reasons.append(
                f"pairwise fixed seed {seed} relative L2 exceeds "
                f"{RELATIVE_L2_LIMIT:g}")

    sequence = value.get("sequence", {}).get("pairwise")
    if not isinstance(sequence, dict):
        sequence = {}
        reasons.append("pairwise sequence row is missing")
    steps = sequence.get("steps")
    if not isinstance(steps, int) or steps < SEQUENCE_STEPS_MIN:
        reasons.append(
            f"sequence must contain at least {SEQUENCE_STEPS_MIN} steps")
    if sequence.get("finite_steps") != steps:
        reasons.append("pairwise sequence contains a non-finite step")
    for field in ("relative_l2", "max_step_relative_l2"):
        metric = sequence.get(field)
        if not finite_number(metric) or float(metric) > RELATIVE_L2_LIMIT:
            reasons.append(
                f"pairwise sequence {field} exceeds {RELATIVE_L2_LIMIT:g}")

    timings = value.get("timings")
    if not isinstance(timings, dict):
        timings = {}
        reasons.append("timing report is missing")
    fixed_speedup = (
        timings.get("pairwise_fixed", {}).get("speedup_vs_reference")
        if isinstance(timings.get("pairwise_fixed"), dict)
        else None
    )
    routed_speedup = (
        timings.get("pairwise_routed", {}).get("speedup_vs_reference")
        if isinstance(timings.get("pairwise_routed"), dict)
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
        "schema": "bi100-moe-pairwise-w13-qualification-v1",
        "qualified": not reasons,
        "limits": {
            "relative_l2": RELATIVE_L2_LIMIT,
            "fixed_speedup": FIXED_SPEEDUP_MIN,
            "routed_speedup": ROUTED_SPEEDUP_MIN,
            "sequence_steps": SEQUENCE_STEPS_MIN,
        },
        "observed": {
            "fixed_relative_l2": fixed_relative_l2,
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
