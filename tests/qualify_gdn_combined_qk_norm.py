#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


RELATIVE_L2_LIMIT = 1.0e-5
SPEEDUP_MIN = 1.25
SAVING_MS_MIN = 0.02
SEQUENCE_STEPS_MIN = 500
EXPECTED_SEEDS = (20260715, 20260727)


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
            "schema": "bi100-gdn-combined-qk-norm-qualification-v1",
            "component_qualified": False,
            "production_promotion_authorized": False,
            "reasons": ["benchmark report is not an object"],
        }
    if value.get("schema") != "bi100-gdn-combined-qk-norm-v1":
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
    for seed in EXPECTED_SEEDS:
        row = fixed.get(str(seed))
        if not isinstance(row, dict):
            reasons.append(f"fixed seed {seed} row is missing")
            continue
        for name in ("normalized", "mapped"):
            metric = row.get(name)
            if not isinstance(metric, dict):
                reasons.append(f"fixed seed {seed} {name} row is missing")
                continue
            if metric.get("finite") is not True:
                reasons.append(f"fixed seed {seed} {name} is non-finite")
            if metric.get("exact") is not True:
                reasons.append(f"fixed seed {seed} {name} is not exact")
            relative_l2 = metric.get("relative_l2")
            if (
                not finite_number(relative_l2)
                or float(relative_l2) > RELATIVE_L2_LIMIT
            ):
                reasons.append(
                    f"fixed seed {seed} {name} relative L2 exceeds "
                    f"{RELATIVE_L2_LIMIT:g}")

    sequence = value.get("sequence")
    if not isinstance(sequence, dict):
        sequence = {}
        reasons.append("sequence row is missing")
    steps = sequence.get("steps")
    if not isinstance(steps, int) or steps < SEQUENCE_STEPS_MIN:
        reasons.append(
            f"sequence must contain at least {SEQUENCE_STEPS_MIN} steps")
    if sequence.get("finite_steps") != steps:
        reasons.append("sequence contains a non-finite step")
    for field in ("normalized_exact_steps", "mapped_exact_steps"):
        if sequence.get(field) != steps:
            reasons.append(f"sequence {field} is not exact")
    for field in (
        "normalized_relative_l2",
        "mapped_relative_l2",
        "max_normalized_relative_l2",
        "max_mapped_relative_l2",
    ):
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
    candidate = timings.get("candidate")
    if not isinstance(candidate, dict):
        candidate = {}
        reasons.append("candidate timing row is missing")
    speedup = candidate.get("speedup_vs_reference")
    saving_ms = candidate.get("saving_ms")
    if not finite_number(speedup) or float(speedup) < SPEEDUP_MIN:
        reasons.append(f"q/k stage speedup is below {SPEEDUP_MIN:g}x")
    if not finite_number(saving_ms) or float(saving_ms) < SAVING_MS_MIN:
        reasons.append(
            f"q/k stage saving is below {SAVING_MS_MIN:g} ms/layer")

    return {
        "schema": "bi100-gdn-combined-qk-norm-qualification-v1",
        "component_qualified": not reasons,
        "production_promotion_authorized": False,
        "limits": {
            "relative_l2": RELATIVE_L2_LIMIT,
            "speedup": SPEEDUP_MIN,
            "saving_ms": SAVING_MS_MIN,
            "sequence_steps": SEQUENCE_STEPS_MIN,
        },
        "observed": {
            "speedup": speedup,
            "saving_ms": saving_ms,
            "projected_30_layer_saving_ms": candidate.get(
                "projected_30_layer_saving_ms"),
            "normalized_relative_l2": sequence.get(
                "normalized_relative_l2"),
            "mapped_relative_l2": sequence.get("mapped_relative_l2"),
            "max_normalized_relative_l2": sequence.get(
                "max_normalized_relative_l2"),
            "max_mapped_relative_l2": sequence.get(
                "max_mapped_relative_l2"),
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
    return 0 if result["component_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
