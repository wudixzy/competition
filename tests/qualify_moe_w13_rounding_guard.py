#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "bi100-moe-w13-rounding-guard-v1"
QUALIFICATION_SCHEMA = "bi100-moe-w13-rounding-guard-qualification-v1"
RELATIVE_L2_LIMIT = 1.0e-5
MAX_FLAGGED_FRACTION = 0.05
MAX_STEP_FLAGGED_FRACTION = 0.10
REQUIRED_SEEDS = (20260716, 20260727)
REQUIRED_SEQUENCE_STEPS = 500


def require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def require_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return value


def require_number(value: Any, field: str) -> float:
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value))):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


def require_count(value: Any, field: str, upper: int) -> int:
    count = require_int(value, field)
    if count < 0 or count > upper:
        raise ValueError(f"{field} must be between zero and {upper}")
    return count


def require_comparison(value: Any, field: str) -> dict[str, Any]:
    result = require_mapping(value, field)
    require_bool(result.get("exact"), f"{field}.exact")
    require_bool(result.get("finite"), f"{field}.finite")
    require_int(result.get("mismatch_count"), f"{field}.mismatch_count")
    for name in ("max_abs", "mean_abs", "relative_l2"):
        require_number(result.get(name), f"{field}.{name}")
    return result


def fraction_matches(value: float, numerator: int, denominator: int) -> bool:
    return math.isclose(
        value,
        numerator / max(denominator, 1),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )


def qualify(report: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if report.get("schema") != SCHEMA:
        reasons.append("report schema is invalid")
    if report.get("version") != 1:
        reasons.append("report version is invalid")
    if not isinstance(report.get("device"), str) or not report["device"]:
        reasons.append("device identity is missing")

    shape = require_mapping(report.get("shape"), "shape")
    expected_shape = {
        "experts": 256,
        "top_k": 8,
        "hidden": 2048,
        "intermediate": 128,
        "dtype": "torch.float16",
    }
    if shape != expected_shape:
        reasons.append("production TP4 rank-local shape contract changed")

    config = require_mapping(report.get("config"), "config")
    seeds = config.get("seeds")
    if seeds != list(REQUIRED_SEEDS):
        reasons.append("fixed seed contract changed")
    if require_int(
            config.get("sequence_steps_per_seed"),
            "config.sequence_steps_per_seed") != REQUIRED_SEQUENCE_STEPS:
        reasons.append("sequence step contract changed")
    if config.get("device") != "cuda:0":
        reasons.append("single-GPU logical device contract changed")
    if require_int(config.get("cpu_threads"), "config.cpu_threads") != 8:
        reasons.append("CPU correction thread contract changed")

    method = require_mapping(report.get("method"), "method")
    if method.get("flag_rule") != (
            "forward_fp16_differs_from_reverse_fp16"):
        reasons.append("rounding-risk flag rule changed")
    if method.get("correction_oracle") != (
            "float64_dot_rounded_to_fp16_for_flagged_rows"):
        reasons.append("correction oracle changed")
    if method.get("fixture_generation") != (
            "hidden_then_router_then_w13_then_sequence"):
        reasons.append("fixture generation order changed")
    if method.get("production_runtime_changed") is not False:
        reasons.append("diagnostic unexpectedly changed production runtime")

    fixtures = report.get("fixtures")
    if not isinstance(fixtures, list) or len(fixtures) != len(REQUIRED_SEEDS):
        raise ValueError("fixtures must contain exactly two records")
    observed_seeds = []
    fixture_results = []
    total_vendor_mismatches = 0
    total_flags = 0
    total_rows = 0
    for index, fixture_value in enumerate(fixtures):
        fixture = require_mapping(fixture_value, f"fixtures[{index}]")
        seed = require_int(fixture.get("seed"), f"fixtures[{index}].seed")
        observed_seeds.append(seed)
        fixed = require_mapping(
            fixture.get("fixed"), f"fixtures[{index}].fixed")
        sequence = require_mapping(
            fixture.get("sequence"), f"fixtures[{index}].sequence")
        prefix = f"seed {seed}"
        fixture_reasons = []

        if not require_bool(fixed.get("finite"), f"{prefix}.fixed.finite"):
            fixture_reasons.append("fixed sums contain NaN/Inf")
        if not require_bool(
                fixed.get("production_forward_exact"),
                f"{prefix}.fixed.production_forward_exact"):
            fixture_reasons.append(
                "diagnostic forward order differs from production W13")
        fixed_rows = require_int(
            fixed.get("rows"), f"{prefix}.fixed.rows")
        if fixed_rows != 2048:
            fixture_reasons.append("fixed row count differs")
        fixed_flags = require_count(
            fixed.get("flags"), f"{prefix}.fixed.flags", fixed_rows)
        fixed_vendor_mismatches = require_count(
            fixed.get("vendor_mismatches"),
            f"{prefix}.fixed.vendor_mismatches",
            fixed_rows,
        )
        fixed_flagged_vendor_mismatches = require_count(
            fixed.get("flagged_vendor_mismatches"),
            f"{prefix}.fixed.flagged_vendor_mismatches",
            fixed_rows,
        )
        fixed_missed = require_count(
            fixed.get("missed_vendor_mismatches"),
            f"{prefix}.fixed.missed_vendor_mismatches",
            fixed_rows,
        )
        fixed_false_positive = require_count(
            fixed.get("false_positive_flags"),
            f"{prefix}.fixed.false_positive_flags",
            fixed_rows,
        )
        fixed_exact_flag_mismatches = require_count(
            fixed.get("exact_flag_mismatches"),
            f"{prefix}.fixed.exact_flag_mismatches",
            fixed_rows,
        )
        if (
            fixed_flagged_vendor_mismatches + fixed_missed
            != fixed_vendor_mismatches
            or fixed_flagged_vendor_mismatches + fixed_false_positive
            != fixed_flags
            or fixed_exact_flag_mismatches > fixed_flags
        ):
            fixture_reasons.append("fixed mismatch counters are inconsistent")
        fixed_flagged_fraction = require_number(
            fixed.get("flagged_fraction"),
            f"{prefix}.fixed.flagged_fraction")
        if not fraction_matches(
                fixed_flagged_fraction, fixed_flags, fixed_rows):
            fixture_reasons.append("fixed flagged fraction is inconsistent")
        if fixed_flagged_fraction > MAX_FLAGGED_FRACTION:
            fixture_reasons.append("fixed flagged fraction exceeds 5%")
        if fixed_missed != 0:
            fixture_reasons.append(
                "reverse-order flag misses a fixed vendor mismatch")
        if fixed_exact_flag_mismatches != 0:
            fixture_reasons.append(
                "float64 correction disagrees with vendor on a fixed flag")

        fixed_direct = require_comparison(
            fixed.get("direct"), f"{prefix}.fixed.direct")
        fixed_exact = require_comparison(
            fixed.get("exact_half"), f"{prefix}.fixed.exact_half")
        fixed_corrected = require_comparison(
            fixed.get("corrected"), f"{prefix}.fixed.corrected")
        require_comparison(
            fixed.get("reverse"), f"{prefix}.fixed.reverse")
        if fixed_direct["mismatch_count"] != fixed_vendor_mismatches:
            fixture_reasons.append(
                "fixed direct comparison count is inconsistent")
        if not fixed_exact["finite"] or not fixed_corrected["finite"]:
            fixture_reasons.append(
                "fixed correction comparison contains NaN/Inf")
        if (
            not fixed_corrected["exact"]
            or fixed_corrected["mismatch_count"] != 0
        ):
            fixture_reasons.append(
                "fixed corrected output is not vendor-exact")
        if require_number(
                fixed_exact.get("relative_l2"),
                f"{prefix}.fixed.exact_half.relative_l2"
        ) > RELATIVE_L2_LIMIT:
            fixture_reasons.append(
                "fixed float64-rounded output exceeds relative L2 limit")
        if require_number(
                fixed_corrected.get("relative_l2"),
                f"{prefix}.fixed.corrected.relative_l2"
        ) > RELATIVE_L2_LIMIT:
            fixture_reasons.append(
                "fixed corrected output exceeds relative L2 limit")

        steps = require_int(
            sequence.get("steps"), f"{prefix}.sequence.steps")
        finite_steps = require_int(
            sequence.get("finite_steps"),
            f"{prefix}.sequence.finite_steps")
        if steps != REQUIRED_SEQUENCE_STEPS or finite_steps != steps:
            fixture_reasons.append("sequence is incomplete or non-finite")
        sequence_rows = require_int(
            sequence.get("rows"), f"{prefix}.sequence.rows")
        if sequence_rows != REQUIRED_SEQUENCE_STEPS * 2048:
            fixture_reasons.append("sequence row count differs")
        sequence_flags = require_count(
            sequence.get("flags"),
            f"{prefix}.sequence.flags",
            sequence_rows,
        )
        sequence_vendor_mismatches = require_count(
            sequence.get("vendor_mismatches"),
            f"{prefix}.sequence.vendor_mismatches",
            sequence_rows,
        )
        sequence_flagged_vendor_mismatches = require_count(
            sequence.get("flagged_vendor_mismatches"),
            f"{prefix}.sequence.flagged_vendor_mismatches",
            sequence_rows,
        )
        sequence_missed = require_count(
            sequence.get("missed_vendor_mismatches"),
            f"{prefix}.sequence.missed_vendor_mismatches",
            sequence_rows,
        )
        sequence_false_positive = require_count(
            sequence.get("false_positive_flags"),
            f"{prefix}.sequence.false_positive_flags",
            sequence_rows,
        )
        sequence_exact_flag_mismatches = require_count(
            sequence.get("exact_flag_mismatches"),
            f"{prefix}.sequence.exact_flag_mismatches",
            sequence_rows,
        )
        if (
            sequence_flagged_vendor_mismatches + sequence_missed
            != sequence_vendor_mismatches
            or sequence_flagged_vendor_mismatches + sequence_false_positive
            != sequence_flags
            or sequence_exact_flag_mismatches > sequence_flags
        ):
            fixture_reasons.append(
                "sequence mismatch counters are inconsistent")
        sequence_flagged_fraction = require_number(
            sequence.get("flagged_fraction"),
            f"{prefix}.sequence.flagged_fraction")
        if not fraction_matches(
                sequence_flagged_fraction, sequence_flags, sequence_rows):
            fixture_reasons.append(
                "sequence flagged fraction is inconsistent")
        max_step_flagged_fraction = require_number(
            sequence.get("max_step_flagged_fraction"),
            f"{prefix}.sequence.max_step_flagged_fraction")
        if sequence_flagged_fraction > MAX_FLAGGED_FRACTION:
            fixture_reasons.append("sequence flagged fraction exceeds 5%")
        if max_step_flagged_fraction > MAX_STEP_FLAGGED_FRACTION:
            fixture_reasons.append(
                "one sequence step flags more than 10% of rows")
        if sequence_missed != 0:
            fixture_reasons.append(
                "reverse-order flag misses a sequence vendor mismatch")
        if sequence_exact_flag_mismatches != 0:
            fixture_reasons.append(
                "float64 correction disagrees with vendor on a sequence flag")
        if require_number(
                sequence.get("max_exact_step_relative_l2"),
                f"{prefix}.sequence.max_exact_step_relative_l2"
        ) > RELATIVE_L2_LIMIT:
            fixture_reasons.append(
                "sequence float64-rounded output exceeds relative L2 limit")
        if require_number(
                sequence.get("max_corrected_step_relative_l2"),
                f"{prefix}.sequence.max_corrected_step_relative_l2"
        ) > RELATIVE_L2_LIMIT:
            fixture_reasons.append(
                "sequence corrected output exceeds relative L2 limit")

        total_vendor_mismatches += fixed_vendor_mismatches
        total_vendor_mismatches += sequence_vendor_mismatches
        total_flags += fixed_flags
        total_flags += sequence_flags
        total_rows += fixed_rows
        total_rows += sequence_rows

        reasons.extend(f"{prefix}: {reason}" for reason in fixture_reasons)
        fixture_results.append({
            "seed": seed,
            "qualified": not fixture_reasons,
            "reasons": fixture_reasons,
            "fixed_flagged_fraction": fixed_flagged_fraction,
            "sequence_flagged_fraction": sequence_flagged_fraction,
            "max_step_flagged_fraction": max_step_flagged_fraction,
        })

    if observed_seeds != list(REQUIRED_SEEDS):
        reasons.append("fixture order or seed identity changed")
    if total_vendor_mismatches <= 0:
        reasons.append(
            "probe did not reproduce the production W13 numerical gap")

    qualified = not reasons
    return {
        "schema": QUALIFICATION_SCHEMA,
        "version": 1,
        "qualified": qualified,
        "reasons": reasons,
        "limits": {
            "relative_l2": RELATIVE_L2_LIMIT,
            "max_flagged_fraction": MAX_FLAGGED_FRACTION,
            "max_step_flagged_fraction": MAX_STEP_FLAGGED_FRACTION,
            "seeds": list(REQUIRED_SEEDS),
            "sequence_steps_per_seed": REQUIRED_SEQUENCE_STEPS,
        },
        "observed": {
            "vendor_mismatches": total_vendor_mismatches,
            "flags": total_flags,
            "rows": total_rows,
            "flagged_fraction": total_flags / max(total_rows, 1),
        },
        "fixtures": fixture_results,
        "decision": {
            "bounded_correction_kernel_authorized": qualified,
            "production_promotion_authorized": False,
            "yaml_change_authorized": False,
            "main_merge_authorized": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("report must be a JSON object")
    result = qualify(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
