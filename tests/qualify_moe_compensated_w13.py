#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any


SCHEMA = "bi100-moe-compensated-w13-v2"
QUALIFICATION_SCHEMA = "bi100-moe-compensated-w13-qualification-v2"
NONINFERIORITY_EPS = 1.0e-8
FIXED_SPEEDUP_MIN = 1.5
ROUTED_SPEEDUP_MIN = 1.25
REQUIRED_SEEDS = (20260716, 20260727)
REQUIRED_SEQUENCE_STEPS = 500
REQUIRED_EXACT_SEQUENCE_INDICES = (
    0,
    1,
    2,
    3,
    7,
    15,
    31,
    63,
    127,
    255,
    383,
    499,
)
REQUIRED_WARMUP = 30
REQUIRED_ITERATIONS = 300
REQUIRED_REPEATS = 9
ROWS_PER_STEP = 2048


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(
                value,
                output,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def require_exact_keys(
    value: dict[str, Any],
    expected: set[str],
    field: str,
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{field} fields differ: expected {sorted(expected)}, "
            f"got {sorted(value)}")


def require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def require_int(
    value: Any,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} is below {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field} exceeds {maximum}")
    return value


def require_number(
    value: Any,
    field: str,
    *,
    minimum: float | None = None,
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} is below {minimum}")
    return result


def require_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def require_comparison(
    value: Any,
    field: str,
    *,
    rows: int,
) -> dict[str, Any]:
    result = require_mapping(value, field)
    require_exact_keys(
        result,
        {
            "exact",
            "finite",
            "mismatch_count",
            "max_abs",
            "mean_abs",
            "relative_l2",
        },
        field,
    )
    exact = require_bool(result["exact"], f"{field}.exact")
    finite = require_bool(result["finite"], f"{field}.finite")
    mismatches = require_int(
        result["mismatch_count"],
        f"{field}.mismatch_count",
        minimum=0,
        maximum=rows,
    )
    if exact != (mismatches == 0):
        raise ValueError(f"{field} exact flag and mismatch count disagree")
    if not finite:
        raise ValueError(f"{field} contains NaN/Inf")
    for metric in ("max_abs", "mean_abs", "relative_l2"):
        require_number(
            result[metric],
            f"{field}.{metric}",
            minimum=0.0,
        )
    if float(result["mean_abs"]) > float(result["max_abs"]):
        raise ValueError(f"{field}.mean_abs exceeds max_abs")
    return result


def require_sequence_metrics(
    value: Any,
    field: str,
    *,
    expected_steps: int = REQUIRED_SEQUENCE_STEPS,
) -> dict[str, Any]:
    result = require_mapping(value, field)
    require_exact_keys(
        result,
        {
            "steps",
            "rows",
            "finite_steps",
            "exact_steps",
            "mismatch_count",
            "max_abs",
            "mean_abs",
            "relative_l2",
            "max_step_relative_l2",
        },
        field,
    )
    steps = require_int(result["steps"], f"{field}.steps", minimum=0)
    if steps != expected_steps:
        raise ValueError(f"{field}.steps differs from the fixed gate")
    rows = require_int(result["rows"], f"{field}.rows", minimum=0)
    if rows != expected_steps * ROWS_PER_STEP:
        raise ValueError(f"{field}.rows differs from the fixed gate")
    finite_steps = require_int(
        result["finite_steps"],
        f"{field}.finite_steps",
        minimum=0,
        maximum=steps,
    )
    if finite_steps != steps:
        raise ValueError(f"{field} contains a non-finite step")
    exact_steps = require_int(
        result["exact_steps"],
        f"{field}.exact_steps",
        minimum=0,
        maximum=steps,
    )
    mismatches = require_int(
        result["mismatch_count"],
        f"{field}.mismatch_count",
        minimum=0,
        maximum=rows,
    )
    for metric in (
        "max_abs",
        "mean_abs",
        "relative_l2",
        "max_step_relative_l2",
    ):
        require_number(
            result[metric],
            f"{field}.{metric}",
            minimum=0.0,
        )
    if (mismatches == 0) != (exact_steps == steps):
        raise ValueError(
            f"{field} exact-step and mismatch counters disagree")
    if float(result["mean_abs"]) > float(result["max_abs"]):
        raise ValueError(f"{field}.mean_abs exceeds max_abs")
    if (
        float(result["relative_l2"])
        > float(result["max_step_relative_l2"]) + 1.0e-15
    ):
        raise ValueError(
            f"{field}.relative_l2 exceeds max-step relative L2")
    return result


def require_timing(value: Any, field: str) -> dict[str, Any]:
    result = require_mapping(value, field)
    require_exact_keys(
        result,
        {"median_ms", "p10_ms", "p90_ms", "trials_ms"},
        field,
    )
    trials = result["trials_ms"]
    if not isinstance(trials, list) or len(trials) != REQUIRED_REPEATS:
        raise ValueError(
            f"{field}.trials_ms must contain {REQUIRED_REPEATS} trials")
    parsed_trials = [
        require_number(item, f"{field}.trials_ms[{index}]", minimum=1.0e-12)
        for index, item in enumerate(trials)
    ]
    ordered = sorted(parsed_trials)
    expected = {
        "median_ms": statistics.median(parsed_trials),
        "p10_ms": ordered[max(
            0,
            int(0.1 * (len(ordered) - 1)),
        )],
        "p90_ms": ordered[min(
            len(ordered) - 1,
            int(0.9 * (len(ordered) - 1)),
        )],
    }
    for metric, expected_value in expected.items():
        observed = require_number(
            result[metric],
            f"{field}.{metric}",
            minimum=1.0e-12,
        )
        if not math.isclose(
            observed,
            expected_value,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"{field}.{metric} is inconsistent with trials")
    return result


def check_exact_reference_noninferiority(
    candidate: dict[str, Any],
    vendor: dict[str, Any],
    field: str,
    reasons: list[str],
) -> None:
    if (
        float(candidate["relative_l2"])
        > float(vendor["relative_l2"]) + NONINFERIORITY_EPS
    ):
        reasons.append(
            f"{field} candidate relative L2 is worse than vendor")
    if (
        float(candidate["max_abs"])
        > float(vendor["max_abs"]) + NONINFERIORITY_EPS
    ):
        reasons.append(
            f"{field} candidate max absolute error is worse than vendor")
    if int(candidate["mismatch_count"]) > int(vendor["mismatch_count"]):
        reasons.append(
            f"{field} candidate mismatch count is worse than vendor")
    if (
        "max_step_relative_l2" in candidate
        and float(candidate["max_step_relative_l2"])
        > float(vendor["max_step_relative_l2"]) + NONINFERIORITY_EPS
    ):
        reasons.append(
            f"{field} candidate max-step relative L2 is worse than vendor")


def _qualify(
    report: dict[str, Any],
    *,
    expected_candidate_sha256: str | None,
    expected_direct_sha256: str | None,
    report_sha256: str | None,
) -> dict[str, Any]:
    reasons: list[str] = []
    require_exact_keys(
        report,
        {
            "schema",
            "version",
            "device",
            "shape",
            "config",
            "method",
            "extensions",
            "fixed",
            "sequence",
            "exact_sequence",
            "timings",
        },
        "report",
    )
    if report["schema"] != SCHEMA:
        reasons.append("benchmark schema is invalid")
    if report["version"] != 2:
        reasons.append("benchmark version is invalid")
    if not isinstance(report["device"], str) or not report["device"]:
        reasons.append("device identity is missing")

    expected_shape = {
        "experts": 256,
        "top_k": 8,
        "hidden": 2048,
        "intermediate": 128,
        "rows_per_expert": 256,
        "dtype": "torch.float16",
    }
    shape = require_mapping(report["shape"], "shape")
    if shape != expected_shape:
        reasons.append("production TP4 rank-local shape contract changed")

    expected_config = {
        "device": "cuda:0",
        "seeds": list(REQUIRED_SEEDS),
        "sequence_steps_per_seed": REQUIRED_SEQUENCE_STEPS,
        "warmup": REQUIRED_WARMUP,
        "iterations": REQUIRED_ITERATIONS,
        "repeats": REQUIRED_REPEATS,
        "cpu_threads": 8,
        "weight_scale": 0.02,
        "exact_sequence_indices": list(
            REQUIRED_EXACT_SEQUENCE_INDICES),
    }
    config = require_mapping(report["config"], "config")
    if config != expected_config:
        reasons.append("fixed benchmark configuration changed")

    expected_method = {
        "algorithm": "per_lane_kahan_fp32_then_rn_warp_tree",
        "quality_reference":
            "cpu_float64_dot_rounded_to_fp16_noninferiority",
        "exact_diagnostic":
            "fixed_fixture_and_stratified_sequence_samples",
        "fixture_generation": "hidden_then_router_then_w13_then_sequence",
        "production_runtime_changed": False,
    }
    method = require_mapping(report["method"], "method")
    if method != expected_method:
        reasons.append("compensated accumulation method contract changed")

    extensions = require_mapping(report["extensions"], "extensions")
    require_exact_keys(
        extensions,
        {
            "candidate_sha256",
            "candidate_size_bytes",
            "direct_sha256",
            "direct_size_bytes",
        },
        "extensions",
    )
    candidate_sha256 = require_sha256(
        extensions["candidate_sha256"],
        "extensions.candidate_sha256",
    )
    direct_sha256 = require_sha256(
        extensions["direct_sha256"],
        "extensions.direct_sha256",
    )
    require_int(
        extensions["candidate_size_bytes"],
        "extensions.candidate_size_bytes",
        minimum=1,
    )
    require_int(
        extensions["direct_size_bytes"],
        "extensions.direct_size_bytes",
        minimum=1,
    )
    if candidate_sha256 == direct_sha256:
        reasons.append("candidate and direct extension identities collide")
    if (
        expected_candidate_sha256 is not None
        and candidate_sha256 != expected_candidate_sha256
    ):
        reasons.append("candidate extension SHA-256 does not match artifact")
    if (
        expected_direct_sha256 is not None
        and direct_sha256 != expected_direct_sha256
    ):
        reasons.append("direct extension SHA-256 does not match artifact")

    expected_seed_keys = {str(seed) for seed in REQUIRED_SEEDS}
    fixed = require_mapping(report["fixed"], "fixed")
    require_exact_keys(fixed, expected_seed_keys, "fixed")
    sequence = require_mapping(report["sequence"], "sequence")
    require_exact_keys(sequence, expected_seed_keys, "sequence")
    exact_sequence = require_mapping(
        report["exact_sequence"],
        "exact_sequence",
    )
    require_exact_keys(
        exact_sequence,
        expected_seed_keys,
        "exact_sequence",
    )

    fixed_observed: dict[str, Any] = {}
    sequence_observed: dict[str, Any] = {}
    exact_sequence_observed: dict[str, Any] = {}
    for seed in REQUIRED_SEEDS:
        key = str(seed)
        fixed_row = require_mapping(fixed[key], f"fixed.{key}")
        require_exact_keys(
            fixed_row,
            {
                "vendor_vs_exact",
                "direct_vs_exact",
                "direct_vs_vendor",
                "compensated_vs_vendor",
                "compensated_vs_exact",
            },
            f"fixed.{key}",
        )
        fixed_comparisons = {
            name: require_comparison(
                fixed_row[name],
                f"fixed.{key}.{name}",
                rows=ROWS_PER_STEP,
            )
            for name in fixed_row
        }
        check_exact_reference_noninferiority(
            fixed_comparisons["compensated_vs_exact"],
            fixed_comparisons["vendor_vs_exact"],
            f"seed {seed} fixed exact-reference",
            reasons,
        )
        fixed_observed[key] = {
            name: {
                "mismatch_count": int(value["mismatch_count"]),
                "max_abs": float(value["max_abs"]),
                "relative_l2": float(value["relative_l2"]),
            }
            for name, value in fixed_comparisons.items()
        }

        sequence_row = require_mapping(sequence[key], f"sequence.{key}")
        require_exact_keys(
            sequence_row,
            {"direct", "compensated"},
            f"sequence.{key}",
        )
        direct_sequence = require_sequence_metrics(
            sequence_row["direct"],
            f"sequence.{key}.direct",
        )
        candidate_sequence = require_sequence_metrics(
            sequence_row["compensated"],
            f"sequence.{key}.compensated",
        )
        candidate_sequence_l2 = float(candidate_sequence["relative_l2"])
        candidate_max_step_l2 = float(
            candidate_sequence["max_step_relative_l2"])
        direct_sequence_l2 = float(direct_sequence["relative_l2"])
        direct_max_step_l2 = float(
            direct_sequence["max_step_relative_l2"])
        sequence_observed[key] = {
            "compensated_relative_l2": candidate_sequence_l2,
            "compensated_max_step_relative_l2": candidate_max_step_l2,
            "direct_relative_l2": direct_sequence_l2,
            "direct_max_step_relative_l2": direct_max_step_l2,
        }

        exact_row = require_mapping(
            exact_sequence[key],
            f"exact_sequence.{key}",
        )
        require_exact_keys(
            exact_row,
            {"sample_indices", "comparisons"},
            f"exact_sequence.{key}",
        )
        sample_indices = exact_row["sample_indices"]
        if sample_indices != list(REQUIRED_EXACT_SEQUENCE_INDICES):
            raise ValueError(
                f"exact_sequence.{key}.sample_indices differs "
                "from the fixed gate")
        exact_comparisons_raw = require_mapping(
            exact_row["comparisons"],
            f"exact_sequence.{key}.comparisons",
        )
        require_exact_keys(
            exact_comparisons_raw,
            {"vendor", "direct", "compensated"},
            f"exact_sequence.{key}.comparisons",
        )
        exact_comparisons = {
            name: require_sequence_metrics(
                value,
                f"exact_sequence.{key}.comparisons.{name}",
                expected_steps=len(REQUIRED_EXACT_SEQUENCE_INDICES),
            )
            for name, value in exact_comparisons_raw.items()
        }
        check_exact_reference_noninferiority(
            exact_comparisons["compensated"],
            exact_comparisons["vendor"],
            f"seed {seed} stratified exact-reference sequence",
            reasons,
        )
        exact_sequence_observed[key] = {
            name: {
                "mismatch_count": int(value["mismatch_count"]),
                "max_abs": float(value["max_abs"]),
                "relative_l2": float(value["relative_l2"]),
                "max_step_relative_l2": float(
                    value["max_step_relative_l2"]),
            }
            for name, value in exact_comparisons.items()
        }

    timings = require_mapping(report["timings"], "timings")
    require_exact_keys(timings, {"cases", "speedups"}, "timings")
    timing_cases = require_mapping(timings["cases"], "timings.cases")
    required_timing_cases = {
        "vendor_fixed",
        "direct_fixed",
        "compensated_fixed",
        "vendor_routed",
        "direct_routed",
        "compensated_routed",
    }
    require_exact_keys(
        timing_cases,
        required_timing_cases,
        "timings.cases",
    )
    parsed_timings = {
        name: require_timing(
            timing_cases[name],
            f"timings.cases.{name}",
        )
        for name in sorted(required_timing_cases)
    }
    speedups = require_mapping(timings["speedups"], "timings.speedups")
    require_exact_keys(
        speedups,
        {
            "compensated_fixed_vs_vendor",
            "compensated_routed_vs_vendor",
        },
        "timings.speedups",
    )
    fixed_speedup = require_number(
        speedups["compensated_fixed_vs_vendor"],
        "timings.speedups.compensated_fixed_vs_vendor",
        minimum=0.0,
    )
    routed_speedup = require_number(
        speedups["compensated_routed_vs_vendor"],
        "timings.speedups.compensated_routed_vs_vendor",
        minimum=0.0,
    )
    expected_fixed_speedup = (
        float(parsed_timings["vendor_fixed"]["median_ms"])
        / float(parsed_timings["compensated_fixed"]["median_ms"])
    )
    expected_routed_speedup = (
        float(parsed_timings["vendor_routed"]["median_ms"])
        / float(parsed_timings["compensated_routed"]["median_ms"])
    )
    if not math.isclose(
        fixed_speedup,
        expected_fixed_speedup,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        reasons.append("fixed speedup is inconsistent with timing medians")
    if not math.isclose(
        routed_speedup,
        expected_routed_speedup,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        reasons.append("routed speedup is inconsistent with timing medians")
    if fixed_speedup < FIXED_SPEEDUP_MIN:
        reasons.append(
            f"fixed speedup is below {FIXED_SPEEDUP_MIN:g}x")
    if routed_speedup < ROUTED_SPEEDUP_MIN:
        reasons.append(
            f"routed speedup is below {ROUTED_SPEEDUP_MIN:g}x")

    qualified = not reasons
    return {
        "schema": QUALIFICATION_SCHEMA,
        "version": 2,
        "qualified": qualified,
        "reasons": reasons,
        "limits": {
            "numerical_oracle":
                "cpu_float64_dot_rounded_to_fp16_noninferiority",
            "noninferiority_epsilon": NONINFERIORITY_EPS,
            "fixed_speedup": FIXED_SPEEDUP_MIN,
            "routed_speedup": ROUTED_SPEEDUP_MIN,
            "seeds": list(REQUIRED_SEEDS),
            "sequence_steps_per_seed": REQUIRED_SEQUENCE_STEPS,
            "exact_sequence_indices": list(
                REQUIRED_EXACT_SEQUENCE_INDICES),
            "warmup": REQUIRED_WARMUP,
            "iterations": REQUIRED_ITERATIONS,
            "repeats": REQUIRED_REPEATS,
        },
        "evidence": {
            "report_sha256": report_sha256,
            "candidate_extension_sha256": candidate_sha256,
            "direct_extension_sha256": direct_sha256,
        },
        "observed": {
            "fixed": fixed_observed,
            "sequence": sequence_observed,
            "exact_sequence": exact_sequence_observed,
            "fixed_speedup": fixed_speedup,
            "routed_speedup": routed_speedup,
        },
        "decision": {
            "single_gpu_numerical_screen_qualified": qualified,
            "production_promotion_authorized": False,
            "yaml_change_authorized": False,
            "main_merge_authorized": False,
        },
    }


def qualify(
    report: Any,
    *,
    expected_candidate_sha256: str | None = None,
    expected_direct_sha256: str | None = None,
    report_sha256: str | None = None,
) -> dict[str, Any]:
    try:
        value = require_mapping(report, "report")
        return _qualify(
            value,
            expected_candidate_sha256=expected_candidate_sha256,
            expected_direct_sha256=expected_direct_sha256,
            report_sha256=report_sha256,
        )
    except (KeyError, TypeError, ValueError) as error:
        return {
            "schema": QUALIFICATION_SCHEMA,
            "version": 2,
            "qualified": False,
            "reasons": [f"invalid benchmark evidence: {error}"],
            "limits": {
                "numerical_oracle":
                    "cpu_float64_dot_rounded_to_fp16_noninferiority",
                "noninferiority_epsilon": NONINFERIORITY_EPS,
                "fixed_speedup": FIXED_SPEEDUP_MIN,
                "routed_speedup": ROUTED_SPEEDUP_MIN,
                "seeds": list(REQUIRED_SEEDS),
                "sequence_steps_per_seed": REQUIRED_SEQUENCE_STEPS,
                "exact_sequence_indices": list(
                    REQUIRED_EXACT_SEQUENCE_INDICES),
                "warmup": REQUIRED_WARMUP,
                "iterations": REQUIRED_ITERATIONS,
                "repeats": REQUIRED_REPEATS,
            },
            "evidence": {
                "report_sha256": report_sha256,
                "candidate_extension_sha256":
                    expected_candidate_sha256,
                "direct_extension_sha256": expected_direct_sha256,
            },
            "observed": {},
            "decision": {
                "single_gpu_numerical_screen_qualified": False,
                "production_promotion_authorized": False,
                "yaml_change_authorized": False,
                "main_merge_authorized": False,
            },
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--candidate-extension", type=Path, required=True)
    parser.add_argument("--direct-extension", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report_bytes = args.report.read_bytes()
    report_sha256 = sha256_bytes(report_bytes)
    report = json.loads(report_bytes)
    result = qualify(
        report,
        expected_candidate_sha256=sha256_file(
            args.candidate_extension),
        expected_direct_sha256=sha256_file(args.direct_extension),
        report_sha256=report_sha256,
    )
    if sha256_bytes(args.report.read_bytes()) != report_sha256:
        raise RuntimeError("benchmark report changed during qualification")
    atomic_write(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
