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


SCHEMA = "bi100-moe-compensated-w13-integration-v1"
QUALIFICATION_SCHEMA = (
    "bi100-moe-compensated-w13-integration-qualification-v1")
REQUIRED_SEEDS = (20260716, 20260727)
REQUIRED_SEQUENCE_STEPS = 500
REQUIRED_WARMUP = 30
REQUIRED_ITERATIONS = 300
REQUIRED_REPEATS = 9
ROWS_PER_STEP = 2048
NONINFERIORITY_EPS = 1.0e-8
MAX_ROUTED_REGRESSION_RATIO = 1.02
TIMING_CASES = (
    "strict_reference",
    "direct_control",
    "compensated_candidate",
)


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
    if not finite:
        raise ValueError(f"{field} contains NaN/Inf")
    mismatches = require_int(
        result["mismatch_count"],
        f"{field}.mismatch_count",
        minimum=0,
        maximum=rows,
    )
    if exact != (mismatches == 0):
        raise ValueError(f"{field} exact flag and mismatch count disagree")
    for metric in ("max_abs", "mean_abs", "relative_l2"):
        require_number(
            result[metric],
            f"{field}.{metric}",
            minimum=0.0,
        )
    if float(result["mean_abs"]) > float(result["max_abs"]):
        raise ValueError(f"{field}.mean_abs exceeds max_abs")
    return result


def require_sequence(
    value: Any,
    field: str,
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
    if steps != REQUIRED_SEQUENCE_STEPS:
        raise ValueError(f"{field}.steps differs from the fixed gate")
    rows = require_int(result["rows"], f"{field}.rows", minimum=0)
    if rows != REQUIRED_SEQUENCE_STEPS * ROWS_PER_STEP:
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
    if (mismatches == 0) != (exact_steps == steps):
        raise ValueError(
            f"{field} exact-step and mismatch counters disagree")
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
    parsed = [
        require_number(
            trial,
            f"{field}.trials_ms[{index}]",
            minimum=1.0e-12,
        )
        for index, trial in enumerate(trials)
    ]
    ordered = sorted(parsed)
    expected = {
        "median_ms": statistics.median(parsed),
        "p10_ms": ordered[max(0, int(0.1 * (len(ordered) - 1)))],
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


def check_noninferiority(
    candidate: dict[str, Any],
    control: dict[str, Any],
    field: str,
    reasons: list[str],
) -> None:
    for metric in ("relative_l2", "max_abs"):
        if (
            float(candidate[metric])
            > float(control[metric]) + NONINFERIORITY_EPS
        ):
            reasons.append(
                f"{field} candidate {metric} is worse than direct control")
    if int(candidate["mismatch_count"]) > int(control["mismatch_count"]):
        reasons.append(
            f"{field} candidate mismatch count is worse than direct control")
    if (
        "max_step_relative_l2" in candidate
        and float(candidate["max_step_relative_l2"])
        > float(control["max_step_relative_l2"]) + NONINFERIORITY_EPS
    ):
        reasons.append(
            f"{field} candidate max-step relative L2 is worse "
            "than direct control")


def _qualify(
    report: dict[str, Any],
    *,
    expected_extension_sha256: str | None,
    expected_exact_reduce_sha256: str | None,
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
            "artifacts",
            "fixed",
            "sequence",
            "timings",
        },
        "report",
    )
    if report["schema"] != SCHEMA:
        reasons.append("benchmark schema is invalid")
    if report["version"] != 1:
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
    if require_mapping(report["shape"], "shape") != expected_shape:
        reasons.append("production TP4 rank-local shape contract changed")

    expected_config = {
        "device": "cuda:0",
        "seeds": list(REQUIRED_SEEDS),
        "sequence_steps_per_seed": REQUIRED_SEQUENCE_STEPS,
        "warmup": REQUIRED_WARMUP,
        "iterations": REQUIRED_ITERATIONS,
        "repeats": REQUIRED_REPEATS,
        "cpu_threads": 8,
        "w13_weight_scale": 0.02,
        "w2_weight_scale": 0.02,
    }
    if require_mapping(report["config"], "config") != expected_config:
        reasons.append("fixed integrated benchmark configuration changed")

    expected_method = {
        "reference":
            "pytorch_gather_linear_vllm_silu_and_mul_bmm_"
            "corex_serial_float_reduce",
        "control": "production_direct_w13_and_w2_reduce",
        "candidate": "production_compensated_w13_and_same_w2_reduce",
        "timing_order": "alternating_forward_reverse",
        "request_semantics_changed": False,
    }
    if require_mapping(report["method"], "method") != expected_method:
        reasons.append("integrated MoE method contract changed")

    artifacts = require_mapping(report["artifacts"], "artifacts")
    require_exact_keys(
        artifacts,
        {
            "extension_sha256",
            "extension_size_bytes",
            "exact_reduce_sha256",
            "exact_reduce_size_bytes",
        },
        "artifacts",
    )
    extension_sha256 = require_sha256(
        artifacts["extension_sha256"],
        "artifacts.extension_sha256",
    )
    exact_reduce_sha256 = require_sha256(
        artifacts["exact_reduce_sha256"],
        "artifacts.exact_reduce_sha256",
    )
    require_int(
        artifacts["extension_size_bytes"],
        "artifacts.extension_size_bytes",
        minimum=1,
    )
    require_int(
        artifacts["exact_reduce_size_bytes"],
        "artifacts.exact_reduce_size_bytes",
        minimum=1,
    )
    if (
        expected_extension_sha256 is not None
        and extension_sha256 != expected_extension_sha256
    ):
        reasons.append("production extension SHA-256 does not match artifact")
    if (
        expected_exact_reduce_sha256 is not None
        and exact_reduce_sha256 != expected_exact_reduce_sha256
    ):
        reasons.append("exact-reduce SHA-256 does not match artifact")

    seed_keys = {str(seed) for seed in REQUIRED_SEEDS}
    fixed = require_mapping(report["fixed"], "fixed")
    require_exact_keys(fixed, seed_keys, "fixed")
    sequence = require_mapping(report["sequence"], "sequence")
    require_exact_keys(sequence, seed_keys, "sequence")
    fixed_observed: dict[str, Any] = {}
    sequence_observed: dict[str, Any] = {}
    for seed in REQUIRED_SEEDS:
        key = str(seed)
        fixed_row = require_mapping(fixed[key], f"fixed.{key}")
        require_exact_keys(
            fixed_row,
            {
                "direct_vs_reference",
                "candidate_vs_reference",
                "candidate_vs_direct",
                "direct_repeat_exact",
                "candidate_repeat_exact",
            },
            f"fixed.{key}",
        )
        direct_fixed = require_comparison(
            fixed_row["direct_vs_reference"],
            f"fixed.{key}.direct_vs_reference",
            rows=ROWS_PER_STEP,
        )
        candidate_fixed = require_comparison(
            fixed_row["candidate_vs_reference"],
            f"fixed.{key}.candidate_vs_reference",
            rows=ROWS_PER_STEP,
        )
        require_comparison(
            fixed_row["candidate_vs_direct"],
            f"fixed.{key}.candidate_vs_direct",
            rows=ROWS_PER_STEP,
        )
        direct_repeat = require_bool(
            fixed_row["direct_repeat_exact"],
            f"fixed.{key}.direct_repeat_exact",
        )
        candidate_repeat = require_bool(
            fixed_row["candidate_repeat_exact"],
            f"fixed.{key}.candidate_repeat_exact",
        )
        if not direct_repeat:
            reasons.append(f"seed {seed} direct control is non-deterministic")
        if not candidate_repeat:
            reasons.append(f"seed {seed} candidate is non-deterministic")
        check_noninferiority(
            candidate_fixed,
            direct_fixed,
            f"seed {seed} fixed routed output",
            reasons,
        )
        fixed_observed[key] = {
            "direct_relative_l2": float(direct_fixed["relative_l2"]),
            "candidate_relative_l2": float(
                candidate_fixed["relative_l2"]),
            "direct_max_abs": float(direct_fixed["max_abs"]),
            "candidate_max_abs": float(candidate_fixed["max_abs"]),
            "direct_mismatch_count": int(
                direct_fixed["mismatch_count"]),
            "candidate_mismatch_count": int(
                candidate_fixed["mismatch_count"]),
        }

        sequence_row = require_mapping(
            sequence[key],
            f"sequence.{key}",
        )
        require_exact_keys(
            sequence_row,
            {"direct_vs_reference", "candidate_vs_reference"},
            f"sequence.{key}",
        )
        direct_sequence = require_sequence(
            sequence_row["direct_vs_reference"],
            f"sequence.{key}.direct_vs_reference",
        )
        candidate_sequence = require_sequence(
            sequence_row["candidate_vs_reference"],
            f"sequence.{key}.candidate_vs_reference",
        )
        check_noninferiority(
            candidate_sequence,
            direct_sequence,
            f"seed {seed} routed sequence",
            reasons,
        )
        sequence_observed[key] = {
            "direct_relative_l2":
                float(direct_sequence["relative_l2"]),
            "candidate_relative_l2":
                float(candidate_sequence["relative_l2"]),
            "direct_max_step_relative_l2":
                float(direct_sequence["max_step_relative_l2"]),
            "candidate_max_step_relative_l2":
                float(candidate_sequence["max_step_relative_l2"]),
            "direct_max_abs": float(direct_sequence["max_abs"]),
            "candidate_max_abs": float(candidate_sequence["max_abs"]),
            "direct_mismatch_count":
                int(direct_sequence["mismatch_count"]),
            "candidate_mismatch_count":
                int(candidate_sequence["mismatch_count"]),
        }

    timings = require_mapping(report["timings"], "timings")
    require_exact_keys(
        timings,
        {
            "cases",
            "orders",
            "candidate_vs_direct_ratio",
            "candidate_vs_reference_speedup",
        },
        "timings",
    )
    cases = require_mapping(timings["cases"], "timings.cases")
    require_exact_keys(cases, set(TIMING_CASES), "timings.cases")
    parsed_timings = {
        name: require_timing(cases[name], f"timings.cases.{name}")
        for name in TIMING_CASES
    }
    expected_orders = [
        list(TIMING_CASES) if repeat % 2 == 0
        else list(reversed(TIMING_CASES))
        for repeat in range(REQUIRED_REPEATS)
    ]
    if timings["orders"] != expected_orders:
        reasons.append("timing arm order differs from the fixed A/B gate")

    direct_median = float(
        parsed_timings["direct_control"]["median_ms"])
    candidate_median = float(
        parsed_timings["compensated_candidate"]["median_ms"])
    strict_median = float(
        parsed_timings["strict_reference"]["median_ms"])
    ratio = require_number(
        timings["candidate_vs_direct_ratio"],
        "timings.candidate_vs_direct_ratio",
        minimum=0.0,
    )
    speedup = require_number(
        timings["candidate_vs_reference_speedup"],
        "timings.candidate_vs_reference_speedup",
        minimum=0.0,
    )
    expected_ratio = candidate_median / direct_median
    expected_speedup = strict_median / candidate_median
    if not math.isclose(
        ratio,
        expected_ratio,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        reasons.append("candidate/direct ratio is inconsistent with medians")
    if not math.isclose(
        speedup,
        expected_speedup,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        reasons.append(
            "candidate/reference speedup is inconsistent with medians")
    if ratio > MAX_ROUTED_REGRESSION_RATIO:
        reasons.append(
            "complete routed candidate regresses by more than 2%")

    qualified = not reasons
    return {
        "schema": QUALIFICATION_SCHEMA,
        "version": 1,
        "qualified": qualified,
        "reasons": reasons,
        "limits": {
            "noninferiority_epsilon": NONINFERIORITY_EPS,
            "max_routed_regression_ratio":
                MAX_ROUTED_REGRESSION_RATIO,
            "seeds": list(REQUIRED_SEEDS),
            "sequence_steps_per_seed": REQUIRED_SEQUENCE_STEPS,
            "warmup": REQUIRED_WARMUP,
            "iterations": REQUIRED_ITERATIONS,
            "repeats": REQUIRED_REPEATS,
        },
        "evidence": {
            "report_sha256": report_sha256,
            "extension_sha256": extension_sha256,
            "exact_reduce_sha256": exact_reduce_sha256,
        },
        "observed": {
            "fixed": fixed_observed,
            "sequence": sequence_observed,
            "candidate_vs_direct_ratio": ratio,
            "candidate_vs_reference_speedup": speedup,
        },
        "decision": {
            "single_gpu_integrated_screen_qualified": qualified,
            "tp4_evaluation_authorized": qualified,
            "production_promotion_authorized": False,
            "yaml_change_authorized": False,
            "main_merge_authorized": False,
        },
    }


def qualify(
    report: Any,
    *,
    expected_extension_sha256: str | None = None,
    expected_exact_reduce_sha256: str | None = None,
    report_sha256: str | None = None,
) -> dict[str, Any]:
    try:
        value = require_mapping(report, "report")
        return _qualify(
            value,
            expected_extension_sha256=expected_extension_sha256,
            expected_exact_reduce_sha256=expected_exact_reduce_sha256,
            report_sha256=report_sha256,
        )
    except (KeyError, TypeError, ValueError) as error:
        return {
            "schema": QUALIFICATION_SCHEMA,
            "version": 1,
            "qualified": False,
            "reasons": [f"invalid benchmark evidence: {error}"],
            "limits": {
                "noninferiority_epsilon": NONINFERIORITY_EPS,
                "max_routed_regression_ratio":
                    MAX_ROUTED_REGRESSION_RATIO,
                "seeds": list(REQUIRED_SEEDS),
                "sequence_steps_per_seed": REQUIRED_SEQUENCE_STEPS,
                "warmup": REQUIRED_WARMUP,
                "iterations": REQUIRED_ITERATIONS,
                "repeats": REQUIRED_REPEATS,
            },
            "evidence": {
                "report_sha256": report_sha256,
                "extension_sha256": expected_extension_sha256,
                "exact_reduce_sha256": expected_exact_reduce_sha256,
            },
            "observed": {},
            "decision": {
                "single_gpu_integrated_screen_qualified": False,
                "tp4_evaluation_authorized": False,
                "production_promotion_authorized": False,
                "yaml_change_authorized": False,
                "main_merge_authorized": False,
            },
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--exact-reduce-extension", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report_bytes = args.report.read_bytes()
    report_sha256 = sha256_bytes(report_bytes)
    report = json.loads(report_bytes)
    result = qualify(
        report,
        expected_extension_sha256=sha256_file(args.extension),
        expected_exact_reduce_sha256=sha256_file(
            args.exact_reduce_extension),
        report_sha256=report_sha256,
    )
    if sha256_bytes(args.report.read_bytes()) != report_sha256:
        raise RuntimeError("benchmark report changed during qualification")
    atomic_write(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
