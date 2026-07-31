#!/usr/bin/env python3
"""Fresh-seed calibrated numeric screen for the M1-157 FP16-QK path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import bench_m1_157_fp16_qk_ab as legacy


SCHEMA = "bi100-m1-162-calibrated-fp16-qk-ab-cell-v1"
ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT / "quality" / "fused_prefill_numeric_adjudication.v2.json")
CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
CONTRACT_SHA256 = hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()
CASES = legacy.CASES
WARMUPS = legacy.WARMUPS
TRIALS = legacy.TRIALS
SEED_BASE = CONTRACT["fresh_synthetic_screen"]["seed_base"]
CASE_SEEDS = {
    case: SEED_BASE + index
    for index, case in enumerate(CASES)
}
HARD = CONTRACT["hard_gates"]
MAX_ERROR_MULTIPLE = HARD["maximum_error_multiple_over_fp16_rounding"]
RATIO_FLOOR = HARD["ratio_denominator_floor"]
MAX_LSE_RELATIVE_L2 = HARD["maximum_lse_relative_l2"]
MIN_CELL_SPEEDUP = (
    CONTRACT["fresh_synthetic_screen"]["minimum_cell_speedup"])


def _finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _relative_l2(actual: Any, expected: Any) -> float:
    return legacy.production._relative_l2(actual, expected)


def _calibrated_metrics(candidate: Any, reference_fp32: Any) -> dict[str, Any]:
    import torch

    rounded = reference_fp32.to(candidate.dtype)
    candidate_fp32 = candidate.float()
    rounded_fp32 = rounded.float()
    candidate_to_fp32_l2 = _relative_l2(
        candidate_fp32, reference_fp32)
    rounding_to_fp32_l2 = _relative_l2(
        rounded_fp32, reference_fp32)
    candidate_to_fp32_max = float(
        (candidate_fp32 - reference_fp32).abs().max().item())
    rounding_to_fp32_max = float(
        (rounded_fp32 - reference_fp32).abs().max().item())
    candidate_to_rounded_l2 = _relative_l2(candidate, rounded)
    candidate_to_rounded_max = float(
        (candidate_fp32 - rounded_fp32).abs().max().item())
    return {
        "candidate_finite": bool(torch.isfinite(candidate).all().item()),
        "reference_fp32_finite": bool(
            torch.isfinite(reference_fp32).all().item()),
        "rounded_reference_finite": bool(
            torch.isfinite(rounded).all().item()),
        "candidate_vs_rounded_relative_l2": candidate_to_rounded_l2,
        "candidate_vs_rounded_max_abs": candidate_to_rounded_max,
        "candidate_to_fp32_relative_l2": candidate_to_fp32_l2,
        "fp16_rounding_to_fp32_relative_l2": rounding_to_fp32_l2,
        "relative_l2_error_multiple_over_fp16_rounding": (
            candidate_to_fp32_l2 / max(rounding_to_fp32_l2, RATIO_FLOOR)),
        "candidate_to_fp32_max_abs": candidate_to_fp32_max,
        "fp16_rounding_to_fp32_max_abs": rounding_to_fp32_max,
        "max_abs_error_multiple_over_fp16_rounding": (
            candidate_to_fp32_max / max(rounding_to_fp32_max, RATIO_FLOOR)),
    }


def evaluate(result: Any) -> dict[str, Any]:
    reasons: list[str] = []
    if not isinstance(result, dict):
        return {"qualified": False, "reasons": ["result must be an object"]}
    if result.get("schema") != SCHEMA:
        reasons.append("cell schema differs")
    case = result.get("case")
    if case not in CASES:
        reasons.append("case is outside the fixed P90 grid")
        return {"qualified": False, "reasons": reasons}
    context_len, query_len = CASES[case]
    for name, expected in (
        ("context_len", context_len),
        ("query_len", query_len),
        ("warmups", WARMUPS),
        ("trials", TRIALS),
        ("seed", CASE_SEEDS[case]),
        ("numeric_contract_sha256", CONTRACT_SHA256),
    ):
        if result.get(name) != expected:
            reasons.append(f"{name} differs")

    numerical = result.get("numerical")
    candidate = (
        numerical.get("candidate_calibrated")
        if isinstance(numerical, dict) else None)
    baseline = (
        numerical.get("baseline_calibrated")
        if isinstance(numerical, dict) else None)
    if not isinstance(candidate, dict) or not isinstance(baseline, dict):
        reasons.append("calibrated numerical reports are missing")
    else:
        for name, values in (
            ("candidate", candidate),
            ("baseline", baseline),
        ):
            if not all(
                    values.get(field) is True
                    for field in (
                        "candidate_finite",
                        "reference_fp32_finite",
                        "rounded_reference_finite",
                    )):
                reasons.append(f"{name} or reference is nonfinite")
            for field in (
                "candidate_vs_rounded_relative_l2",
                "candidate_vs_rounded_max_abs",
                "candidate_to_fp32_relative_l2",
                "fp16_rounding_to_fp32_relative_l2",
                "relative_l2_error_multiple_over_fp16_rounding",
                "candidate_to_fp32_max_abs",
                "fp16_rounding_to_fp32_max_abs",
                "max_abs_error_multiple_over_fp16_rounding",
            ):
                if not _finite_nonnegative(values.get(field)):
                    reasons.append(f"{name}.{field} is invalid")
        for field in (
            "relative_l2_error_multiple_over_fp16_rounding",
            "max_abs_error_multiple_over_fp16_rounding",
        ):
            value = candidate.get(field)
            if (
                _finite_nonnegative(value)
                and float(value) > MAX_ERROR_MULTIPLE
            ):
                reasons.append(
                    f"candidate.{field} exceeds {MAX_ERROR_MULTIPLE:g}")

    lse_relative_l2 = (
        numerical.get("candidate_lse_relative_l2")
        if isinstance(numerical, dict) else None)
    if (
        not _finite_nonnegative(lse_relative_l2)
        or float(lse_relative_l2) > MAX_LSE_RELATIVE_L2
    ):
        reasons.append(
            f"candidate_lse_relative_l2 exceeds {MAX_LSE_RELATIVE_L2:g}")
    repeat = (
        numerical.get("candidate_repeat")
        if isinstance(numerical, dict) else None)
    if (
        not isinstance(repeat, dict)
        or repeat.get("output_exact") is not True
        or repeat.get("lse_exact") is not True
    ):
        reasons.append("candidate repeat is not exact")

    timings = result.get("timings")
    speedup = timings.get("speedup") if isinstance(timings, dict) else None
    if not _finite_nonnegative(speedup):
        reasons.append("speedup is invalid")
    elif float(speedup) < MIN_CELL_SPEEDUP:
        reasons.append(
            f"speedup is below the {MIN_CELL_SPEEDUP:.2f}x cell floor")
    for side in ("baseline", "candidate"):
        values = timings.get(side) if isinstance(timings, dict) else None
        trials = (
            values.get("cuda_trials_ms") if isinstance(values, dict) else None)
        if (
            not isinstance(trials, list)
            or len(trials) != TRIALS
            or not all(
                _finite_nonnegative(value) and value > 0
                for value in trials)
        ):
            reasons.append(f"{side} timing trials are invalid")

    if result.get("authorization") != {
        "operator_screen_only": True,
        "real_activation_replay_authorized": False,
        "tp4_service_authorized": False,
        "main_or_yaml_change_authorized": False,
    }:
        reasons.append("authorization must fail closed")
    return {"qualified": not reasons, "reasons": reasons}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("M1-162 requires one visible healthy BI100")
    context_len, query_len = CASES[args.case]
    baseline, baseline_artifact = legacy._load_extension(
        args.baseline_extension,
        args.expected_baseline_sha256,
        args.baseline_module_name,
    )
    candidate, candidate_artifact = legacy._load_extension(
        args.candidate_extension,
        args.expected_candidate_sha256,
        args.candidate_module_name,
    )
    seed = CASE_SEEDS[args.case]
    inputs = legacy.production._make_inputs(
        context_len, query_len, seed=seed)
    scale = legacy.production.HEAD_DIM**-0.5
    baseline_call = lambda: tuple(
        baseline.forward(*inputs, context_len, scale))
    candidate_call = lambda: tuple(
        candidate.forward(*inputs, context_len, scale))
    timings, baseline_result, candidate_result = legacy._paired_measure(
        baseline_call, candidate_call)

    reference_output_fp32, reference_lse = (
        legacy.production.reference_forward_fp32(
            *inputs, context_len, scale))
    repeat_result = candidate_call()
    torch.cuda.synchronize()
    baseline_output, baseline_lse = baseline_result
    candidate_output, candidate_lse = candidate_result
    repeat_output, repeat_lse = repeat_result
    rounded_reference = reference_output_fp32.to(candidate_output.dtype)
    candidate_vs_reference = {
        "finite": bool(
            torch.isfinite(candidate_output).all().item()
            and torch.isfinite(candidate_lse).all().item()),
        "output_max_abs": float(
            (candidate_output.float() - rounded_reference.float())
            .abs()
            .max()
            .item()),
        "output_relative_l2": _relative_l2(
            candidate_output, rounded_reference),
        "lse_relative_l2": _relative_l2(
            candidate_lse, reference_lse),
    }
    numerical = {
        "baseline_calibrated": _calibrated_metrics(
            baseline_output, reference_output_fp32),
        "candidate_calibrated": _calibrated_metrics(
            candidate_output, reference_output_fp32),
        "candidate_vs_reference": candidate_vs_reference,
        "candidate_vs_baseline": legacy._comparison(
            candidate_result, baseline_result),
        "baseline_lse_relative_l2": _relative_l2(
            baseline_lse, reference_lse),
        "candidate_lse_relative_l2": _relative_l2(
            candidate_lse, reference_lse),
        "candidate_repeat": {
            "output_exact": bool(
                torch.equal(candidate_output, repeat_output)),
            "lse_exact": bool(torch.equal(candidate_lse, repeat_lse)),
        },
    }
    result = {
        "schema": SCHEMA,
        "version": 1,
        "source_revision": args.source_revision,
        "runtime_identity": args.runtime_identity,
        "instance": args.instance,
        "visible_physical_gpu": args.visible_physical_gpu,
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "case": args.case,
        "context_len": context_len,
        "query_len": query_len,
        "total_kv_len": context_len + query_len,
        "seed": seed,
        "warmups": WARMUPS,
        "trials": TRIALS,
        "numeric_contract_sha256": CONTRACT_SHA256,
        "baseline_extension": baseline_artifact,
        "candidate_extension": candidate_artifact,
        "timings": timings,
        "numerical": numerical,
        "authorization": {
            "operator_screen_only": True,
            "real_activation_replay_authorized": False,
            "tp4_service_authorized": False,
            "main_or_yaml_change_authorized": False,
        },
    }
    result["evaluation"] = evaluate(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=tuple(CASES))
    parser.add_argument("--baseline-extension", required=True, type=Path)
    parser.add_argument("--candidate-extension", required=True, type=Path)
    parser.add_argument("--expected-baseline-sha256", required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument(
        "--baseline-module-name",
        default="corex_fused_paged_prefill",
    )
    parser.add_argument(
        "--candidate-module-name",
        default="corex_fused_paged_prefill_fp16_qk",
    )
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--runtime-identity", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--visible-physical-gpu", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            result,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    print(json.dumps({
        "case": result["case"],
        "qualified": result["evaluation"]["qualified"],
        "speedup": result["timings"]["speedup"],
        "candidate_vs_rounded_l2": result["numerical"][
            "candidate_calibrated"
        ]["candidate_vs_rounded_relative_l2"],
        "candidate_fp32_l2_multiple": result["numerical"][
            "candidate_calibrated"
        ]["relative_l2_error_multiple_over_fp16_rounding"],
        "candidate_fp32_max_multiple": result["numerical"][
            "candidate_calibrated"
        ]["max_abs_error_multiple_over_fp16_rounding"],
        "reasons": result["evaluation"]["reasons"],
    }, sort_keys=True))
    return 0 if result["evaluation"]["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
