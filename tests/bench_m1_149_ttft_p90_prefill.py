#!/usr/bin/env python3
"""Screen fused prefill on the chunk positions that drive platform TTFT P90."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import bench_m1_55_production_prefill as production


SCHEMA = "bi100-m1-149-ttft-p90-prefill-cell-v1"
QUERY_LEN = 8176
MIN_SPEEDUP = 1.2
CASES = {
    f"p90_total_{total_k // 1024:02d}k_q8176": (
        total_k - 8192,
        QUERY_LEN,
    )
    for total_k in range(8192, 65537, 8192)
}


def _finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def evaluate(result: Any) -> dict[str, Any]:
    reasons: list[str] = []
    if not isinstance(result, dict):
        return {"qualified": False, "reasons": ["result must be an object"]}
    case = result.get("case")
    if result.get("schema") != SCHEMA:
        reasons.append("cell schema differs")
    if result.get("version") != 1:
        reasons.append("cell version differs")
    if not _hex(result.get("source_commit"), 40):
        reasons.append("source commit is invalid")
    if case not in CASES:
        reasons.append("case is outside the frozen P90 grid")
        return {"qualified": False, "reasons": reasons}
    context_len, query_len = CASES[case]
    for name, expected in (
        ("context_len", context_len),
        ("query_len", query_len),
        ("total_kv_len", context_len + query_len),
        ("seed", production.SEED),
        ("warmups", production.WARMUPS),
        ("trials", production.TRIALS),
    ):
        if result.get(name) != expected:
            reasons.append(f"{name} differs")

    numerical = result.get("numerical")
    if not isinstance(numerical, dict):
        reasons.append("numerical report is missing")
    else:
        if numerical.get("finite") is not True:
            reasons.append("candidate output is nonfinite")
        for name, limit in (
            ("output_relative_l2", production.RELATIVE_L2_LIMIT),
            ("lse_relative_l2", production.RELATIVE_L2_LIMIT),
            ("output_max_abs", production.MAX_ABS_LIMIT),
        ):
            value = numerical.get(name)
            if not _finite_nonnegative(value) or float(value) > limit:
                reasons.append(f"numerical.{name} exceeds {limit:g}")

    timings = result.get("timings")
    speedup = timings.get("speedup") if isinstance(timings, dict) else None
    if isinstance(timings, dict):
        for side in ("reference", "candidate"):
            timing = timings.get(side)
            median = (
                timing.get("cuda_median_ms")
                if isinstance(timing, dict) else None
            )
            if not _finite_nonnegative(median) or float(median) <= 0.0:
                reasons.append(f"{side} CUDA median is invalid")
    if not _finite_nonnegative(speedup) or float(speedup) < MIN_SPEEDUP:
        reasons.append(f"speedup is below {MIN_SPEEDUP:.1f}x")
    if result.get("authorization") != {
        "short_tp4_p90_screen_authorized": False,
        "l2_capture_authorized": False,
        "main_or_yaml_change_authorized": False,
        "official_score_claim_authorized": False,
    }:
        reasons.append("cell authorization must fail closed")
    return {"qualified": not reasons, "reasons": reasons}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "M1-149 requires exactly one visible healthy BI100")
    context_len, query_len = CASES[args.case]
    extension, artifact = production._load_extension(
        args.extension, args.expected_extension_sha256)
    inputs = production._make_inputs(context_len, query_len)
    scale = production.HEAD_DIM ** -0.5
    reference_call = lambda: production.reference_forward(
        *inputs, context_len, scale)
    candidate_call = lambda: tuple(
        extension.forward(*inputs, context_len, scale))

    reference_timing, reference = production._measure(reference_call)
    candidate_timing, candidate = production._measure(candidate_call)
    candidate_output, candidate_lse = candidate
    reference_output, reference_lse = reference
    numerical = {
        "finite": bool(
            torch.isfinite(candidate_output).all().item()
            and torch.isfinite(candidate_lse).all().item()
        ),
        "output_max_abs": float(
            (candidate_output.float() - reference_output.float())
            .abs()
            .max()
            .item()
        ),
        "output_relative_l2": production._relative_l2(
            candidate_output, reference_output),
        "lse_relative_l2": production._relative_l2(
            candidate_lse, reference_lse),
    }
    result = {
        "schema": SCHEMA,
        "version": 1,
        "source_commit": args.source_commit,
        "runtime_identity": args.runtime_identity,
        "instance": args.instance,
        "visible_physical_gpu": args.visible_physical_gpu,
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "case": args.case,
        "context_len": context_len,
        "query_len": query_len,
        "total_kv_len": context_len + query_len,
        "seed": production.SEED,
        "warmups": production.WARMUPS,
        "trials": production.TRIALS,
        "physical_block_permutation": context_len > 0,
        "extension": artifact,
        "timings": {
            "reference": reference_timing,
            "candidate": candidate_timing,
            "speedup": (
                reference_timing["cuda_median_ms"]
                / candidate_timing["cuda_median_ms"]
            ),
        },
        "numerical": numerical,
        "authorization": {
            "short_tp4_p90_screen_authorized": False,
            "l2_capture_authorized": False,
            "main_or_yaml_change_authorized": False,
            "official_score_claim_authorized": False,
        },
    }
    result["evaluation"] = evaluate(result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=tuple(CASES))
    parser.add_argument("--extension", required=True, type=Path)
    parser.add_argument("--expected-extension-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--runtime-identity", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--visible-physical-gpu", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            result,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="ascii",
    )
    print(json.dumps({
        "case": result["case"],
        "qualified": result["evaluation"]["qualified"],
        "speedup": result["timings"]["speedup"],
        "output_relative_l2": result["numerical"]["output_relative_l2"],
        "reasons": result["evaluation"]["reasons"],
    }, sort_keys=True))
    return 0 if result["evaluation"]["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
