#!/usr/bin/env python3
"""Profile only the M1-109 candidate range on one healthy BI100."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
from pathlib import Path
from typing import Any

import bench_m1_55_production_prefill as production


SCHEMA = "bi100-m1-155-fused-prefill-phase-cell-v1"
PROFILE_TRIALS = 3
CASES = {
    "p90_total_16k_q8176": (8192, 8176),
    "p90_total_32k_q8176": (24576, 8176),
    "p90_total_64k_q8176": (57344, 8176),
}


def _finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def expected_launches(context_len: int, query_len: int) -> dict[str, int]:
    group_tokens = 2048
    tile_tokens = 512
    context_groups = (context_len + group_tokens - 1) // group_tokens
    query_groups = (query_len + group_tokens - 1) // group_tokens
    groups = context_groups + query_groups
    context_splits = (
        (context_len + tile_tokens - 1) // tile_tokens
        if context_len else 0
    )
    query_splits = (query_len + tile_tokens - 1) // tile_tokens
    return {
        "convert_query": PROFILE_TRIALS,
        "gather": groups * PROFILE_TRIALS,
        "qk": (context_splits + query_splits) * PROFILE_TRIALS,
        "mask": query_groups * PROFILE_TRIALS,
        "normalize": groups * PROFILE_TRIALS,
        "pv": (context_splits + query_splits) * PROFILE_TRIALS,
        "merge": groups * PROFILE_TRIALS,
    }


def evaluate(result: Any) -> dict[str, Any]:
    reasons: list[str] = []
    if not isinstance(result, dict):
        return {"qualified": False, "reasons": ["result must be an object"]}
    case = result.get("case")
    if result.get("schema") != SCHEMA or result.get("version") != 1:
        reasons.append("cell schema or version differs")
    if case not in CASES:
        reasons.append("case is outside the frozen phase matrix")
        return {"qualified": False, "reasons": reasons}
    context_len, query_len = CASES[case]
    for field, expected in (
        ("context_len", context_len),
        ("query_len", query_len),
        ("profile_trials", PROFILE_TRIALS),
        ("expected_launches", expected_launches(context_len, query_len)),
    ):
        if result.get(field) != expected:
            reasons.append(f"{field} differs")
    numerical = result.get("numerical")
    if not isinstance(numerical, dict):
        reasons.append("numerical report is missing")
    else:
        if numerical.get("finite") is not True:
            reasons.append("candidate output is nonfinite")
        for field, limit in (
            ("output_relative_l2", production.RELATIVE_L2_LIMIT),
            ("lse_relative_l2", production.RELATIVE_L2_LIMIT),
            ("output_max_abs", production.MAX_ABS_LIMIT),
        ):
            value = numerical.get(field)
            if not _finite_nonnegative(value) or float(value) > limit:
                reasons.append(f"numerical.{field} exceeds {limit:g}")
    profile_cuda_ms = result.get("profile_cuda_ms")
    if not _finite_nonnegative(profile_cuda_ms) or profile_cuda_ms <= 0.0:
        reasons.append("profile CUDA time is invalid")
    if result.get("profiler_start_rc") != 0:
        reasons.append("cudaProfilerStart failed")
    if result.get("profiler_stop_rc") != 0:
        reasons.append("cudaProfilerStop failed")
    if result.get("authorization") != {
        "implementation_direction_authorized": False,
        "tp4_service_authorized": False,
        "main_or_yaml_change_authorized": False,
        "official_score_claim_authorized": False,
    }:
        reasons.append("authorization must fail closed")
    return {"qualified": not reasons, "reasons": reasons}


def _cudart() -> Any:
    library = ctypes.CDLL("libcudart.so")
    library.cudaProfilerStart.restype = ctypes.c_int
    library.cudaProfilerStop.restype = ctypes.c_int
    return library


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "M1-155 requires exactly one visible healthy BI100")
    context_len, query_len = CASES[args.case]
    extension, artifact = production._load_extension(
        args.extension, args.expected_extension_sha256)
    inputs = production._make_inputs(context_len, query_len)
    scale = production.HEAD_DIM ** -0.5
    reference = production.reference_forward(*inputs, context_len, scale)
    candidate = tuple(extension.forward(*inputs, context_len, scale))
    torch.cuda.synchronize()

    runtime = _cudart()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    profiler_start_rc = int(runtime.cudaProfilerStart())
    start_event.record()
    for _ in range(PROFILE_TRIALS):
        candidate = tuple(extension.forward(*inputs, context_len, scale))
    end_event.record()
    torch.cuda.synchronize()
    profiler_stop_rc = int(runtime.cudaProfilerStop())

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
        "source_revision": args.source_revision,
        "instance": args.instance,
        "visible_physical_gpu": args.visible_physical_gpu,
        "case": args.case,
        "context_len": context_len,
        "query_len": query_len,
        "total_kv_len": context_len + query_len,
        "profile_trials": PROFILE_TRIALS,
        "profile_cuda_ms": float(start_event.elapsed_time(end_event)),
        "profiler_start_rc": profiler_start_rc,
        "profiler_stop_rc": profiler_stop_rc,
        "expected_launches": expected_launches(context_len, query_len),
        "extension": artifact,
        "numerical": numerical,
        "authorization": {
            "implementation_direction_authorized": False,
            "tp4_service_authorized": False,
            "main_or_yaml_change_authorized": False,
            "official_score_claim_authorized": False,
        },
        "privacy": {
            "raw_tensors_recorded": False,
            "model_outputs_recorded": False,
            "prompts_recorded": False,
            "credentials_recorded": False,
        },
    }
    result["evaluation"] = evaluate(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=tuple(CASES))
    parser.add_argument("--extension", required=True, type=Path)
    parser.add_argument("--expected-extension-sha256", required=True)
    parser.add_argument("--source-revision", required=True)
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
        ) + "\n",
        encoding="ascii",
    )
    print(json.dumps({
        "case": result["case"],
        "profile_cuda_ms": result["profile_cuda_ms"],
        "qualified": result["evaluation"]["qualified"],
        "reasons": result["evaluation"]["reasons"],
    }, sort_keys=True))
    return 0 if result["evaluation"]["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
