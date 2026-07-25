#!/usr/bin/env python3
"""Fixed production-shape gate for BI100 paged-prefill candidates.

This is a single-GPU component benchmark. It cannot qualify a TP4 service,
model quality, or a submission configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable


SCHEMA = "bi100-m1-55-production-prefill-cell-v1"
BLOCK_SIZE = 16
BLOCKS_PER_TILE = 32
TILE_TOKENS = BLOCK_SIZE * BLOCKS_PER_TILE
HEAD_DIM = 256
NUM_QUERY_HEADS = 4
NUM_KV_HEADS = 1
WARMUPS = 1
TRIALS = 3
RELATIVE_L2_LIMIT = 1e-5
MAX_ABS_LIMIT = 1e-3
MIN_PRODUCTION_SPEEDUP = 1.5
EXTENSION_MODULE_NAME = "corex_fused_paged_prefill"
SEED = 20260725

CASES = {
    "golden_dense_q1": (0, 1, "numerical"),
    "golden_dense_q8": (0, 8, "numerical"),
    "golden_dense_q256": (0, 256, "numerical"),
    "golden_paged_240_q16": (240, 16, "numerical"),
    "boundary_65520_q16": (65_520, 16, "numerical"),
    "boundary_234992_q8": (234_992, 8, "numerical"),
    "legacy_74k_q256": (73_728, 256, "legacy"),
    "legacy_128k_q256": (130_816, 256, "legacy"),
    "legacy_235k_q256": (234_736, 256, "legacy"),
    "production_dense_q8176": (0, 8_176, "production"),
    "production_65k_q8176": (65_536, 8_176, "production"),
    "production_128k_q8176": (122_880, 8_176, "production"),
    "production_235k_q5616": (229_376, 5_616, "production"),
    "fallback_q16": (8_176, 16, "fallback"),
    "fallback_q8": (229_376, 8, "fallback"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split4_aux_workspace_bytes(query_len: int) -> int:
    """Exact auxiliary FP32 allocation made by the M1-47 split4 source."""
    rows = NUM_QUERY_HEADS * query_len
    split_count = 4
    converted_query = rows * HEAD_DIM
    key_value_tiles = 2 * split_count * TILE_TOKENS * HEAD_DIM
    scores = split_count * rows * TILE_TOKENS
    split_output = split_count * rows * HEAD_DIM
    running_max_sum = 2 * rows
    running_output = rows * HEAD_DIM
    return 4 * (
        converted_query
        + key_value_tiles
        + scores
        + split_output
        + running_max_sum
        + running_output
    )


def _relative_l2(actual: Any, expected: Any) -> float:
    difference = (actual.float() - expected.float()).norm().item()
    denominator = expected.float().norm().item()
    if denominator == 0:
        return 0.0 if difference == 0 else math.inf
    return difference / denominator


def _update_online(
    scores: Any,
    value: Any,
    running_max: Any,
    running_sum: Any,
    running_output: Any,
) -> None:
    import torch

    block_max = scores.amax(dim=-1)
    new_max = torch.maximum(running_max, block_max)
    correction = torch.exp(running_max - new_max)
    probabilities = scores.sub(new_max.unsqueeze(-1)).exp_()
    running_sum.mul_(correction).add_(probabilities.sum(dim=-1))
    running_output.mul_(correction.unsqueeze(-1)).add_(
        torch.matmul(probabilities, value))
    running_max.copy_(new_max)


def reference_forward(
    query: Any,
    key_new: Any,
    value_new: Any,
    key_cache: Any,
    value_cache: Any,
    block_table: Any,
    context_len: int,
    scale: float,
) -> tuple[Any, Any]:
    """Match the installed K-major FP32 online-softmax partitioning."""
    import torch

    query_len = query.shape[0]
    query_fp32 = (
        query.permute(1, 0, 2).float().mul(scale).unsqueeze(0))
    running_max = torch.full(
        (1, NUM_QUERY_HEADS, query_len),
        float("-inf"),
        dtype=torch.float32,
        device=query.device,
    )
    running_sum = torch.zeros_like(running_max)
    running_output = torch.zeros(
        (1, NUM_QUERY_HEADS, query_len, HEAD_DIM),
        dtype=torch.float32,
        device=query.device,
    )

    for token_start in range(0, context_len, TILE_TOKENS):
        token_end = min(token_start + TILE_TOKENS, context_len)
        first_block = token_start // BLOCK_SIZE
        last_block = (token_end + BLOCK_SIZE - 1) // BLOCK_SIZE
        block_ids = block_table[first_block:last_block]
        key = (
            key_cache[block_ids]
            .permute(0, 3, 1, 2, 4)
            .contiguous()
            .view(-1, NUM_KV_HEADS, HEAD_DIM)
        )[:token_end - token_start]
        value = (
            value_cache[block_ids]
            .permute(0, 3, 1, 2)
            .contiguous()
            .view(-1, NUM_KV_HEADS, HEAD_DIM)
        )[:token_end - token_start]
        key_matrix = (
            key.permute(1, 0, 2).unsqueeze(1).transpose(-1, -2).float())
        value_matrix = value.permute(1, 0, 2).unsqueeze(1).float()
        _update_online(
            torch.matmul(query_fp32, key_matrix),
            value_matrix,
            running_max,
            running_sum,
            running_output,
        )

    key_positions = torch.arange(query_len, device=query.device)
    query_positions = torch.arange(query_len, device=query.device)
    for key_start in range(0, query_len, TILE_TOKENS):
        key_end = min(key_start + TILE_TOKENS, query_len)
        key_matrix = (
            key_new[key_start:key_end]
            .permute(1, 0, 2)
            .unsqueeze(1)
            .transpose(-1, -2)
            .float()
        )
        value_matrix = (
            value_new[key_start:key_end]
            .permute(1, 0, 2)
            .unsqueeze(1)
            .float()
        )
        scores = torch.matmul(query_fp32, key_matrix)
        mask = (
            key_positions[key_start:key_end].unsqueeze(0)
            > query_positions.unsqueeze(1)
        )
        scores.masked_fill_(
            mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        _update_online(
            scores,
            value_matrix,
            running_max,
            running_sum,
            running_output,
        )

    output = (
        running_output.div(running_sum.unsqueeze(-1))
        .squeeze(0)
        .permute(1, 0, 2)
        .to(query.dtype)
        .contiguous()
    )
    lse = (
        running_max.add(torch.log(running_sum))
        .squeeze(0)
        .transpose(0, 1)
        .contiguous()
    )
    return output, lse


def _load_extension(path: Path, expected_sha256: str) -> tuple[Any, dict[str, Any]]:
    resolved = path.resolve(strict=True)
    actual_sha256 = sha256_file(resolved)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "extension SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}")
    spec = importlib.util.spec_from_file_location(
        EXTENSION_MODULE_NAME, resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension spec from {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "forward", None)):
        raise RuntimeError("extension does not expose callable forward")
    return module, {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": actual_sha256,
    }


def _make_inputs(context_len: int, query_len: int) -> tuple[Any, ...]:
    import torch

    torch.manual_seed(SEED)
    device = torch.device("cuda")
    query = torch.randn(
        (query_len, NUM_QUERY_HEADS, HEAD_DIM),
        dtype=torch.float16,
        device=device,
    )
    key_new = torch.randn(
        (query_len, NUM_KV_HEADS, HEAD_DIM),
        dtype=torch.float16,
        device=device,
    )
    value_new = torch.randn_like(key_new)
    required_blocks = (context_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    physical_blocks = max(1, required_blocks + 17)
    key_cache = torch.randn(
        (
            physical_blocks,
            NUM_KV_HEADS,
            HEAD_DIM // 8,
            BLOCK_SIZE,
            8,
        ),
        dtype=torch.float16,
        device=device,
    )
    value_cache = torch.randn(
        (physical_blocks, NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE),
        dtype=torch.float16,
        device=device,
    )
    if required_blocks:
        logical = torch.arange(required_blocks, dtype=torch.int32)
        block_table = (
            logical.roll(7).add(11).remainder(physical_blocks).cuda())
        if torch.unique(block_table).numel() != required_blocks:
            block_table = logical.roll(7).cuda()
    else:
        block_table = torch.empty(0, dtype=torch.int32, device=device)
    return query, key_new, value_new, key_cache, value_cache, block_table


def _measure(
    function: Callable[[], tuple[Any, Any]],
) -> tuple[dict[str, Any], tuple[Any, Any]]:
    import torch

    for _ in range(WARMUPS):
        output = function()
        torch.cuda.synchronize()
        del output

    cuda_trials_ms = []
    host_trials_ms = []
    result = None
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated_bytes = int(torch.cuda.memory_allocated())
    for _ in range(TRIALS):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        host_start = time.perf_counter()
        start_event.record()
        result = function()
        end_event.record()
        torch.cuda.synchronize()
        host_trials_ms.append((time.perf_counter() - host_start) * 1000.0)
        cuda_trials_ms.append(float(start_event.elapsed_time(end_event)))
    assert result is not None
    return {
        "warmups": WARMUPS,
        "trials": TRIALS,
        "cuda_trials_ms": cuda_trials_ms,
        "cuda_median_ms": statistics.median(cuda_trials_ms),
        "host_trials_ms": host_trials_ms,
        "host_median_ms": statistics.median(host_trials_ms),
        "baseline_allocated_bytes": baseline_allocated_bytes,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_incremental_bytes": int(
            torch.cuda.max_memory_allocated() - baseline_allocated_bytes),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }, result


def _finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def evaluate_cell(result: Any) -> dict[str, Any]:
    reasons: list[str] = []
    if not isinstance(result, dict):
        return {"qualified": False, "reasons": ["result must be an object"]}
    if result.get("schema") != SCHEMA:
        reasons.append(f"schema must equal {SCHEMA}")
    case_name = result.get("case")
    if case_name not in CASES:
        reasons.append("case is not in the frozen matrix")
        return {"qualified": False, "reasons": reasons}
    context_len, query_len, kind = CASES[case_name]
    for field, expected in (
        ("context_len", context_len),
        ("query_len", query_len),
        ("kind", kind),
        ("seed", SEED),
        ("warmups", WARMUPS),
        ("trials", TRIALS),
    ):
        if result.get(field) != expected:
            reasons.append(f"{field} must equal {expected!r}")
    expected_permutation = context_len > 0
    if result.get("physical_block_permutation") is not expected_permutation:
        reasons.append(
            "physical_block_permutation must describe the frozen case")
    numerical = result.get("numerical")
    if not isinstance(numerical, dict):
        reasons.append("numerical must be an object")
    else:
        if numerical.get("finite") is not True:
            reasons.append("candidate output is not finite")
        for field, limit in (
            ("output_relative_l2", RELATIVE_L2_LIMIT),
            ("lse_relative_l2", RELATIVE_L2_LIMIT),
            ("output_max_abs", MAX_ABS_LIMIT),
        ):
            value = numerical.get(field)
            if not _finite_nonnegative(value):
                reasons.append(f"numerical.{field} is invalid")
            elif value > limit + 1e-12:
                reasons.append(
                    f"numerical.{field}={value:.9g} exceeds {limit:.9g}")
    timings = result.get("timings")
    if not isinstance(timings, dict):
        reasons.append("timings must be an object")
    else:
        for side in ("reference", "candidate"):
            timing = timings.get(side)
            if not isinstance(timing, dict):
                reasons.append(f"timings.{side} must be an object")
                continue
            trials = timing.get("cuda_trials_ms")
            median = timing.get("cuda_median_ms")
            if (
                not isinstance(trials, list)
                or len(trials) != TRIALS
                or not all(_finite_nonnegative(value) and value > 0
                           for value in trials)
            ):
                reasons.append(
                    f"timings.{side}.cuda_trials_ms must have "
                    f"{TRIALS} positive values")
            elif (
                not _finite_nonnegative(median)
                or not math.isclose(
                    float(median),
                    statistics.median(trials),
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            ):
                reasons.append(
                    f"timings.{side}.cuda_median_ms does not match trials")
        speedup = timings.get("speedup")
        if not _finite_nonnegative(speedup):
            reasons.append("timings.speedup is invalid")
        elif kind == "production" and speedup < MIN_PRODUCTION_SPEEDUP:
            reasons.append(
                f"production speedup {speedup:.6f} is below "
                f"{MIN_PRODUCTION_SPEEDUP:.1f}x")
    authorization = result.get("authorization")
    if authorization != {
        "tp4_service_authorized": False,
        "main_or_yaml_change_authorized": False,
        "official_score_claim_authorized": False,
    }:
        reasons.append("authorization must fail closed")
    return {"qualified": not reasons, "reasons": reasons}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/CoreX device is not available")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "benchmark requires exactly one visible device; set "
            "CUDA_VISIBLE_DEVICES to one healthy physical GPU")
    context_len, query_len, kind = CASES[args.case]
    extension, artifact = _load_extension(
        args.extension, args.expected_extension_sha256)
    inputs = _make_inputs(context_len, query_len)
    query, key_new, value_new, key_cache, value_cache, block_table = inputs
    scale = HEAD_DIM ** -0.5
    reference_call = lambda: reference_forward(*inputs, context_len, scale)
    candidate_call = lambda: tuple(
        extension.forward(*inputs, context_len, scale))

    reference_timing, reference = _measure(reference_call)
    candidate_timing, candidate = _measure(candidate_call)
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
        "output_relative_l2": _relative_l2(
            candidate_output, reference_output),
        "lse_relative_l2": _relative_l2(candidate_lse, reference_lse),
    }
    result = {
        "schema": SCHEMA,
        "source_commit": args.source_commit,
        "runtime_identity": args.runtime_identity,
        "instance": args.instance,
        "visible_physical_gpu": args.visible_physical_gpu,
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "case": args.case,
        "kind": kind,
        "context_len": context_len,
        "query_len": query_len,
        "total_kv_len": context_len + query_len,
        "seed": SEED,
        "warmups": WARMUPS,
        "trials": TRIALS,
        "physical_block_permutation": context_len > 0,
        "extension": artifact,
        "split4_aux_workspace_bytes": split4_aux_workspace_bytes(query_len),
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
            "tp4_service_authorized": False,
            "main_or_yaml_change_authorized": False,
            "official_score_claim_authorized": False,
        },
    }
    result["evaluation"] = evaluate_cell(result)
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
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "case": result["case"],
        "qualified": result["evaluation"]["qualified"],
        "speedup": result["timings"]["speedup"],
        "output_relative_l2": result["numerical"]["output_relative_l2"],
        "reasons": result["evaluation"]["reasons"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
