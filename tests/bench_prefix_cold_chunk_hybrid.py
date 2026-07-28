#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Callable

import torch

from bench_prefix_attention_breakdown import (
    make_case,
    percentile,
    run_segment,
)


HYBRID_QUERY_LEN = 8192


def randomize_cache(tensors: dict[str, torch.Tensor], seed: int) -> None:
    generator = torch.Generator(device=tensors["key_cache"].device)
    generator.manual_seed(seed)
    tensors["key_cache"].normal_(mean=0.0, std=0.02, generator=generator)
    tensors["value_cache"].normal_(mean=0.0, std=0.02, generator=generator)


def hybrid_segment(
    tensors: dict[str, torch.Tensor],
    context_len: int,
    tile_size: int,
) -> torch.Tensor:
    query = tensors["query"]
    if query.shape[0] != HYBRID_QUERY_LEN:
        return run_segment(tensors, context_len, tile_size, False)[0]

    key = tensors["key"]
    value = tensors["value"]
    key_cache = tensors["key_cache"]
    value_cache = tensors["value_cache"]
    block_table = tensors["block_table"]

    query_len, num_query_heads, head_dim = query.shape
    num_kv_heads = key_cache.shape[1]
    gqa_ratio = num_query_heads // num_kv_heads
    block_size = value_cache.shape[3]
    scale = head_dim ** -0.5
    original_dtype = query.dtype

    query_seq = (query.permute(1, 0, 2).float()
                 .view(num_kv_heads, gqa_ratio, query_len, head_dim)
                 .mul(scale))
    running_max = torch.full(
        (num_kv_heads, gqa_ratio, query_len), float("-inf"),
        dtype=torch.float32, device=query.device)
    running_sum = torch.zeros_like(running_max)
    running_output = torch.zeros(
        (num_kv_heads, gqa_ratio, query_len, head_dim),
        dtype=torch.float32, device=query.device)

    if context_len > 0:
        running_lse = torch.full_like(running_max, float("-inf"))
        normalized_output = torch.zeros_like(running_output)

        for block_start in range(0, context_len, tile_size):
            block_end = min(block_start + tile_size, context_len)
            first_block = block_start // block_size
            last_block = (block_end + block_size - 1) // block_size
            block_ids = block_table[first_block:last_block]
            key_blocks = (key_cache[block_ids]
                          .permute(0, 3, 1, 2, 4)
                          .contiguous()
                          .view(-1, num_kv_heads, head_dim))
            value_blocks = (value_cache[block_ids]
                            .permute(0, 3, 1, 2)
                            .contiguous()
                            .view(-1, num_kv_heads, head_dim))
            offset = block_start - first_block * block_size
            length = block_end - block_start
            key_context = key_blocks[offset:offset + length]
            value_context = value_blocks[offset:offset + length]
            k_t = (key_context.permute(1, 0, 2)
                   .unsqueeze(1).transpose(-1, -2).float())
            v_t = value_context.permute(1, 0, 2).unsqueeze(1).float()

            scores = torch.matmul(query_seq, k_t)
            block_max, max_index = scores.max(dim=-1)
            probabilities = torch.softmax(scores, dim=-1)
            max_probability = probabilities.gather(
                -1, max_index.unsqueeze(-1)).squeeze(-1)
            tile_lse = block_max - torch.log(max_probability)
            tile_output = torch.matmul(probabilities, v_t)

            new_lse = torch.logaddexp(running_lse, tile_lse)
            old_weight = torch.exp(running_lse - new_lse)
            tile_weight = torch.exp(tile_lse - new_lse)
            normalized_output.mul_(old_weight.unsqueeze(-1)).add_(
                tile_output * tile_weight.unsqueeze(-1))
            running_lse = new_lse
            running_max.copy_(torch.maximum(running_max, block_max))

        # Restore the same global-max scale used by the reference (m, l, o)
        # recurrence before processing masked current-chunk tiles.
        restored_sum = torch.exp(running_lse - running_max)
        running_sum.copy_(restored_sum)
        running_output.copy_(
            normalized_output * restored_sum.unsqueeze(-1))

    for key_start in range(0, query_len, tile_size):
        key_end = min(key_start + tile_size, query_len)
        k_t = (key[key_start:key_end].permute(1, 0, 2)
               .unsqueeze(1).transpose(-1, -2).float())
        v_t = (value[key_start:key_end].permute(1, 0, 2)
               .unsqueeze(1).float())
        scores = torch.matmul(query_seq, k_t)
        key_positions = torch.arange(key_start, key_end, device=query.device)
        query_positions = torch.arange(query_len, device=query.device)
        mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        block_max = scores.amax(dim=-1)
        new_max = torch.maximum(running_max, block_max)
        exp_scores = scores - new_max.unsqueeze(-1)
        exp_scores.exp_()
        correction = torch.exp(running_max - new_max)
        running_max.copy_(new_max)
        running_sum.mul_(correction).add_(exp_scores.sum(dim=-1))
        running_output.mul_(correction.unsqueeze(-1)).add_(
            torch.matmul(exp_scores, v_t))

    running_output.div_(running_sum.unsqueeze(-1))
    return (running_output.view(num_query_heads, query_len, head_dim)
            .permute(1, 0, 2).to(original_dtype))


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    delta = (reference.float() - candidate.float()).abs()
    denominator = torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
    return {
        "finite": bool(
            torch.isfinite(reference).all() and torch.isfinite(candidate).all()),
        "max_abs": float(delta.max().item()),
        "relative_l2": float(
            (torch.linalg.vector_norm(delta) / denominator).item()),
        "exact": bool(torch.equal(reference, candidate)),
    }


def measure(operation: Callable[[], torch.Tensor],
            warmup: int, repeats: int) -> list[float]:
    trials = []
    for trial in range(warmup + repeats):
        torch.cuda.synchronize()
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        started.record()
        operation()
        finished.record()
        finished.synchronize()
        if trial >= warmup:
            trials.append(float(started.elapsed_time(finished)))
    return trials


def parse_cases(raw: str) -> list[tuple[int, int]]:
    cases = []
    for item in raw.split(","):
        query_len, context_len = item.split(":", 1)
        cases.append((int(query_len), int(context_len)))
    return cases


def run_case(
    query_len: int,
    context_len: int,
    tile_size: int,
    device: torch.device,
    seed: int,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    tensors = make_case(
        query_len, context_len, device, seed,
        num_query_heads=6, num_kv_heads=1,
        head_dim=256, block_size=16)
    randomize_cache(tensors, seed + context_len)

    def baseline() -> torch.Tensor:
        return run_segment(tensors, context_len, tile_size, False)[0]

    def candidate() -> torch.Tensor:
        return hybrid_segment(tensors, context_len, tile_size)

    reference = baseline()
    actual = candidate()
    torch.cuda.synchronize()
    parity = compare(reference, actual)
    baseline_trials = measure(baseline, warmup, repeats)
    candidate_trials = measure(candidate, warmup, repeats)
    baseline_ms = statistics.median(baseline_trials)
    candidate_ms = statistics.median(candidate_trials)
    return {
        "parity": parity,
        "baseline": {
            "median_ms": baseline_ms,
            "p10_ms": percentile(baseline_trials, 10),
            "p90_ms": percentile(baseline_trials, 90),
            "trials_ms": baseline_trials,
        },
        "candidate": {
            "median_ms": candidate_ms,
            "p10_ms": percentile(candidate_trials, 10),
            "p90_ms": percentile(candidate_trials, 90),
            "trials_ms": candidate_trials,
        },
        "speedup": baseline_ms / candidate_ms,
        "reduction": 1.0 - candidate_ms / baseline_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cases", default="456:234544,8192:65536")
    parser.add_argument("--partial-context", type=int, default=65552)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--max-abs-gate", type=float, default=1e-3)
    parser.add_argument("--relative-l2-gate", type=float, default=1e-5)
    parser.add_argument("--cold-reduction-gate", type=float, default=0.15)
    parser.add_argument("--warm-regression-gate", type=float, default=0.02)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    cases = {}
    for index, (query_len, context_len) in enumerate(parse_cases(args.cases)):
        cases[str(query_len)] = run_case(
            query_len, context_len, args.tile_size, device,
            args.seed + index, args.warmup, args.repeats)

    partial = run_case(
        HYBRID_QUERY_LEN, args.partial_context, args.tile_size, device,
        args.seed + 100, warmup=0, repeats=1)
    parity_ok = all(
        case["parity"]["finite"]
        and case["parity"]["max_abs"] <= args.max_abs_gate
        and case["parity"]["relative_l2"] <= args.relative_l2_gate
        for case in [*cases.values(), partial])
    performance_ok = (
        cases["456"]["reduction"] >= -args.warm_regression_gate
        and cases["8192"]["reduction"] >= args.cold_reduction_gate)
    report = {
        "ok": bool(parity_ok and performance_ok),
        "parity_ok": parity_ok,
        "performance_ok": performance_ok,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "config": {
            "hybrid_query_len": HYBRID_QUERY_LEN,
            "cases": args.cases,
            "partial_context": args.partial_context,
            "tile_size": args.tile_size,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "max_abs_gate": args.max_abs_gate,
            "relative_l2_gate": args.relative_l2_gate,
            "cold_reduction_gate": args.cold_reduction_gate,
            "warm_regression_gate": args.warm_regression_gate,
        },
        "cases": cases,
        "partial_tile_case": partial,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
