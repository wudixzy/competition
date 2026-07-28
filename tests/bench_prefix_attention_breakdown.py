#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import torch


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def parse_cases(raw: str) -> list[tuple[int, int]]:
    cases = []
    for item in raw.split(","):
        query_len, context_len = item.split(":", 1)
        parsed = (int(query_len), int(context_len))
        if parsed[0] <= 0 or parsed[1] < 0:
            raise argparse.ArgumentTypeError(
                "case lengths must satisfy query > 0 and context >= 0")
        cases.append(parsed)
    return cases


def new_event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def timed_stage(
    events: Optional[list[tuple[str, torch.cuda.Event, torch.cuda.Event]]],
    name: str,
    operation,
):
    if events is None:
        return operation()
    started = new_event()
    finished = new_event()
    started.record()
    result = operation()
    finished.record()
    events.append((name, started, finished))
    return result


def make_case(
    query_len: int,
    context_len: int,
    device: torch.device,
    seed: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    dtype = torch.float16
    key_pack = 16 // torch.tensor([], dtype=dtype).element_size()
    num_blocks = max(1, (context_len + block_size - 1) // block_size)
    query = torch.randn(
        (query_len, num_query_heads, head_dim), device=device, dtype=dtype,
        generator=generator) * 0.02
    key = torch.randn(
        (query_len, num_kv_heads, head_dim), device=device, dtype=dtype,
        generator=generator) * 0.02
    value = torch.randn(
        (query_len, num_kv_heads, head_dim), device=device, dtype=dtype,
        generator=generator) * 0.02
    key_cache = torch.zeros(
        (num_blocks, num_kv_heads, head_dim // key_pack,
         block_size, key_pack),
        device=device, dtype=dtype)
    value_cache = torch.zeros(
        (num_blocks, num_kv_heads, head_dim, block_size),
        device=device, dtype=dtype)
    block_table = torch.randperm(
        num_blocks, device=device, dtype=torch.int64,
        generator=generator).to(torch.int32)
    return {
        "query": query,
        "key": key,
        "value": value,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_table": block_table,
    }


def run_segment(
    tensors: dict[str, torch.Tensor],
    context_len: int,
    tile_size: int,
    record_events: bool,
) -> tuple[torch.Tensor, Optional[list[tuple[str, torch.cuda.Event,
                                               torch.cuda.Event]]]]:
    query = tensors["query"]
    key = tensors["key"]
    value = tensors["value"]
    key_cache = tensors["key_cache"]
    value_cache = tensors["value_cache"]
    block_table = tensors["block_table"]
    events = [] if record_events else None

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

    def update(k_t: torch.Tensor, v_t: torch.Tensor,
               prefix: str, mask: Optional[torch.Tensor] = None) -> None:
        scores = timed_stage(
            events,
            f"{prefix}.qk",
            lambda: torch.matmul(query_seq, k_t),
        )
        if mask is not None:
            timed_stage(
                events,
                f"{prefix}.causal_mask",
                lambda: scores.masked_fill_(
                    mask.unsqueeze(0).unsqueeze(0), float("-inf")),
            )

        pointwise: dict[str, torch.Tensor] = {}

        def softmax_update() -> None:
            block_max = scores.amax(dim=-1)
            new_max = torch.maximum(running_max, block_max)
            exp_scores = scores - new_max.unsqueeze(-1)
            exp_scores.exp_()
            correction = torch.exp(running_max - new_max)
            running_max.copy_(new_max)
            running_sum.mul_(correction).add_(exp_scores.sum(dim=-1))
            pointwise["exp_scores"] = exp_scores
            pointwise["correction"] = correction

        timed_stage(events, f"{prefix}.softmax", softmax_update)
        product = timed_stage(
            events,
            f"{prefix}.pv",
            lambda: torch.matmul(pointwise["exp_scores"], v_t),
        )

        def output_update() -> None:
            running_output.mul_(pointwise["correction"].unsqueeze(-1))
            running_output.add_(product)

        timed_stage(events, f"{prefix}.state_update", output_update)

    for block_start in range(0, context_len, tile_size):
        block_end = min(block_start + tile_size, context_len)

        def gather_context() -> tuple[torch.Tensor, torch.Tensor]:
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
            return k_t, v_t

        k_t, v_t = timed_stage(
            events, "context.gather_prepare", gather_context)
        update(k_t, v_t, "context")

    for key_start in range(0, query_len, tile_size):
        key_end = min(key_start + tile_size, query_len)

        def prepare_current() -> tuple[torch.Tensor, torch.Tensor,
                                       torch.Tensor]:
            k_t = (key[key_start:key_end].permute(1, 0, 2)
                   .unsqueeze(1).transpose(-1, -2).float())
            v_t = (value[key_start:key_end].permute(1, 0, 2)
                   .unsqueeze(1).float())
            key_positions = torch.arange(
                key_start, key_end, device=query.device)
            query_positions = torch.arange(query_len, device=query.device)
            mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
            return k_t, v_t, mask

        k_t, v_t, mask = timed_stage(
            events, "current.prepare", prepare_current)
        update(k_t, v_t, "current", mask)

    running_output.div_(running_sum.unsqueeze(-1))
    output = (running_output.view(
        num_query_heads, query_len, head_dim)
        .permute(1, 0, 2).to(original_dtype))
    return output, events


def summarize_events(
    events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]],
    wall_ms: float,
) -> dict[str, Any]:
    stages: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for name, started, finished in events:
        stages[name] += float(started.elapsed_time(finished))
        counts[name] += 1
    gpu_total = sum(stages.values())
    return {
        "wall_ms": wall_ms,
        "gpu_stage_total_ms": gpu_total,
        "unattributed_wall_ms": wall_ms - gpu_total,
        "stages_ms": dict(stages),
        "stage_counts": dict(counts),
        "gpu_stage_share_pct": {
            name: value * 100.0 / gpu_total
            for name, value in stages.items()
        },
        "wall_share_pct": {
            name: value * 100.0 / wall_ms
            for name, value in stages.items()
        },
    }


def benchmark_case(
    query_len: int,
    context_len: int,
    args: argparse.Namespace,
    device: torch.device,
    case_seed: int,
) -> dict[str, Any]:
    tensors = make_case(
        query_len, context_len, device, case_seed,
        args.query_heads, args.kv_heads, args.head_dim, args.block_size)
    torch.cuda.synchronize()

    for _ in range(args.warmup):
        run_segment(tensors, context_len, args.tile_size, False)
    torch.cuda.synchronize()

    trials = []
    finite = True
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        output, _ = run_segment(
            tensors, context_len, args.tile_size, False)
        torch.cuda.synchronize()
        trials.append((time.perf_counter() - started) * 1000.0)
        finite = finite and bool(torch.isfinite(output).all())

    torch.cuda.synchronize()
    profile_started = time.perf_counter()
    profile_output, events = run_segment(
        tensors, context_len, args.tile_size, True)
    torch.cuda.synchronize()
    profile_wall_ms = (time.perf_counter() - profile_started) * 1000.0
    finite = finite and bool(torch.isfinite(profile_output).all())
    assert events is not None

    allocated = int(torch.cuda.memory_allocated())
    reserved = int(torch.cuda.memory_reserved())
    result = {
        "query_len": query_len,
        "context_len": context_len,
        "context_tiles": (context_len + args.tile_size - 1)
        // args.tile_size,
        "current_tiles": (query_len + args.tile_size - 1)
        // args.tile_size,
        "finite": finite,
        "full_boundary": {
            "median_ms": statistics.median(trials),
            "p10_ms": percentile(trials, 10),
            "p90_ms": percentile(trials, 90),
            "trials_ms": trials,
        },
        "event_profile": summarize_events(events, profile_wall_ms),
        "memory": {
            "allocated_bytes_after_profile": allocated,
            "reserved_bytes_after_profile": reserved,
        },
    }
    del tensors, output, profile_output
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--cases", default="456:234544,8192:65536",
        help="comma-separated query_len:context_len pairs")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--query-heads", type=int, default=6)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.tile_size <= 0 or args.block_size <= 0:
        parser.error("tile and block sizes must be positive")
    if args.query_heads % args.kv_heads:
        parser.error("query heads must be divisible by KV heads")

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    cases = parse_cases(args.cases)
    results = {}
    for index, (query_len, context_len) in enumerate(cases):
        key = f"q{query_len}_ctx{context_len}"
        results[key] = benchmark_case(
            query_len, context_len, args, device, args.seed + index)

    report = {
        "ok": all(case["finite"] for case in results.values()),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "config": vars(args) | {"out": str(args.out)},
        "cases": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
