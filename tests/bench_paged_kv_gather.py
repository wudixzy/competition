#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch


Case = Callable[[], object]


def load_extension(path: Path):
    spec = importlib.util.spec_from_file_location("corex_paged_kv_gather", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def measure(case: Case, warmup: int, iterations: int,
            repeats: int) -> dict[str, object]:
    for _ in range(warmup):
        case()
    torch.cuda.synchronize()
    trials = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            case()
        torch.cuda.synchronize()
        trials.append((time.perf_counter() - started) * 1000.0 / iterations)
    return {
        "median_ms": statistics.median(trials),
        "p10_ms": percentile(trials, 10),
        "p90_ms": percentile(trials, 90),
        "trials_ms": trials,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lengths", default="32768,65536,100000")
    parser.add_argument("--grid-caps", default="")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    extension = load_extension(args.extension)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    lengths = [int(value) for value in args.lengths.split(",")]
    grid_caps = ([int(value) for value in args.grid_caps.split(",")]
                 if args.grid_caps else [])
    num_heads, num_kv_heads, head_size, block_size = 6, 1, 256, 16
    key_pack = 16 // torch.empty((), dtype=torch.float16).element_size()
    max_blocks = (max(lengths) + block_size - 1) // block_size
    key_cache = torch.randn(
        (max_blocks, num_kv_heads, head_size // key_pack,
         block_size, key_pack), device=device, dtype=torch.float16,
        generator=generator)
    value_cache = torch.randn(
        (max_blocks, num_kv_heads, head_size, block_size),
        device=device, dtype=torch.float16, generator=generator)
    block_table = torch.randperm(
        max_blocks, device=device, dtype=torch.int32)
    query = torch.randn(
        (1, num_heads, head_size), device=device, dtype=torch.float16,
        generator=generator)
    scale = head_size ** -0.5

    def native_gather(seq_len: int):
        num_blocks = (seq_len + block_size - 1) // block_size
        block_ids = block_table[:num_blocks]
        key = (key_cache[block_ids]
               .permute(0, 3, 1, 2, 4)
               .contiguous()
               .view(-1, num_kv_heads, head_size))[:seq_len] \
              .permute(1, 2, 0).contiguous().float()
        value = (value_cache[block_ids]
                 .permute(0, 3, 1, 2)
                 .contiguous()
                 .view(-1, num_kv_heads, head_size))[:seq_len] \
                .permute(1, 0, 2).contiguous().float()
        return key, value

    def attention(key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        grouped_query = query.float().view(
            num_kv_heads, num_heads // num_kv_heads, 1, head_size)
        weights = torch.matmul(grouped_query * scale, key.unsqueeze(1))
        weights = torch.softmax(weights, dim=-1)
        output = torch.matmul(weights, value.unsqueeze(1))
        return output.view(1, num_heads, head_size).to(query.dtype)

    results = {}
    for seq_len in lengths:
        native_key, native_value = native_gather(seq_len)
        custom_key, custom_value = extension.gather(
            key_cache, value_cache, block_table, seq_len)
        native_output = attention(native_key, native_value)
        custom_output = attention(custom_key, custom_value)
        checks = {
            "key_exact": bool(torch.equal(custom_key, native_key)),
            "value_exact": bool(torch.equal(custom_value, native_value)),
            "output_exact": bool(torch.equal(custom_output, native_output)),
            "output_max_abs": float(
                (custom_output.float() - native_output.float()).abs().max()),
        }
        cases = {
            "native_gather": lambda n=seq_len: native_gather(n),
            "custom_gather": lambda n=seq_len: extension.gather(
                key_cache, value_cache, block_table, n),
            "native_full": lambda n=seq_len: attention(*native_gather(n)),
            "custom_full": lambda n=seq_len: attention(*extension.gather(
                key_cache, value_cache, block_table, n)),
        }
        timings = {
            name: measure(case, args.warmup, args.iterations, args.repeats)
            for name, case in cases.items()
        }
        timings["custom_gather"]["speedup_vs_native"] = (
            timings["native_gather"]["median_ms"] /
            timings["custom_gather"]["median_ms"])
        timings["custom_full"]["speedup_vs_native"] = (
            timings["native_full"]["median_ms"] /
            timings["custom_full"]["median_ms"])
        grid_scan = {}
        for grid_cap in grid_caps:
            grid_key, grid_value = extension.gather_grid(
                key_cache, value_cache, block_table, seq_len, grid_cap)
            grid_output = attention(grid_key, grid_value)
            gather_case = lambda cap=grid_cap, n=seq_len: \
                extension.gather_grid(
                    key_cache, value_cache, block_table, n, cap)
            full_case = lambda cap=grid_cap, n=seq_len: attention(
                *extension.gather_grid(
                    key_cache, value_cache, block_table, n, cap))
            gather_timing = measure(
                gather_case, args.warmup, args.iterations, args.repeats)
            full_timing = measure(
                full_case, args.warmup, args.iterations, args.repeats)
            gather_timing["speedup_vs_native"] = (
                timings["native_gather"]["median_ms"] /
                gather_timing["median_ms"])
            full_timing["speedup_vs_native"] = (
                timings["native_full"]["median_ms"] /
                full_timing["median_ms"])
            grid_scan[str(grid_cap)] = {
                "key_exact": bool(torch.equal(grid_key, native_key)),
                "value_exact": bool(torch.equal(grid_value, native_value)),
                "output_exact": bool(torch.equal(grid_output, native_output)),
                "output_max_abs": float((
                    grid_output.float() - native_output.float()).abs().max()),
                "gather": gather_timing,
                "full": full_timing,
            }
        results[str(seq_len)] = {
            "checks": checks, "timings": timings, "grid_scan": grid_scan}

    report = {
        "device": torch.cuda.get_device_name(device),
        "config": vars(args) | {
            "extension": str(args.extension), "out": str(args.out)},
        "shape": {
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_size": head_size,
            "block_size": block_size,
            "dtype": "float16",
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
