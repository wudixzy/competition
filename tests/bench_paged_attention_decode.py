#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch


def load_extension(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure(case, warmup: int, iterations: int, repeats: int) -> dict:
    for _ in range(warmup):
        case()
    torch.cuda.synchronize()
    trials = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(iterations):
            case()
        torch.cuda.synchronize()
        trials.append((time.perf_counter() - started) * 1000.0 / iterations)
    return {"median_ms": statistics.median(trials), "trials_ms": trials}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--gather-extension", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lengths", default="65536,100000")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sequence-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    direct = load_extension("corex_paged_attention_decode", args.extension)
    gather = load_extension("corex_paged_kv_gather", args.gather_extension)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    lengths = [int(value) for value in args.lengths.split(",")]
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
    scale = head_size ** -0.5

    def reference(query: torch.Tensor, seq_len: int) -> torch.Tensor:
        key, value = gather.gather(
            key_cache, value_cache, block_table, seq_len)
        grouped_query = query.float().view(
            num_kv_heads, num_heads // num_kv_heads, 1, head_size)
        weights = torch.matmul(grouped_query * scale, key.unsqueeze(1))
        weights = torch.softmax(weights, dim=-1)
        output = torch.matmul(weights, value.unsqueeze(1))
        return output.view(1, num_heads, head_size).to(query.dtype)

    results = {}
    for seq_len in lengths:
        query = torch.randn(
            (1, num_heads, head_size), device=device, dtype=torch.float16,
            generator=generator)
        expected = reference(query, seq_len)
        actual = direct.forward(
            query, key_cache, value_cache, block_table, seq_len, scale)
        difference = (actual.float() - expected.float()).abs()
        cases = {
            "gather_reference": lambda q=query, n=seq_len: reference(q, n),
            "direct": lambda q=query, n=seq_len: direct.forward(
                q, key_cache, value_cache, block_table, n, scale),
        }
        timings = {
            name: measure(case, args.warmup, args.iterations, args.repeats)
            for name, case in cases.items()
        }
        timings["direct"]["speedup_vs_gather_reference"] = (
            timings["gather_reference"]["median_ms"] /
            timings["direct"]["median_ms"])

        exact_steps = 0
        close_steps = 0
        sequence_max_abs = 0.0
        for _ in range(args.sequence_steps):
            step_query = torch.randn(
                (1, num_heads, head_size), device=device,
                dtype=torch.float16, generator=generator)
            step_expected = reference(step_query, seq_len)
            step_actual = direct.forward(
                step_query, key_cache, value_cache, block_table, seq_len,
                scale)
            exact_steps += int(torch.equal(step_actual, step_expected))
            close_steps += int(torch.allclose(
                step_actual, step_expected, rtol=1e-3, atol=1e-3))
            sequence_max_abs = max(sequence_max_abs, float((
                step_actual.float() - step_expected.float()).abs().max()))
        results[str(seq_len)] = {
            "exact": bool(torch.equal(actual, expected)),
            "close_rtol_1e-3_atol_1e-3": bool(torch.allclose(
                actual, expected, rtol=1e-3, atol=1e-3)),
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
            "sequence": {
                "exact_steps": exact_steps,
                "close_steps": close_steps,
                "steps": args.sequence_steps,
                "max_abs": sequence_max_abs,
            },
            "timings": timings,
        }

    report = {
        "device": torch.cuda.get_device_name(device),
        "config": vars(args) | {
            "extension": str(args.extension),
            "gather_extension": str(args.gather_extension),
            "out": str(args.out),
        },
        "shape": {
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_size": head_size,
            "block_size": block_size,
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
