#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch


def load_extension(path: Path):
    spec = importlib.util.spec_from_file_location("corex_paged_kv_gather", path)
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
    parser.add_argument("--gather-extension", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lengths", default="65536,100000")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--sequence-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    gather = load_extension(args.gather_extension)
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
    gqa_ratio = num_heads // num_kv_heads

    def matmul_attention(query, key, value):
        grouped_query = query.float().view(
            num_kv_heads, gqa_ratio, 1, head_size)
        weights = torch.matmul(grouped_query * scale, key.unsqueeze(1))
        weights = torch.softmax(weights, dim=-1)
        output = torch.matmul(weights, value.unsqueeze(1))
        return output.view(1, num_heads, head_size).to(query.dtype)

    def bmm_attention(query, key, value):
        flat_query = query.float().view(num_heads, 1, head_size) * scale
        expanded_key = key.repeat_interleave(gqa_ratio, dim=0)
        weights = torch.bmm(flat_query, expanded_key)
        weights = torch.softmax(weights, dim=-1)
        expanded_value = value.repeat_interleave(gqa_ratio, dim=0)
        output = torch.bmm(weights, expanded_value)
        return output.view(1, num_heads, head_size).to(query.dtype)

    def bmm_expand_attention(query, key, value):
        flat_query = query.float().view(num_heads, 1, head_size) * scale
        expanded_key = key.expand(num_heads, head_size, key.shape[-1])
        weights = torch.bmm(flat_query, expanded_key)
        weights = torch.softmax(weights, dim=-1)
        expanded_value = value.expand(num_heads, value.shape[-2], head_size)
        output = torch.bmm(weights, expanded_value)
        return output.view(1, num_heads, head_size).to(query.dtype)

    variants = {
        "matmul": matmul_attention,
        "bmm_repeat": bmm_attention,
        "bmm_expand": bmm_expand_attention,
    }
    results = {}
    for seq_len in lengths:
        key, value = gather.gather(
            key_cache, value_cache, block_table, seq_len)
        query = torch.randn(
            (1, num_heads, head_size), device=device, dtype=torch.float16,
            generator=generator)
        expected = matmul_attention(query, key, value)
        checks = {}
        attention_timings = {}
        full_timings = {}
        for name, variant in variants.items():
            actual = variant(query, key, value)
            checks[name] = {
                "exact": bool(torch.equal(actual, expected)),
                "max_abs": float((
                    actual.float() - expected.float()).abs().max()),
            }
            attention_timings[name] = measure(
                lambda fn=variant: fn(query, key, value),
                args.warmup, args.iterations, args.repeats)
            full_timings[name] = measure(
                lambda fn=variant: fn(query, *gather.gather(
                    key_cache, value_cache, block_table, seq_len)),
                args.warmup, args.iterations, args.repeats)

        baseline_attention = attention_timings["matmul"]["median_ms"]
        baseline_full = full_timings["matmul"]["median_ms"]
        for timing in attention_timings.values():
            timing["speedup_vs_matmul"] = baseline_attention / timing["median_ms"]
        for timing in full_timings.values():
            timing["speedup_vs_matmul"] = baseline_full / timing["median_ms"]

        sequence = {}
        for name, variant in variants.items():
            if name == "matmul":
                continue
            exact_steps = 0
            max_abs = 0.0
            for _ in range(args.sequence_steps):
                step_query = torch.randn(
                    (1, num_heads, head_size), device=device,
                    dtype=torch.float16, generator=generator)
                step_expected = matmul_attention(step_query, key, value)
                step_actual = variant(step_query, key, value)
                exact_steps += int(torch.equal(step_actual, step_expected))
                max_abs = max(max_abs, float((
                    step_actual.float() - step_expected.float()).abs().max()))
            sequence[name] = {
                "exact_steps": exact_steps,
                "steps": args.sequence_steps,
                "max_abs": max_abs,
            }
        results[str(seq_len)] = {
            "checks": checks,
            "sequence": sequence,
            "attention_timings": attention_timings,
            "full_timings": full_timings,
        }

    report = {
        "device": torch.cuda.get_device_name(device),
        "config": vars(args) | {
            "gather_extension": str(args.gather_extension),
            "out": str(args.out),
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
