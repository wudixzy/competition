#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(
        "paged_attn_corex_gather_probe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure(case, warmup: int, iterations: int, repeats: int) -> list[float]:
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
    return trials


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paged-attn", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seq-len", type=int, default=100000)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    module = load_module(args.paged_attn)
    if module._corex_paged_kv_gather is None:
        raise RuntimeError("candidate paged_attn did not import the extension")

    generator = torch.Generator(device=device).manual_seed(args.seed)
    num_heads, num_kv_heads, head_size, block_size = 6, 1, 256, 16
    key_pack = 16 // torch.empty((), dtype=torch.float16).element_size()
    num_blocks = (args.seq_len + block_size - 1) // block_size
    key_cache = torch.randn(
        (num_blocks, num_kv_heads, head_size // key_pack,
         block_size, key_pack), device=device, dtype=torch.float16,
        generator=generator)
    value_cache = torch.randn(
        (num_blocks, num_kv_heads, head_size, block_size),
        device=device, dtype=torch.float16, generator=generator)
    query = torch.randn(
        (1, num_heads, head_size), device=device, dtype=torch.float16,
        generator=generator)
    block_tables = torch.randperm(
        num_blocks, device=device, dtype=torch.int32).view(1, -1)
    seq_lens = torch.tensor([args.seq_len], device=device, dtype=torch.int32)
    scale = head_size ** -0.5

    def run(enabled: bool):
        module._USE_COREX_PAGED_KV_GATHER = enabled
        return module.PagedAttention._forward_decode_pytorch(
            query, key_cache, value_cache, block_tables, seq_lens, scale)

    reference = run(False)
    candidate = run(True)
    baseline_trials = measure(
        lambda: run(False), args.warmup, args.iterations, args.repeats)
    candidate_trials = measure(
        lambda: run(True), args.warmup, args.iterations, args.repeats)
    baseline_ms = statistics.median(baseline_trials)
    candidate_ms = statistics.median(candidate_trials)
    report = {
        "device": torch.cuda.get_device_name(device),
        "config": vars(args) | {
            "paged_attn": str(args.paged_attn), "out": str(args.out)},
        "extension_loaded": True,
        "exact": bool(torch.equal(candidate, reference)),
        "max_abs": float((candidate.float() - reference.float()).abs().max()),
        "baseline_median_ms": baseline_ms,
        "candidate_median_ms": candidate_ms,
        "speedup": baseline_ms / candidate_ms,
        "baseline_trials_ms": baseline_trials,
        "candidate_trials_ms": candidate_trials,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
