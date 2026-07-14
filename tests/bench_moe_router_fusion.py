#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def measure(case: Callable[[], object], warmup: int,
            iterations: int, repeats: int) -> list[float]:
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
    return trials


def benchmark_shape(device: torch.device, tokens: int, hidden_size: int,
                    num_experts: int, generator: torch.Generator,
                    warmup: int, iterations: int, repeats: int) -> dict:
    hidden = torch.randn(
        (tokens, hidden_size), device=device, dtype=torch.float16,
        generator=generator)
    router_weight = torch.randn(
        (num_experts, hidden_size), device=device, dtype=torch.float16,
        generator=generator)
    shared_gate_weight = torch.randn(
        (1, hidden_size), device=device, dtype=torch.float16,
        generator=generator)
    fused_weight = torch.cat((router_weight, shared_gate_weight), dim=0)

    def separate() -> tuple[torch.Tensor, torch.Tensor]:
        router = F.linear(hidden, router_weight)
        shared_gate = F.linear(hidden, shared_gate_weight)
        return router, shared_gate

    def fused() -> tuple[torch.Tensor, torch.Tensor]:
        output = F.linear(hidden, fused_weight)
        return output[..., :-1], output[..., -1:]

    reference_router, reference_gate = separate()
    candidate_router, candidate_gate = fused()
    torch.cuda.synchronize()
    router_difference = (
        candidate_router.float() - reference_router.float()).abs()
    gate_difference = (
        candidate_gate.float() - reference_gate.float()).abs()
    separate_trials = measure(separate, warmup, iterations, repeats)
    fused_trials = measure(fused, warmup, iterations, repeats)
    separate_median = statistics.median(separate_trials)
    fused_median = statistics.median(fused_trials)
    return {
        "tokens": tokens,
        "exact": bool(
            torch.equal(candidate_router, reference_router)
            and torch.equal(candidate_gate, reference_gate)),
        "max_abs": max(
            float(router_difference.max()), float(gate_difference.max())),
        "separate": {
            "p10_ms": percentile(separate_trials, 10),
            "median_ms": separate_median,
            "p90_ms": percentile(separate_trials, 90),
            "trials_ms": separate_trials,
        },
        "fused": {
            "p10_ms": percentile(fused_trials, 10),
            "median_ms": fused_median,
            "p90_ms": percentile(fused_trials, 90),
            "trials_ms": fused_trials,
        },
        "speedup": separate_median / fused_median,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 64])
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    results = [
        benchmark_shape(
            device, tokens, args.hidden_size, args.num_experts, generator,
            args.warmup, args.iterations, args.repeats)
        for tokens in args.tokens
    ]
    report = {
        "config": vars(args) | {"out": str(args.out)},
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
