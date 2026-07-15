#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch


Case = Callable[[], torch.Tensor]


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


def baseline(logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    values = logits.float()
    values.div_(temperature)
    torch.softmax(values, dim=-1, dtype=torch.float)
    logprobs = torch.log_softmax(values, dim=-1, dtype=torch.float)
    return torch.argmax(logprobs, dim=-1)


def candidate(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--random-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    logits = torch.randn(
        (1, args.vocab_size), device=device, dtype=torch.float16,
        generator=generator)
    temperature = torch.ones((1, 1), device=device, dtype=torch.float32)

    exact_steps = 0
    for _ in range(args.random_steps):
        random_logits = torch.randn(
            (1, args.vocab_size), device=device, dtype=torch.float16,
            generator=generator)
        exact_steps += int(torch.equal(
            baseline(random_logits, temperature), candidate(random_logits)))

    tie_cases = []
    for first, second in ((0, 1), (17, 18),
                          (args.vocab_size - 2, args.vocab_size - 1)):
        tied = torch.full_like(logits, -4.0)
        tied[0, first] = 7.0
        tied[0, second] = 7.0
        expected = baseline(tied, temperature)
        actual = candidate(tied)
        tie_cases.append({
            "indices": [first, second],
            "baseline": int(expected.item()),
            "candidate": int(actual.item()),
            "exact": bool(torch.equal(expected, actual)),
        })

    baseline_timing = measure(
        lambda: baseline(logits, temperature),
        args.warmup, args.iterations, args.repeats)
    candidate_timing = measure(
        lambda: candidate(logits),
        args.warmup, args.iterations, args.repeats)
    speedup = baseline_timing["median_ms"] / candidate_timing["median_ms"]
    saving_ms = baseline_timing["median_ms"] - candidate_timing["median_ms"]
    exact_ok = (exact_steps == args.random_steps
                and all(case["exact"] for case in tie_cases))
    report = {
        "ok": exact_ok,
        "device": torch.cuda.get_device_name(device),
        "shape": [1, args.vocab_size],
        "config": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "random_steps": args.random_steps,
            "seed": args.seed,
        },
        "random_exact_steps": exact_steps,
        "ties": tie_cases,
        "timings": {
            "baseline": baseline_timing,
            "candidate": candidate_timing,
            "speedup": speedup,
            "saving_ms": saving_ms,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if exact_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
