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
    spec = importlib.util.spec_from_file_location(
        "corex_attn_head_rms_norm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(weight: torch.Tensor, epsilon: float,
              value: torch.Tensor) -> torch.Tensor:
    original_dtype = value.dtype
    converted = value.float()
    variance = converted.pow(2).mean(dim=-1, keepdim=True)
    normalized = converted * torch.rsqrt(variance + epsilon)
    return (normalized * (1.0 + weight.float())).to(original_dtype)


def candidate(extension, weight: torch.Tensor, epsilon: float,
              value: torch.Tensor) -> torch.Tensor:
    converted, squares = extension.prepare(value)
    inverse = torch.rsqrt(squares.mean(dim=-1, keepdim=True) + epsilon)
    return extension.apply_inverse(converted, weight, inverse)


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


def exact(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    delta = (actual.float() - expected.float()).abs()
    return {
        "exact": bool(torch.equal(actual, expected)),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--random-steps", type=int, default=1000)
    parser.add_argument("--min-speedup", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    extension = load_extension(args.extension)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    head_dim, epsilon = 256, 1e-6
    q = torch.randn((4, head_dim), device=device, generator=generator,
                    dtype=torch.float16) * 0.2
    k = torch.randn((1, head_dim), device=device, generator=generator,
                    dtype=torch.float16) * 0.2
    q_weight = torch.randn((head_dim,), device=device, generator=generator,
                           dtype=torch.float16) * 0.05
    k_weight = torch.randn((head_dim,), device=device, generator=generator,
                           dtype=torch.float16) * 0.05

    cases = {
        "q": (
            lambda: reference(q_weight, epsilon, q),
            lambda: candidate(extension, q_weight, epsilon, q),
        ),
        "k": (
            lambda: reference(k_weight, epsilon, k),
            lambda: candidate(extension, k_weight, epsilon, k),
        ),
        "qk_pair": (
            lambda: (
                reference(q_weight, epsilon, q),
                reference(k_weight, epsilon, k),
            ),
            lambda: (
                candidate(extension, q_weight, epsilon, q),
                candidate(extension, k_weight, epsilon, k),
            ),
        ),
    }

    one_step = {
        "q": exact(cases["q"][1](), cases["q"][0]()),
        "k": exact(cases["k"][1](), cases["k"][0]()),
    }
    random_q = torch.randn(
        (args.random_steps, 4, head_dim), device=device,
        generator=generator, dtype=torch.float16) * 0.5
    random_k = torch.randn(
        (args.random_steps, 1, head_dim), device=device,
        generator=generator, dtype=torch.float16) * 0.5
    random_check = {}
    for name, values, weight in (
            ("q", random_q, q_weight), ("k", random_k, k_weight)):
        exact_steps = 0
        max_abs = 0.0
        for step in range(args.random_steps):
            expected = reference(weight, epsilon, values[step])
            actual = candidate(extension, weight, epsilon, values[step])
            result = exact(actual, expected)
            exact_steps += int(result["exact"])
            max_abs = max(max_abs, result["max_abs"])
        random_check[name] = {
            "steps": args.random_steps,
            "exact_steps": exact_steps,
            "max_abs": max_abs,
        }

    timings = {}
    for name, (reference_case, candidate_case) in cases.items():
        reference_timing = measure(
            reference_case, args.warmup, args.iterations, args.repeats)
        candidate_timing = measure(
            candidate_case, args.warmup, args.iterations, args.repeats)
        candidate_timing["speedup_vs_reference"] = (
            reference_timing["median_ms"] / candidate_timing["median_ms"])
        timings[f"reference_{name}"] = reference_timing
        timings[f"candidate_{name}"] = candidate_timing

    exact_ok = bool(
        all(result["exact"] for result in one_step.values())
        and all(result["exact_steps"] == args.random_steps
                for result in random_check.values()))
    pair_speedup = timings["candidate_qk_pair"]["speedup_vs_reference"]
    report = {
        "ok": exact_ok,
        "performance_gate_passed": pair_speedup >= args.min_speedup,
        "device": torch.cuda.get_device_name(device),
        "shape": {"q_rows": 4, "k_rows": 1, "head_dim": head_dim},
        "config": {
            "epsilon": epsilon,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "random_steps": args.random_steps,
            "min_speedup": args.min_speedup,
            "seed": args.seed,
        },
        "one_step": one_step,
        "random": random_check,
        "timings": timings,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if exact_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
