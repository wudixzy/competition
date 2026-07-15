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
        "corex_gemma_rms_norm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(weight: torch.Tensor, epsilon: float, x: torch.Tensor,
              residual: torch.Tensor | None = None):
    original_dtype = x.dtype
    if residual is not None:
        x = x + residual
        residual = x
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    x = x * (1.0 + weight.float())
    x = x.to(original_dtype)
    return x if residual is None else (x, residual)


def candidate(extension, weight: torch.Tensor, epsilon: float,
              x: torch.Tensor, residual: torch.Tensor | None = None):
    if residual is None:
        converted, squares = extension.prepare(x)
        saved_residual = None
    else:
        converted, squares, saved_residual = extension.prepare_residual(
            x, residual)
    inverse = torch.rsqrt(squares.mean(dim=-1, keepdim=True) + epsilon)
    output = extension.apply_inverse(converted, weight, inverse)
    return output if saved_residual is None else (output, saved_residual)


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


def exact(actual, expected) -> dict[str, object]:
    actual_values = actual if isinstance(actual, tuple) else (actual,)
    expected_values = expected if isinstance(expected, tuple) else (expected,)
    checks = []
    for actual_value, expected_value in zip(actual_values, expected_values):
        delta = (actual_value.float() - expected_value.float()).abs()
        checks.append({
            "exact": bool(torch.equal(actual_value, expected_value)),
            "max_abs": float(delta.max()),
            "mean_abs": float(delta.mean()),
        })
    return {
        "exact": all(check["exact"] for check in checks),
        "values": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
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
    extension = load_extension(args.extension)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    rows, hidden, epsilon = 1, 2048, 1e-6
    x = torch.randn((rows, hidden), device=device, generator=generator,
                    dtype=torch.float16) * 0.2
    residual = torch.randn((rows, hidden), device=device,
                           generator=generator,
                           dtype=torch.float16) * 0.2
    weight = torch.randn((hidden,), device=device, generator=generator,
                         dtype=torch.float16) * 0.05

    reference_cases = {
        "no_residual": lambda: reference(weight, epsilon, x),
        "residual": lambda: reference(weight, epsilon, x, residual),
    }
    candidate_cases = {
        "no_residual": lambda: candidate(extension, weight, epsilon, x),
        "residual": lambda: candidate(
            extension, weight, epsilon, x, residual),
    }
    one_step = {
        name: exact(candidate_cases[name](), reference_cases[name]())
        for name in reference_cases
    }

    random_x = torch.randn(
        (args.random_steps, rows, hidden), device=device,
        generator=generator, dtype=torch.float16) * 0.5
    random_residual = torch.randn(
        (args.random_steps, rows, hidden), device=device,
        generator=generator, dtype=torch.float16) * 0.5
    random_check = {}
    for mode in ("no_residual", "residual"):
        exact_steps = 0
        max_abs = 0.0
        max_residual_abs = 0.0
        for step in range(args.random_steps):
            step_residual = (None if mode == "no_residual"
                             else random_residual[step])
            expected = reference(
                weight, epsilon, random_x[step], step_residual)
            actual = candidate(
                extension, weight, epsilon, random_x[step], step_residual)
            result = exact(actual, expected)
            exact_steps += int(result["exact"])
            max_abs = max(max_abs, result["values"][0]["max_abs"])
            if len(result["values"]) > 1:
                max_residual_abs = max(
                    max_residual_abs, result["values"][1]["max_abs"])
        random_check[mode] = {
            "steps": args.random_steps,
            "exact_steps": exact_steps,
            "max_abs": max_abs,
            "max_residual_abs": max_residual_abs,
        }

    timings = {}
    for name in reference_cases:
        timings[f"reference_{name}"] = measure(
            reference_cases[name], args.warmup, args.iterations, args.repeats)
        timings[f"candidate_{name}"] = measure(
            candidate_cases[name], args.warmup, args.iterations, args.repeats)
        timings[f"candidate_{name}"]["speedup_vs_reference"] = (
            timings[f"reference_{name}"]["median_ms"]
            / timings[f"candidate_{name}"]["median_ms"])

    def reference_pair():
        first, saved = reference(weight, epsilon, x, residual)
        return reference(weight, epsilon, first, saved)

    def candidate_pair():
        first, saved = candidate(extension, weight, epsilon, x, residual)
        return candidate(extension, weight, epsilon, first, saved)

    timings["reference_pair"] = measure(
        reference_pair, args.warmup, args.iterations, args.repeats)
    timings["candidate_pair"] = measure(
        candidate_pair, args.warmup, args.iterations, args.repeats)
    timings["candidate_pair"]["speedup_vs_reference"] = (
        timings["reference_pair"]["median_ms"]
        / timings["candidate_pair"]["median_ms"])

    ok = bool(
        all(result["exact"] for result in one_step.values())
        and all(result["exact_steps"] == args.random_steps
                for result in random_check.values()))
    report = {
        "ok": ok,
        "device": torch.cuda.get_device_name(device),
        "shape": {"rows": rows, "hidden": hidden},
        "config": {
            "epsilon": epsilon,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "random_steps": args.random_steps,
            "seed": args.seed,
        },
        "one_step": one_step,
        "random": random_check,
        "timings": timings,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
