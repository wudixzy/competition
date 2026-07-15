#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def load_extension(path: Path):
    spec = importlib.util.spec_from_file_location(
        "corex_gdn_projection_cublas", path)
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
    return {
        "median_ms": statistics.median(trials),
        "p10_ms": percentile(trials, 10),
        "p90_ms": percentile(trials, 90),
        "trials_ms": trials,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument(
        "--modes", default="-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,"
        "14,15,16,17,18,19,20,21,22,23,99,100,101,102,103,104,105,"
        "106,107,108,109,110,111,112,113,114,115")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--random-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    extension = load_extension(args.extension)
    modes = [int(value) for value in args.modes.split(",")]
    generator = torch.Generator(device=device).manual_seed(args.seed)
    dtype = torch.float16
    cases = {
        "input_projection": (2048, 3088),
        "output_projection": (1024, 2048),
    }
    report_cases = {}
    for name, (input_size, output_size) in cases.items():
        value = torch.randn(
            (1, input_size), device=device, dtype=dtype,
            generator=generator) * 0.05
        weight = torch.randn(
            (output_size, input_size), device=device, dtype=dtype,
            generator=generator) * 0.02
        expected = F.linear(value, weight)
        baseline = measure(
            lambda: F.linear(value, weight),
            args.warmup, args.iterations, args.repeats)
        results = {}
        for mode in modes:
            try:
                actual = extension.linear(value, weight, mode)
                torch.cuda.synchronize()
                timing = measure(
                    lambda current=mode: extension.linear(
                        value, weight, current),
                    args.warmup, args.iterations, args.repeats)
                timing["speedup_vs_baseline"] = (
                    baseline["median_ms"] / timing["median_ms"])
                results[str(mode)] = {
                    "exact": bool(torch.equal(actual, expected)),
                    "max_abs": float((actual - expected).abs().max()),
                    "timing": timing,
                }
            except RuntimeError as exc:
                results[str(mode)] = {"error": str(exc)}

        exact_modes = [
            mode for mode in modes
            if "error" not in results[str(mode)]
            and results[str(mode)]["exact"]]
        best_mode = min(
            exact_modes,
            key=lambda mode: results[str(mode)]["timing"]["median_ms"],
        ) if exact_modes else None
        random_exact_steps = 0
        random_max_abs = 0.0
        if best_mode is not None:
            for _ in range(args.random_steps):
                step_value = torch.randn(
                    (1, input_size), device=device, dtype=dtype,
                    generator=generator)
                step_expected = F.linear(step_value, weight)
                step_actual = extension.linear(
                    step_value, weight, best_mode)
                random_exact_steps += int(torch.equal(
                    step_actual, step_expected))
                random_max_abs = max(random_max_abs, float(
                    (step_actual - step_expected).abs().max()))
        report_cases[name] = {
            "shape": {
                "input": [1, input_size],
                "weight": [output_size, input_size],
            },
            "baseline": baseline,
            "best_exact_mode": best_mode,
            "random": {
                "steps": args.random_steps,
                "exact_steps": random_exact_steps,
                "max_abs": random_max_abs,
            },
            "results": results,
        }

    report = {
        "device": torch.cuda.get_device_name(device),
        "config": {
            "modes": modes, "warmup": args.warmup,
            "iterations": args.iterations, "repeats": args.repeats,
            "random_steps": args.random_steps, "seed": args.seed,
        },
        "cases": report_cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
