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
import torch.nn.functional as F


Case = Callable[[], torch.Tensor]


def load_extension(path: Path):
    spec = importlib.util.spec_from_file_location(
        "corex_gdn_beta_decay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(beta_input: torch.Tensor, decay_input: torch.Tensor,
              a_log: torch.Tensor,
              dt_bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    beta = beta_input.sigmoid().float()
    decay = (-a_log.float().exp() * F.softplus(
        decay_input.float() + dt_bias)).exp()
    return beta, decay


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


def differences(actual: torch.Tensor,
                expected: torch.Tensor) -> dict[str, object]:
    delta = (actual - expected).abs()
    return {
        "exact": bool(torch.equal(actual, expected)),
        "close": bool(torch.allclose(actual, expected, rtol=1e-6, atol=1e-7)),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "finite": bool(torch.isfinite(actual).all()),
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
    batch, heads = 1, 8
    # Use views from the real merged TP4 projection layout. With fixed
    # max_num_seqs=1 these narrow slices are contiguous and take the fast path.
    projected = torch.randn(
        (batch, 3088), device=device, generator=generator,
        dtype=torch.float16)
    _, _, beta_input, decay_input = torch.split(
        projected, [2048, 1024, heads, heads], dim=-1)
    a_log = torch.randn(
        (heads,), device=device, generator=generator,
        dtype=torch.float16) * 0.1
    dt_bias = torch.randn(
        (heads,), device=device, generator=generator,
        dtype=torch.float16) * 0.1

    expected_beta, expected_decay = reference(
        beta_input, decay_input, a_log, dt_bias)
    actual = extension.beta_decay(
        beta_input, decay_input, a_log, dt_bias)
    one_step = {
        "beta": differences(actual[0], expected_beta),
        "decay": differences(actual[1], expected_decay),
    }

    random_beta = torch.randn(
        (args.random_steps, batch, heads), device=device,
        generator=generator, dtype=torch.float16)
    random_decay = torch.randn(
        (args.random_steps, batch, heads), device=device,
        generator=generator, dtype=torch.float16)
    exact_beta_steps = 0
    exact_decay_steps = 0
    max_beta_abs = 0.0
    max_decay_abs = 0.0
    for step in range(args.random_steps):
        step_beta, step_decay = reference(
            random_beta[step], random_decay[step], a_log, dt_bias)
        step_actual = extension.beta_decay(
            random_beta[step], random_decay[step], a_log, dt_bias)
        exact_beta_steps += int(torch.equal(step_actual[0], step_beta))
        exact_decay_steps += int(torch.equal(step_actual[1], step_decay))
        max_beta_abs = max(max_beta_abs, float(
            (step_actual[0] - step_beta).abs().max()))
        max_decay_abs = max(max_decay_abs, float(
            (step_actual[1] - step_decay).abs().max()))
    random_check = {
        "steps": args.random_steps,
        "exact_beta_steps": exact_beta_steps,
        "exact_decay_steps": exact_decay_steps,
        "max_beta_abs": max_beta_abs,
        "max_decay_abs": max_decay_abs,
    }

    def reference_case() -> torch.Tensor:
        _, decay = reference(beta_input, decay_input, a_log, dt_bias)
        return decay

    def candidate_case() -> torch.Tensor:
        return extension.beta_decay(
            beta_input, decay_input, a_log, dt_bias)

    timings = {
        "reference": measure(
            reference_case, args.warmup, args.iterations, args.repeats),
        "candidate": measure(
            candidate_case, args.warmup, args.iterations, args.repeats),
    }
    baseline = timings["reference"]["median_ms"]
    for result in timings.values():
        result["speedup_vs_reference"] = baseline / result["median_ms"]

    report = {
        "device": torch.cuda.get_device_name(device),
        "shape": {"batch": batch, "heads": heads},
        "layout": {
            "beta_contiguous": beta_input.is_contiguous(),
            "decay_contiguous": decay_input.is_contiguous(),
        },
        "config": {
            "warmup": args.warmup, "iterations": args.iterations,
            "repeats": args.repeats, "random_steps": args.random_steps,
            "seed": args.seed,
        },
        "one_step": one_step,
        "random": random_check,
        "timings": timings,
    }
    report["ok"] = bool(
        one_step["beta"]["exact"] and one_step["decay"]["exact"]
        and exact_beta_steps == args.random_steps
        and exact_decay_steps == args.random_steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
