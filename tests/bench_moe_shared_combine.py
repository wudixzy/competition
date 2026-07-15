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
    spec = importlib.util.spec_from_file_location(
        "corex_moe_shared_combine", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(routed: torch.Tensor, shared: torch.Tensor,
              gate: torch.Tensor) -> torch.Tensor:
    return routed + shared * torch.sigmoid(gate)


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
    batch, hidden = 1, 2048
    routed = torch.randn(
        (batch, hidden), device=device, generator=generator,
        dtype=torch.float16)
    shared = torch.randn(
        (batch, hidden), device=device, generator=generator,
        dtype=torch.float16)
    gate = torch.randn(
        (batch, 1), device=device, generator=generator,
        dtype=torch.float16)

    expected = reference(routed, shared, gate)
    actual_direct = extension.shared_combine(routed, shared, gate)
    actual_pytorch_sigmoid = extension.shared_combine_gate(
        routed, shared, torch.sigmoid(gate))
    one_step = {
        "direct": {
            "exact": bool(torch.equal(actual_direct, expected)),
            "max_abs": float((actual_direct - expected).abs().max()),
        },
        "pytorch_sigmoid": {
            "exact": bool(torch.equal(actual_pytorch_sigmoid, expected)),
            "max_abs": float(
                (actual_pytorch_sigmoid - expected).abs().max()),
        },
        "finite": bool(torch.isfinite(actual_pytorch_sigmoid).all()),
    }

    exact_steps = {"direct": 0, "pytorch_sigmoid": 0}
    max_abs = {"direct": 0.0, "pytorch_sigmoid": 0.0}
    for _ in range(args.random_steps):
        step_routed = torch.randn(
            routed.shape, device=device, generator=generator,
            dtype=torch.float16)
        step_shared = torch.randn(
            shared.shape, device=device, generator=generator,
            dtype=torch.float16)
        step_gate = torch.randn(
            gate.shape, device=device, generator=generator,
            dtype=torch.float16)
        step_expected = reference(step_routed, step_shared, step_gate)
        step_direct = extension.shared_combine(
            step_routed, step_shared, step_gate)
        step_pytorch_sigmoid = extension.shared_combine_gate(
            step_routed, step_shared, torch.sigmoid(step_gate))
        for name, step_actual in {
                "direct": step_direct,
                "pytorch_sigmoid": step_pytorch_sigmoid}.items():
            exact_steps[name] += int(torch.equal(step_actual, step_expected))
            max_abs[name] = max(max_abs[name], float(
                (step_actual - step_expected).abs().max()))

    # Exercise every finite FP16 gate bit pattern. This catches sigmoid
    # differences outside the normal-distribution range used above.
    all_half = torch.arange(65536, dtype=torch.int32).to(
        torch.int16).view(torch.float16)
    finite_gate = all_half[torch.isfinite(all_half)].reshape(-1, 1).to(device)
    exhaustive_routed = torch.randn(
        finite_gate.shape, device=device, generator=generator,
        dtype=torch.float16)
    exhaustive_shared = torch.randn(
        finite_gate.shape, device=device, generator=generator,
        dtype=torch.float16)
    exhaustive_expected = reference(
        exhaustive_routed, exhaustive_shared, finite_gate)
    exhaustive_actual = extension.shared_combine(
        exhaustive_routed, exhaustive_shared, finite_gate)
    exhaustive_gate = {
        "finite_patterns": int(finite_gate.numel()),
        "exact": bool(torch.equal(exhaustive_actual, exhaustive_expected)),
        "max_abs": float(
            (exhaustive_actual - exhaustive_expected).abs().max()),
    }

    timings = {
        "reference": measure(
            lambda: reference(routed, shared, gate),
            args.warmup, args.iterations, args.repeats),
        "candidate_direct": measure(
            lambda: extension.shared_combine(routed, shared, gate),
            args.warmup, args.iterations, args.repeats),
        "candidate_pytorch_sigmoid": measure(
            lambda: extension.shared_combine_gate(
                routed, shared, torch.sigmoid(gate)),
            args.warmup, args.iterations, args.repeats),
    }
    for name in ("candidate_direct", "candidate_pytorch_sigmoid"):
        timings[name]["speedup_vs_reference"] = (
            timings["reference"]["median_ms"]
            / timings[name]["median_ms"])

    report = {
        "device": torch.cuda.get_device_name(device),
        "shape": {"batch": batch, "hidden": hidden},
        "config": {
            "warmup": args.warmup, "iterations": args.iterations,
            "repeats": args.repeats, "random_steps": args.random_steps,
            "seed": args.seed,
        },
        "one_step": one_step,
        "random": {
            "steps": args.random_steps, "exact_steps": exact_steps,
            "max_abs": max_abs,
        },
        "exhaustive_gate": exhaustive_gate,
        "timings": timings,
    }
    report["ok"] = bool(
        one_step["pytorch_sigmoid"]["exact"]
        and exact_steps["pytorch_sigmoid"] == args.random_steps
        and exhaustive_gate["exact"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
