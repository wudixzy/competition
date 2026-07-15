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
from vllm.model_executor.layers.linear import ReplicatedLinear


Case = Callable[[], torch.Tensor]


def load_extension(path: Path):
    spec = importlib.util.spec_from_file_location("corex_gdn_gated_norm", path)
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


def diff(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    delta = (actual.float() - expected.float()).abs()
    return {
        "exact": bool(torch.equal(actual, expected)),
        "close": bool(torch.allclose(actual, expected,
                                     rtol=1e-4, atol=1e-5)),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
    }


def reference(core: torch.Tensor, gate: torch.Tensor,
              weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    hs = core.float()
    variance = hs.pow(2).mean(-1, keepdim=True)
    hs = hs * torch.rsqrt(variance + epsilon)
    hs = weight * hs
    return (hs * F.silu(gate.float())).to(torch.float16)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--sequence-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    extension = load_extension(args.extension)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    rows, head_dim, local_value_dim, hidden = 8, 128, 1024, 2048
    epsilon = 1e-6
    core = torch.randn(
        (rows, head_dim), device=device, generator=generator)
    gate = torch.randn(
        (rows, head_dim), device=device, generator=generator,
        dtype=torch.float16)
    weight = torch.randn(
        (head_dim,), device=device, generator=generator,
        dtype=torch.float16)
    output_weight = torch.randn(
        (hidden, local_value_dim), device=device, generator=generator,
        dtype=torch.float16) * 0.02
    out_proj = ReplicatedLinear(
        local_value_dim, hidden, bias=False, params_dtype=torch.float16,
        quant_config=None, prefix="probe.gdn_out_proj").to(device)
    out_proj.weight_loader(out_proj.weight, output_weight)

    variants: dict[str, Case] = {
        "reference": lambda: reference(core, gate, weight, epsilon),
        "tree": lambda: extension.tree(core, gate, weight, epsilon),
        "serial": lambda: extension.serial(core, gate, weight, epsilon),
        "pytorch_inverse": lambda: extension.apply_inverse(
            core, gate, weight,
            torch.rsqrt(core.pow(2).mean(-1, keepdim=True) + epsilon)),
    }
    expected = variants["reference"]()
    expected_tail, _ = out_proj(expected.reshape(1, -1))
    checks = {}
    for name, case in variants.items():
        actual = case()
        actual_tail, _ = out_proj(actual.reshape(1, -1))
        checks[name] = {
            "norm": diff(actual, expected),
            "tail": diff(actual_tail, expected_tail),
        }

    sequence_core = torch.randn(
        (args.sequence_steps, rows, head_dim), device=device,
        generator=generator)
    sequence_gate = torch.randn(
        (args.sequence_steps, rows, head_dim), device=device,
        generator=generator, dtype=torch.float16)
    sequence = {}
    for name, function in {
            "tree": extension.tree,
            "serial": extension.serial,
            "pytorch_inverse": None}.items():
        exact_steps = 0
        max_abs = 0.0
        max_tail_abs = 0.0
        for step in range(args.sequence_steps):
            expected_step = reference(
                sequence_core[step], sequence_gate[step], weight, epsilon)
            if function is None:
                inverse = torch.rsqrt(
                    sequence_core[step].pow(2).mean(-1, keepdim=True)
                    + epsilon)
                actual_step = extension.apply_inverse(
                    sequence_core[step], sequence_gate[step], weight, inverse)
            else:
                actual_step = function(
                    sequence_core[step], sequence_gate[step], weight, epsilon)
            if torch.equal(actual_step, expected_step):
                exact_steps += 1
            max_abs = max(max_abs, float(
                (actual_step.float() - expected_step.float()).abs().max()))
            expected_step_tail, _ = out_proj(expected_step.reshape(1, -1))
            actual_step_tail, _ = out_proj(actual_step.reshape(1, -1))
            max_tail_abs = max(max_tail_abs, float(
                (actual_step_tail.float()
                 - expected_step_tail.float()).abs().max()))
        sequence[name] = {
            "steps": args.sequence_steps,
            "exact_steps": exact_steps,
            "max_abs": max_abs,
            "max_tail_abs": max_tail_abs,
        }

    timings = {}
    for name, norm_case in variants.items():
        def tail_case(current: Case = norm_case) -> torch.Tensor:
            output, _ = out_proj(current().reshape(1, -1))
            return output

        timings[f"{name}_norm"] = measure(
            norm_case, args.warmup, args.iterations, args.repeats)
        timings[f"{name}_tail"] = measure(
            tail_case, args.warmup, args.iterations, args.repeats)
    reference_norm_ms = timings["reference_norm"]["median_ms"]
    reference_tail_ms = timings["reference_tail"]["median_ms"]
    for name in variants:
        timings[f"{name}_norm"]["speedup_vs_reference"] = (
            reference_norm_ms / timings[f"{name}_norm"]["median_ms"])
        timings[f"{name}_tail"]["speedup_vs_reference"] = (
            reference_tail_ms / timings[f"{name}_tail"]["median_ms"])

    report = {
        "device": torch.cuda.get_device_name(device),
        "config": {
            "rows": rows, "head_dim": head_dim,
            "local_value_dim": local_value_dim, "hidden": hidden,
            "epsilon": epsilon, "sequence_steps": args.sequence_steps,
            "warmup": args.warmup, "iterations": args.iterations,
            "repeats": args.repeats, "seed": args.seed},
        "checks": checks,
        "sequence": sequence,
        "timings": timings,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
