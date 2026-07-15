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
    spec = importlib.util.spec_from_file_location("corex_moe_exact_reduce", path)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=300)
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
    experts, top_k, hidden, intermediate = 256, 8, 2048, 128
    dtype = torch.float16
    hidden_states = torch.randn(
        (1, hidden), device=device, dtype=dtype, generator=generator)
    router_logits = torch.randn(
        (1, experts), device=device, dtype=dtype, generator=generator)
    w13 = torch.randn(
        (experts, 2 * intermediate, hidden), device=device, dtype=dtype,
        generator=generator) * 0.02
    w2 = torch.randn(
        (experts, hidden, intermediate), device=device, dtype=dtype,
        generator=generator) * 0.02

    reducers = {
        "existing": lambda output, weights: (
            output * weights.unsqueeze(-1)).sum(0, keepdim=True),
        "serial_float": extension.serial_float,
        "tree_float": extension.tree_float,
        "serial_half": extension.serial_half,
    }
    sequence_outputs = torch.randn(
        (args.sequence_steps, top_k, hidden), device=device, dtype=dtype,
        generator=generator)
    sequence_weights = torch.softmax(torch.randn(
        (args.sequence_steps, top_k), device=device, generator=generator),
        dim=-1).to(dtype)
    sequence = {}
    for name, reducer in reducers.items():
        if name == "existing":
            continue
        exact_steps = 0
        max_abs = 0.0
        for step in range(args.sequence_steps):
            expected = reducers["existing"](
                sequence_outputs[step], sequence_weights[step])
            actual = reducer(sequence_outputs[step], sequence_weights[step])
            exact_steps += int(torch.equal(actual, expected))
            max_abs = max(max_abs, float(
                (actual.float() - expected.float()).abs().max()))
        sequence[name] = {
            "exact_steps": exact_steps,
            "steps": args.sequence_steps,
            "max_abs": max_abs,
        }

    topk_logits, topk_ids = torch.topk(
        router_logits.float(), top_k, dim=-1)
    fixed_weights = torch.softmax(topk_logits, dim=-1)[0].to(dtype)
    fixed_w13 = w13[topk_ids[0]]
    fixed_w2 = w2[topk_ids[0]]
    gate_up = F.linear(
        hidden_states, fixed_w13.reshape(-1, hidden)).view(top_k, -1)
    gate, up = gate_up.chunk(2, dim=-1)
    activation = F.silu(gate) * up
    fixed_expert_output = torch.bmm(
        fixed_w2, activation.unsqueeze(-1)).squeeze(-1)

    def full_forward(reducer: Callable) -> torch.Tensor:
        logits, ids = torch.topk(router_logits.float(), top_k, dim=-1)
        weights = torch.softmax(logits, dim=-1)[0].to(dtype)
        selected_w13 = w13[ids[0]]
        selected_w2 = w2[ids[0]]
        current_gate_up = F.linear(
            hidden_states, selected_w13.reshape(-1, hidden)).view(top_k, -1)
        current_gate, current_up = current_gate_up.chunk(2, dim=-1)
        current_activation = F.silu(current_gate) * current_up
        expert_output = torch.bmm(
            selected_w2, current_activation.unsqueeze(-1)).squeeze(-1)
        return reducer(expert_output, weights).to(dtype)

    timings = {}
    checks = {}
    expected_reduce = reducers["existing"](
        fixed_expert_output, fixed_weights)
    expected_full = full_forward(reducers["existing"])
    for name, reducer in reducers.items():
        reduce_case = lambda current=reducer: current(
            fixed_expert_output, fixed_weights)
        full_case = lambda current=reducer: full_forward(current)
        actual_reduce = reduce_case()
        actual_full = full_case()
        checks[name] = {
            "reduce_exact": bool(torch.equal(actual_reduce, expected_reduce)),
            "reduce_max_abs": float((
                actual_reduce.float() - expected_reduce.float()).abs().max()),
            "full_exact": bool(torch.equal(actual_full, expected_full)),
            "full_max_abs": float((
                actual_full.float() - expected_full.float()).abs().max()),
        }
        timings[f"{name}_reduce"] = measure(
            reduce_case, args.warmup, args.iterations, args.repeats)
        timings[f"{name}_full"] = measure(
            full_case, args.warmup, args.iterations, args.repeats)

    baseline_reduce = timings["existing_reduce"]["median_ms"]
    baseline_full = timings["existing_full"]["median_ms"]
    for name in reducers:
        timings[f"{name}_reduce"]["speedup_vs_existing"] = (
            baseline_reduce / timings[f"{name}_reduce"]["median_ms"])
        timings[f"{name}_full"]["speedup_vs_existing"] = (
            baseline_full / timings[f"{name}_full"]["median_ms"])

    report = {
        "device": torch.cuda.get_device_name(device),
        "config": vars(args) | {
            "extension": str(args.extension), "out": str(args.out)},
        "sequence": sequence,
        "checks": checks,
        "timings": timings,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
