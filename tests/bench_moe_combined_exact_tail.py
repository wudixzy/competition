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
from vllm.model_executor.layers.activation import SiluAndMul


Case = Callable[[], torch.Tensor]


def load_extension(path: Path):
    spec = importlib.util.spec_from_file_location("corex_moe_exact_reduce", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension: {path}")
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
    fused_silu = SiluAndMul()

    def native_activation(value: torch.Tensor) -> torch.Tensor:
        gate, up = value.chunk(2, dim=-1)
        return F.silu(gate) * up

    def existing_reduce(output: torch.Tensor,
                        weights: torch.Tensor) -> torch.Tensor:
        return (output * weights.unsqueeze(-1)).sum(0, keepdim=True)

    def full_forward(current_hidden: torch.Tensor,
                     current_logits: torch.Tensor,
                     activation: Callable[[torch.Tensor], torch.Tensor],
                     reducer: Callable[[torch.Tensor, torch.Tensor],
                                       torch.Tensor]) -> torch.Tensor:
        logits, ids = torch.topk(current_logits.float(), top_k, dim=-1)
        weights = torch.softmax(logits, dim=-1)[0].to(dtype)
        selected_w13 = w13[ids[0]]
        selected_w2 = w2[ids[0]]
        gate_up = F.linear(
            current_hidden, selected_w13.reshape(-1, hidden)).view(top_k, -1)
        activated = activation(gate_up)
        expert_output = torch.bmm(
            selected_w2, activated.unsqueeze(-1)).squeeze(-1)
        return reducer(expert_output, weights).to(dtype)

    cases = {
        "native": lambda: full_forward(
            hidden_states, router_logits, native_activation, existing_reduce),
        "fused_activation": lambda: full_forward(
            hidden_states, router_logits, fused_silu, existing_reduce),
        "exact_reduce": lambda: full_forward(
            hidden_states, router_logits, native_activation,
            extension.serial_float),
        "combined": lambda: full_forward(
            hidden_states, router_logits, fused_silu,
            extension.serial_float),
    }

    reference = cases["native"]()
    checks = {}
    timings = {}
    for name, case in cases.items():
        actual = case()
        checks[name] = {
            "exact": bool(torch.equal(actual, reference)),
            "max_abs": float((actual.float() - reference.float()).abs().max()),
        }
        timings[name] = measure(
            case, args.warmup, args.iterations, args.repeats)

    baseline_ms = timings["native"]["median_ms"]
    for timing in timings.values():
        timing["speedup_vs_native"] = baseline_ms / timing["median_ms"]

    exact_steps = 0
    max_abs = 0.0
    for _ in range(args.sequence_steps):
        step_hidden = torch.randn(
            (1, hidden), device=device, dtype=dtype, generator=generator)
        step_logits = torch.randn(
            (1, experts), device=device, dtype=dtype, generator=generator)
        expected = full_forward(
            step_hidden, step_logits, native_activation, existing_reduce)
        actual = full_forward(
            step_hidden, step_logits, fused_silu, extension.serial_float)
        exact_steps += int(torch.equal(actual, expected))
        max_abs = max(
            max_abs, float((actual.float() - expected.float()).abs().max()))

    report = {
        "device": torch.cuda.get_device_name(device),
        "config": vars(args) | {
            "extension": str(args.extension), "out": str(args.out)},
        "shape": {
            "experts": experts,
            "top_k": top_k,
            "hidden": hidden,
            "local_intermediate": intermediate,
            "dtype": str(dtype),
        },
        "checks": checks,
        "sequence": {
            "exact_steps": exact_steps,
            "steps": args.sequence_steps,
            "max_abs": max_abs,
        },
        "timings": timings,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
