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
from vllm.model_executor.layers.activation import SiluAndMul


Case = Callable[[], torch.Tensor]


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def measure(case: Case, warmup: int, iterations: int,
            repeats: int) -> list[float]:
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


def summarize(cases: dict[str, Case], warmup: int, iterations: int,
              repeats: int) -> dict:
    reference = cases["native"]()
    torch.cuda.synchronize()
    results = {}
    for name, case in cases.items():
        output = case()
        torch.cuda.synchronize()
        difference = (output.float() - reference.float()).abs()
        trials = measure(case, warmup, iterations, repeats)
        results[name] = {
            "exact": bool(torch.equal(output, reference)),
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
            "p10_ms": percentile(trials, 10),
            "median_ms": statistics.median(trials),
            "p90_ms": percentile(trials, 90),
            "trials_ms": trials,
        }

    baseline_ms = results["native"]["median_ms"]
    for result in results.values():
        result["speedup_vs_native"] = baseline_ms / result["median_ms"]
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    dtype = torch.float16
    generator = torch.Generator(device=device).manual_seed(args.seed)
    hidden_states = torch.randn(
        (1, args.hidden_size), device=device, dtype=dtype,
        generator=generator)
    router_logits = torch.randn(
        (1, args.num_experts), device=device, dtype=dtype,
        generator=generator)
    w13 = torch.empty(
        (args.num_experts, 2 * args.intermediate_size, args.hidden_size),
        device=device, dtype=dtype)
    w2 = torch.empty(
        (args.num_experts, args.hidden_size, args.intermediate_size),
        device=device, dtype=dtype)
    w13.normal_(mean=0.0, std=0.02, generator=generator)
    w2.normal_(mean=0.0, std=0.02, generator=generator)

    topk_logits, topk_ids = torch.topk(
        router_logits.float(), args.top_k, dim=-1)
    weights = torch.softmax(topk_logits, dim=-1)[0].to(dtype)
    w13_sel = w13[topk_ids[0]]
    w2_sel = w2[topk_ids[0]]
    gate_up = F.linear(
        hidden_states, w13_sel.reshape(-1, args.hidden_size)).view(
            args.top_k, -1)
    fused_silu = SiluAndMul()

    def native_activation(value: torch.Tensor) -> torch.Tensor:
        gate, up = value.chunk(2, dim=-1)
        return F.silu(gate) * up

    def fused_activation(value: torch.Tensor) -> torch.Tensor:
        return fused_silu(value)

    activation_cases = {
        "native": lambda: native_activation(gate_up),
        "fused": lambda: fused_activation(gate_up),
    }

    def full_forward(activation: Callable[[torch.Tensor], torch.Tensor]) \
            -> torch.Tensor:
        current_logits, current_ids = torch.topk(
            router_logits.float(), args.top_k, dim=-1)
        current_weights = torch.softmax(current_logits, dim=-1)[0].to(dtype)
        current_w13 = w13[current_ids[0]]
        current_w2 = w2[current_ids[0]]
        current_gate_up = F.linear(
            hidden_states, current_w13.reshape(-1, args.hidden_size)).view(
                args.top_k, -1)
        act = activation(current_gate_up)
        expert_out = torch.bmm(
            current_w2, act.unsqueeze(-1)).squeeze(-1)
        return (expert_out * current_weights.unsqueeze(-1)).sum(
            0, keepdim=True).to(dtype)

    full_cases = {
        "native": lambda: full_forward(native_activation),
        "fused": lambda: full_forward(fused_activation),
    }
    report = {
        "config": vars(args) | {"out": str(args.out)},
        "metadata": {
            "gate_up_shape": list(gate_up.shape),
            "w13_shape": list(w13.shape),
            "w2_shape": list(w2.shape),
            "allocated_bytes": torch.cuda.memory_allocated(device),
        },
        "activation_only": summarize(
            activation_cases, args.warmup, args.iterations, args.repeats),
        "full_path": summarize(
            full_cases, args.warmup, args.iterations, args.repeats),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
