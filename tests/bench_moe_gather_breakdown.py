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


Case = Callable[[], object]


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
        started = time.perf_counter()
        for _ in range(iterations):
            case()
        torch.cuda.synchronize()
        trials.append((time.perf_counter() - started) * 1000.0 / iterations)
    return trials


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    experts, top_k, hidden, intermediate = 256, 8, 2048, 128
    dtype = torch.float16

    hidden_states = torch.randn(
        (1, hidden), device=device, dtype=dtype, generator=generator)
    router_logits = torch.randn(
        (1, experts), device=device, dtype=dtype, generator=generator)
    w13 = torch.full(
        (experts, 2 * intermediate, hidden), 0.01,
        device=device, dtype=dtype)
    w2 = torch.full(
        (experts, hidden, intermediate), 0.01,
        device=device, dtype=dtype)

    def route() -> tuple[torch.Tensor, torch.Tensor]:
        logits, ids = torch.topk(router_logits.float(), top_k, dim=-1)
        return torch.softmax(logits, dim=-1).to(dtype), ids

    topk_weights, topk_ids = route()
    selected_w13 = w13[topk_ids[0]]
    selected_w2 = w2[topk_ids[0]]

    def compute(current_w13: torch.Tensor,
                current_w2: torch.Tensor,
                weights: torch.Tensor) -> torch.Tensor:
        gate_up = F.linear(
            hidden_states, current_w13.reshape(-1, hidden)).view(top_k, -1)
        gate, up = gate_up.chunk(2, dim=-1)
        activation = F.silu(gate) * up
        expert_out = torch.bmm(
            current_w2, activation.unsqueeze(-1)).squeeze(-1)
        return (expert_out * weights[0].unsqueeze(-1)).sum(
            0, keepdim=True).to(dtype)

    def route_only() -> object:
        return route()

    def gather_only() -> object:
        return w13[topk_ids[0]], w2[topk_ids[0]]

    def compute_only() -> torch.Tensor:
        return compute(selected_w13, selected_w2, topk_weights)

    def gather_compute() -> torch.Tensor:
        return compute(w13[topk_ids[0]], w2[topk_ids[0]], topk_weights)

    def full_current() -> torch.Tensor:
        weights, ids = route()
        return compute(w13[ids[0]], w2[ids[0]], weights)

    reference = full_current()
    torch.cuda.synchronize()
    output_checks = {}
    for name, case in {
            "compute_only": compute_only,
            "gather_compute": gather_compute,
            "full_current": full_current}.items():
        output = case()
        torch.cuda.synchronize()
        difference = (output.float() - reference.float()).abs()
        output_checks[name] = {
            "exact": bool(torch.equal(output, reference)),
            "max_abs": float(difference.max()),
        }

    results = {}
    cases: dict[str, Case] = {
        "route_only": route_only,
        "gather_only": gather_only,
        "compute_only": compute_only,
        "gather_compute": gather_compute,
        "full_current": full_current,
    }
    for name, case in cases.items():
        trials = measure(case, args.warmup, args.iterations, args.repeats)
        results[name] = {
            "median_ms": statistics.median(trials),
            "p10_ms": percentile(trials, 10),
            "p90_ms": percentile(trials, 90),
            "trials_ms": trials,
        }

    full_ms = results["full_current"]["median_ms"]
    for result in results.values():
        result["share_of_full"] = result["median_ms"] / full_ms

    report = {
        "device": torch.cuda.get_device_name(device),
        "config": vars(args) | {"out": str(args.out)},
        "shapes": {
            "w13": list(w13.shape),
            "w2": list(w2.shape),
            "selected_w13": list(selected_w13.shape),
            "selected_w2": list(selected_w2.shape),
        },
        "selected_weight_bytes": (
            selected_w13.numel() + selected_w2.numel())
            * selected_w13.element_size(),
        "output_checks": output_checks,
        "results": results,
        "inferred": {
            "route_in_full_ms": full_ms
            - results["gather_compute"]["median_ms"],
            "gather_in_compute_ms": results["gather_compute"]["median_ms"]
            - results["compute_only"]["median_ms"],
        },
    }
    report["ok"] = all(
        check["exact"] for check in output_checks.values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
