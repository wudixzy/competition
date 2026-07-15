#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch

from vllm.model_executor.layers.linear import ReplicatedLinear


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

    routed_w2 = torch.randn(
        (experts, hidden, intermediate), device=device, dtype=dtype,
        generator=generator) * 0.02
    shared_w2 = torch.randn(
        (hidden, intermediate), device=device, dtype=dtype,
        generator=generator) * 0.02
    packed_w2 = torch.cat((routed_w2, shared_w2.unsqueeze(0)), dim=0)
    expert_ids = torch.randperm(
        experts, device=device, generator=generator)[:top_k]
    shared_id = torch.tensor([experts], device=device, dtype=torch.int64)
    routed_activation = torch.randn(
        (top_k, intermediate), device=device, dtype=dtype,
        generator=generator)
    shared_activation = torch.randn(
        (1, intermediate), device=device, dtype=dtype,
        generator=generator)
    routing_weights = torch.softmax(torch.randn(
        (top_k,), device=device, generator=generator), dim=0).to(dtype)
    shared_gate = torch.sigmoid(torch.randn(
        (1, 1), device=device, generator=generator)).to(dtype)
    shared_down_layer = ReplicatedLinear(
        intermediate, hidden, bias=False, params_dtype=dtype,
        quant_config=None, prefix="probe.shared_down").to(device)
    shared_down_layer.weight_loader(shared_down_layer.weight, shared_w2)

    def baseline_down() -> tuple[torch.Tensor, torch.Tensor]:
        selected = routed_w2[expert_ids]
        routed = torch.bmm(
            selected, routed_activation.unsqueeze(-1)).squeeze(-1)
        shared, _ = shared_down_layer(shared_activation)
        return routed, shared

    def packed_down() -> tuple[torch.Tensor, torch.Tensor]:
        selected_ids = torch.cat((expert_ids, shared_id))
        selected = packed_w2[selected_ids]
        activation = torch.cat((routed_activation, shared_activation), dim=0)
        projected = torch.bmm(
            selected, activation.unsqueeze(-1)).squeeze(-1)
        return projected[:top_k], projected[top_k:]

    def finish(routed: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
        routed_output = (routed * routing_weights.unsqueeze(-1)).sum(
            0, keepdim=True).to(dtype)
        return routed_output + shared * shared_gate

    def baseline_full() -> torch.Tensor:
        return finish(*baseline_down())

    def packed_full() -> torch.Tensor:
        return finish(*packed_down())

    expected_parts = baseline_down()
    actual_parts = packed_down()
    expected_output = baseline_full()
    actual_output = packed_full()
    torch.cuda.synchronize()
    checks = {}
    for name, actual, expected in (
            ("routed_down", actual_parts[0], expected_parts[0]),
            ("shared_down", actual_parts[1], expected_parts[1]),
            ("full_output", actual_output, expected_output)):
        difference = (actual.float() - expected.float()).abs()
        checks[name] = {
            "exact": bool(torch.equal(actual, expected)),
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
        }

    cases: dict[str, Case] = {
        "baseline_down": baseline_down,
        "packed_down": packed_down,
        "baseline_full": baseline_full,
        "packed_full": packed_full,
    }
    timings = {}
    for name, case in cases.items():
        trials = measure(case, args.warmup, args.iterations, args.repeats)
        timings[name] = {
            "median_ms": statistics.median(trials),
            "p10_ms": percentile(trials, 10),
            "p90_ms": percentile(trials, 90),
            "trials_ms": trials,
        }
    timings["packed_down"]["speedup_vs_baseline"] = (
        timings["baseline_down"]["median_ms"]
        / timings["packed_down"]["median_ms"])
    timings["packed_full"]["speedup_vs_baseline"] = (
        timings["baseline_full"]["median_ms"]
        / timings["packed_full"]["median_ms"])

    report = {
        "device": torch.cuda.get_device_name(device),
        "config": vars(args) | {"out": str(args.out)},
        "shapes": {
            "routed_w2": list(routed_w2.shape),
            "shared_w2": list(shared_w2.shape),
            "packed_w2": list(packed_w2.shape),
        },
        "checks": checks,
        "timings": timings,
    }
    report["ok"] = all(check["exact"] for check in checks.values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
