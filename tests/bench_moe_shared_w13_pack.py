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
    parser.add_argument("--iterations", type=int, default=200)
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
    router_weight = torch.randn(
        (experts + 1, hidden), device=device, dtype=dtype,
        generator=generator) * 0.02
    routed_w13 = torch.randn(
        (experts, 2 * intermediate, hidden), device=device, dtype=dtype,
        generator=generator) * 0.02
    shared_w13 = torch.randn(
        (2 * intermediate, hidden), device=device, dtype=dtype,
        generator=generator) * 0.02
    packed_w13 = torch.cat((routed_w13, shared_w13.unsqueeze(0)), dim=0)
    routed_w2 = torch.randn(
        (experts, hidden, intermediate), device=device, dtype=dtype,
        generator=generator) * 0.02
    shared_w2 = torch.randn(
        (hidden, intermediate), device=device, dtype=dtype,
        generator=generator) * 0.02
    shared_id = torch.tensor([experts], device=device, dtype=torch.int64)
    shared_gate_up_layer = ReplicatedLinear(
        hidden, 2 * intermediate, bias=False, params_dtype=dtype,
        quant_config=None, prefix="probe.shared_gate_up").to(device)
    shared_gate_up_layer.weight_loader(
        shared_gate_up_layer.weight, shared_w13)

    def route() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router = F.linear(hidden_states, router_weight)
        logits, ids = torch.topk(router[:, :experts].float(), top_k, dim=-1)
        weights = torch.softmax(logits, dim=-1).to(dtype)
        return weights, ids, router[:, experts:]

    weights, ids, gate_score = route()

    def baseline_gate_up() -> tuple[torch.Tensor, torch.Tensor]:
        selected = routed_w13[ids[0]]
        routed = F.linear(
            hidden_states, selected.reshape(-1, hidden)).view(top_k, -1)
        shared, _ = shared_gate_up_layer(hidden_states)
        return routed, shared

    def packed_gate_up() -> tuple[torch.Tensor, torch.Tensor]:
        selected_ids = torch.cat((ids[0], shared_id))
        selected = packed_w13[selected_ids]
        projected = F.linear(
            hidden_states, selected.reshape(-1, hidden)).view(top_k + 1, -1)
        return projected[:top_k], projected[top_k:]

    def finish(routed_gate_up: torch.Tensor,
               shared_gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = routed_gate_up.chunk(2, dim=-1)
        activation = F.silu(gate) * up
        selected_w2 = routed_w2[ids[0]]
        expert_out = torch.bmm(
            selected_w2, activation.unsqueeze(-1)).squeeze(-1)
        routed = (expert_out * weights[0].unsqueeze(-1)).sum(
            0, keepdim=True).to(dtype)

        shared_gate, shared_up = shared_gate_up.chunk(2, dim=-1)
        shared_activation = F.silu(shared_gate) * shared_up
        shared = F.linear(shared_activation, shared_w2)
        shared = shared * torch.sigmoid(gate_score)
        return routed + shared

    def baseline_full() -> torch.Tensor:
        return finish(*baseline_gate_up())

    def packed_full() -> torch.Tensor:
        return finish(*packed_gate_up())

    baseline_parts = baseline_gate_up()
    packed_parts = packed_gate_up()
    baseline_output = baseline_full()
    packed_output = packed_full()
    torch.cuda.synchronize()
    checks = {}
    for name, actual, expected in (
            ("routed_gate_up", packed_parts[0], baseline_parts[0]),
            ("shared_gate_up", packed_parts[1], baseline_parts[1]),
            ("full_output", packed_output, baseline_output)):
        difference = (actual.float() - expected.float()).abs()
        checks[name] = {
            "exact": bool(torch.equal(actual, expected)),
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
        }

    cases: dict[str, Case] = {
        "baseline_gate_up": baseline_gate_up,
        "packed_gate_up": packed_gate_up,
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
    timings["packed_gate_up"]["speedup_vs_baseline"] = (
        timings["baseline_gate_up"]["median_ms"]
        / timings["packed_gate_up"]["median_ms"])
    timings["packed_full"]["speedup_vs_baseline"] = (
        timings["baseline_full"]["median_ms"]
        / timings["packed_full"]["median_ms"])

    report = {
        "device": torch.cuda.get_device_name(device),
        "config": vars(args) | {"out": str(args.out)},
        "shapes": {
            "routed_w13": list(routed_w13.shape),
            "shared_w13": list(shared_w13.shape),
            "packed_w13": list(packed_w13.shape),
            "routed_w2": list(routed_w2.shape),
            "shared_w2": list(shared_w2.shape),
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
