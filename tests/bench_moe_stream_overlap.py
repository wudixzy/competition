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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = torch.float16
    generator = torch.Generator(device=device).manual_seed(args.seed)
    hidden = torch.randn(
        (1, 2048), device=device, dtype=dtype, generator=generator)

    router_shared_weight = torch.full(
        (257, 2048), 0.01, device=device, dtype=dtype)
    routed_w13 = torch.full(
        (256, 256, 2048), 0.01, device=device, dtype=dtype)
    routed_w2 = torch.full(
        (256, 2048, 128), 0.01, device=device, dtype=dtype)
    shared_w13 = torch.full(
        (256, 2048), 0.01, device=device, dtype=dtype)
    shared_w2 = torch.full(
        (2048, 128), 0.01, device=device, dtype=dtype)
    shared_activation = SiluAndMul()
    shared_stream = torch.cuda.Stream()
    routed_stream = torch.cuda.Stream()

    def route() -> tuple[torch.Tensor, torch.Tensor]:
        router_and_gate = F.linear(hidden, router_shared_weight)
        return router_and_gate[..., :256], router_and_gate[..., 256:]

    def routed(router_logits: torch.Tensor) -> torch.Tensor:
        topk_logits, topk_ids = torch.topk(router_logits.float(), 8, dim=-1)
        weights = torch.softmax(topk_logits, dim=-1)[0].to(dtype)
        w13 = routed_w13[topk_ids[0]]
        w2 = routed_w2[topk_ids[0]]
        gate_up = F.linear(hidden, w13.reshape(-1, 2048)).view(8, -1)
        gate, up = gate_up.chunk(2, dim=-1)
        act = F.silu(gate) * up
        expert_out = torch.bmm(w2, act.unsqueeze(-1)).squeeze(-1)
        return (expert_out * weights.unsqueeze(-1)).sum(0, keepdim=True)

    def shared(gate_score: torch.Tensor) -> torch.Tensor:
        gate_up = F.linear(hidden, shared_w13)
        act = shared_activation(gate_up)
        output = F.linear(act, shared_w2)
        return output * torch.sigmoid(gate_score)

    def sequential() -> torch.Tensor:
        router_logits, gate_score = route()
        routed_out = routed(router_logits)
        shared_out = shared(gate_score)
        return routed_out + shared_out

    def shared_on_aux() -> torch.Tensor:
        router_logits, gate_score = route()
        current = torch.cuda.current_stream()
        shared_stream.wait_stream(current)
        with torch.cuda.stream(shared_stream):
            shared_out = shared(gate_score)
        routed_out = routed(router_logits)
        current.wait_stream(shared_stream)
        shared_out.record_stream(current)
        return routed_out + shared_out

    def routed_on_aux() -> torch.Tensor:
        router_logits, gate_score = route()
        current = torch.cuda.current_stream()
        routed_stream.wait_stream(current)
        with torch.cuda.stream(routed_stream):
            routed_out = routed(router_logits)
        shared_out = shared(gate_score)
        current.wait_stream(routed_stream)
        routed_out.record_stream(current)
        return routed_out + shared_out

    cases: dict[str, Case] = {
        "sequential": sequential,
        "shared_on_aux": shared_on_aux,
        "routed_on_aux": routed_on_aux,
    }
    reference = sequential()
    torch.cuda.synchronize()
    results = {}
    for name, case in cases.items():
        output = case()
        torch.cuda.synchronize()
        finite = bool(
            torch.isfinite(reference).all() and torch.isfinite(output).all())
        difference = (output.float() - reference.float()).abs()
        trials = measure(case, args.warmup, args.iterations, args.repeats)
        results[name] = {
            "finite": finite,
            "exact": bool(finite and torch.equal(output, reference)),
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
            "median_ms": statistics.median(trials),
            "p10_ms": percentile(trials, 10),
            "p90_ms": percentile(trials, 90),
            "trials_ms": trials,
        }

    baseline_ms = results["sequential"]["median_ms"]
    for result in results.values():
        result["speedup_vs_sequential"] = baseline_ms / result["median_ms"]
    report = {
        "device": torch.cuda.get_device_name(device),
        "config": vars(args) | {"out": str(args.out)},
        "shapes": {
            "router_shared": list(router_shared_weight.shape),
            "routed_w13": list(routed_w13.shape),
            "routed_w2": list(routed_w2.shape),
            "shared_w13": list(shared_w13.shape),
            "shared_w2": list(shared_w2.shape),
        },
        "results": results,
    }
    report["ok"] = bool(all(
        result["finite"] and result["exact"] for result in results.values()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
