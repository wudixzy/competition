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


TensorPair = tuple[torch.Tensor, torch.Tensor]


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def route_full_softmax(router_logits: torch.Tensor,
                       top_k: int) -> TensorPair:
    routing_weights = torch.softmax(router_logits.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(
        routing_weights, top_k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.to(router_logits.dtype), topk_ids


def route_topk_logits(router_logits: torch.Tensor,
                      top_k: int) -> TensorPair:
    topk_logits, topk_ids = torch.topk(
        router_logits.float(), top_k, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1)
    return topk_weights.to(router_logits.dtype), topk_ids


def expert_forward(hidden_states: torch.Tensor, topk_weights: torch.Tensor,
                   topk_ids: torch.Tensor, w13_sel: torch.Tensor,
                   w2_sel: torch.Tensor) -> torch.Tensor:
    top_k = topk_ids.shape[-1]
    hidden_size = hidden_states.shape[-1]
    gate_up = F.linear(hidden_states, w13_sel.reshape(-1, hidden_size))
    gate, up = gate_up.view(top_k, -1).chunk(2, dim=-1)
    act = F.silu(gate) * up
    expert_out = torch.bmm(w2_sel, act.unsqueeze(-1)).squeeze(-1)
    return (expert_out * topk_weights[0].unsqueeze(-1)).sum(
        0, keepdim=True).to(hidden_states.dtype)


def build_cases(args: argparse.Namespace) -> tuple[dict[str, Callable], dict]:
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

    workspace = torch.empty(
        args.top_k * 2 * args.intermediate_size * args.hidden_size,
        device=device, dtype=dtype)
    w13_workspace = workspace.view(
        args.top_k, 2 * args.intermediate_size, args.hidden_size)
    w2_workspace = workspace[:args.top_k * args.hidden_size
                             * args.intermediate_size].view(
        args.top_k, args.hidden_size, args.intermediate_size)

    def gather_forward(route: Callable, use_index_select: bool = False,
                       gate_bmm: bool = False) -> torch.Tensor:
        topk_weights, topk_ids = route(router_logits, args.top_k)
        eids = topk_ids[0]
        if use_index_select:
            w13_sel = torch.index_select(w13, 0, eids)
            w2_sel = torch.index_select(w2, 0, eids)
        else:
            w13_sel = w13[eids]
            w2_sel = w2[eids]
        if not gate_bmm:
            return expert_forward(
                hidden_states, topk_weights, topk_ids, w13_sel, w2_sel)

        hidden_batch = hidden_states.expand(args.top_k, -1).unsqueeze(-1)
        gate_up = torch.bmm(w13_sel, hidden_batch).squeeze(-1)
        gate, up = gate_up.chunk(2, dim=-1)
        act = F.silu(gate) * up
        expert_out = torch.bmm(w2_sel, act.unsqueeze(-1)).squeeze(-1)
        return (expert_out * topk_weights[0].unsqueeze(-1)).sum(
            0, keepdim=True).to(hidden_states.dtype)

    def workspace_forward(route: Callable) -> torch.Tensor:
        topk_weights, topk_ids = route(router_logits, args.top_k)
        eids = topk_ids[0]
        torch.index_select(w13, 0, eids, out=w13_workspace)
        gate_up = F.linear(
            hidden_states, w13_workspace.reshape(-1, args.hidden_size))
        gate, up = gate_up.view(args.top_k, -1).chunk(2, dim=-1)
        act = F.silu(gate) * up
        torch.index_select(w2, 0, eids, out=w2_workspace)
        expert_out = torch.bmm(
            w2_workspace, act.unsqueeze(-1)).squeeze(-1)
        return (expert_out * topk_weights[0].unsqueeze(-1)).sum(
            0, keepdim=True).to(hidden_states.dtype)

    cases: dict[str, Callable] = {
        "existing": lambda: gather_forward(route_full_softmax),
        "index_select": lambda: gather_forward(
            route_full_softmax, use_index_select=True),
        "workspace": lambda: workspace_forward(route_full_softmax),
        "gate_bmm": lambda: gather_forward(
            route_full_softmax, gate_bmm=True),
        "topk_logits": lambda: gather_forward(route_topk_logits),
        "workspace_topk_logits": lambda: workspace_forward(
            route_topk_logits),
    }
    metadata = {
        "weight_shapes": {
            "w13": list(w13.shape),
            "w2": list(w2.shape),
        },
        "workspace_bytes": workspace.numel() * workspace.element_size(),
        "allocated_bytes": torch.cuda.memory_allocated(device),
    }
    return cases, metadata


def measure(case: Callable, warmup: int, iterations: int,
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
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    cases, metadata = build_cases(args)
    reference = cases["existing"]()
    torch.cuda.synchronize()

    results = {}
    for name, case in cases.items():
        output = case()
        torch.cuda.synchronize()
        difference = (output.float() - reference.float()).abs()
        trials = measure(case, args.warmup, args.iterations, args.repeats)
        results[name] = {
            "exact": bool(torch.equal(output, reference)),
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
            "p10_ms": percentile(trials, 10),
            "median_ms": statistics.median(trials),
            "p90_ms": percentile(trials, 90),
            "trials_ms": trials,
        }

    baseline_ms = results["existing"]["median_ms"]
    for result in results.values():
        result["speedup_vs_existing"] = baseline_ms / result["median_ms"]

    report = {
        "config": vars(args) | {"out": str(args.out)},
        "metadata": metadata,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
