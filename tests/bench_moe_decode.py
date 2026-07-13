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


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def benchmark(
    function: Callable[[], torch.Tensor | tuple[torch.Tensor, ...]],
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, object]:
    result = None
    for _ in range(warmup):
        result = function()
    torch.cuda.synchronize()

    trials = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(iterations):
            result = function()
        torch.cuda.synchronize()
        trials.append((time.perf_counter() - started) * 1000.0 / iterations)
    assert result is not None
    return {
        "median_ms": statistics.median(trials),
        "p10_ms": percentile(trials, 10),
        "p90_ms": percentile(trials, 90),
        "trials_ms": trials,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.top_k > args.experts:
        parser.error("--top-k cannot exceed --experts")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float16
    hidden = torch.randn(1, args.hidden, device=device, dtype=dtype)
    logits = torch.randn(1, args.experts, device=device, dtype=dtype)
    eids = torch.arange(args.top_k, device=device, dtype=torch.long)
    weights = torch.softmax(
        torch.randn(args.top_k, device=device, dtype=torch.float32), dim=0,
    ).to(dtype)

    # Full TP-sharded tensors are zero-filled to commit realistic HBM pages
    # without spending time initializing 384 MiB of random expert parameters.
    w13 = torch.zeros(
        args.experts, 2 * args.intermediate, args.hidden,
        device=device, dtype=dtype,
    )
    w2 = torch.zeros(
        args.experts, args.hidden, args.intermediate,
        device=device, dtype=dtype,
    )
    torch.cuda.synchronize()

    weight_scale = 0.02
    selected_w13 = torch.randn(
        args.top_k, 2 * args.intermediate, args.hidden,
        device=device, dtype=dtype,
    ) * weight_scale
    selected_w2 = torch.randn(
        args.top_k, args.hidden, args.intermediate,
        device=device, dtype=dtype,
    ) * weight_scale
    w13[:args.top_k].copy_(selected_w13)
    w2[:args.top_k].copy_(selected_w2)

    def compute_flat(w13_selected: torch.Tensor,
                     w2_selected: torch.Tensor) -> torch.Tensor:
        gate_up = F.linear(hidden, w13_selected.reshape(-1, args.hidden))
        gate, up = gate_up.view(args.top_k, -1).chunk(2, dim=-1)
        activation = F.silu(gate) * up
        expert_out = torch.bmm(
            w2_selected, activation.unsqueeze(-1)).squeeze(-1)
        return (expert_out * weights.unsqueeze(-1)).sum(0, keepdim=True)

    def compute_bmm(w13_selected: torch.Tensor,
                    w2_selected: torch.Tensor) -> torch.Tensor:
        expanded_hidden = hidden.expand(args.top_k, -1).unsqueeze(-1)
        gate_up = torch.bmm(w13_selected, expanded_hidden).squeeze(-1)
        gate, up = gate_up.chunk(2, dim=-1)
        activation = F.silu(gate) * up
        expert_out = torch.bmm(
            w2_selected, activation.unsqueeze(-1)).squeeze(-1)
        return (expert_out * weights.unsqueeze(-1)).sum(0, keepdim=True)

    def compute_loop_views(expert_ids: list[int]) -> torch.Tensor:
        outputs = []
        for position, expert_id in enumerate(expert_ids):
            gate_up = F.linear(hidden, w13[expert_id])
            gate, up = gate_up.chunk(2, dim=-1)
            activation = F.silu(gate) * up
            outputs.append(F.linear(activation, w2[expert_id]))
        expert_out = torch.cat(outputs, dim=0)
        return (expert_out * weights.unsqueeze(-1)).sum(0, keepdim=True)

    fixed_ids = list(range(args.top_k))

    with torch.no_grad():
        flat_reference = compute_flat(selected_w13, selected_w2)
        bmm_candidate = compute_bmm(selected_w13, selected_w2)
        current_reference = compute_flat(w13[eids], w2[eids])
        loop_candidate = compute_loop_views(fixed_ids)
        if not all(torch.isfinite(output).all().item() for output in (
                flat_reference, bmm_candidate, current_reference,
                loop_candidate)):
            raise RuntimeError("non-finite MoE output; benchmark parity is invalid")
        route_full = torch.softmax(logits.float(), dim=-1)
        route_weights, route_ids = torch.topk(
            route_full, args.top_k, dim=-1)
        route_weights = route_weights / route_weights.sum(dim=-1, keepdim=True)
        selected_logits, selected_ids = torch.topk(
            logits.float(), args.top_k, dim=-1)
        selected_weights = torch.softmax(selected_logits, dim=-1)

        parity = {
            "bmm_max_abs": float(
                (flat_reference - bmm_candidate).abs().max().item()),
            "bmm_max_rel": float((
                (flat_reference - bmm_candidate).abs()
                / flat_reference.abs().clamp_min(1e-6)
            ).max().item()),
            "current_max_abs": float(
                (flat_reference - current_reference).abs().max().item()),
            "loop_max_abs": float(
                (flat_reference - loop_candidate).abs().max().item()),
            "loop_mean_abs": float(
                (flat_reference - loop_candidate).abs().mean().item()),
            "all_outputs_finite": True,
            "route_ids_equal": bool(torch.equal(route_ids, selected_ids)),
            "route_weight_max_abs": float(
                (route_weights - selected_weights).abs().max().item()),
        }

        functions = {
            "gather_advanced": lambda: (w13[eids], w2[eids]),
            "gather_index_select": lambda: (
                torch.index_select(w13, 0, eids),
                torch.index_select(w2, 0, eids),
            ),
            "current_advanced": lambda: compute_flat(w13[eids], w2[eids]),
            "current_index_select": lambda: compute_flat(
                torch.index_select(w13, 0, eids),
                torch.index_select(w2, 0, eids),
            ),
            "double_bmm_advanced": lambda: compute_bmm(w13[eids], w2[eids]),
            "loop_views_fixed_ids": lambda: compute_loop_views(fixed_ids),
            "loop_views_sync_ids": lambda: compute_loop_views(eids.tolist()),
            "compute_flat_preselected": lambda: compute_flat(
                selected_w13, selected_w2),
            "compute_bmm_preselected": lambda: compute_bmm(
                selected_w13, selected_w2),
            "route_full_softmax": lambda: torch.topk(
                torch.softmax(logits.float(), dim=-1),
                args.top_k, dim=-1,
            ),
            "route_selected_softmax": lambda: (
                lambda values_ids: (
                    torch.softmax(values_ids.values, dim=-1),
                    values_ids.indices,
                )
            )(torch.topk(logits.float(), args.top_k, dim=-1)),
        }
        timings = {
            name: benchmark(function, args.warmup, args.iterations, args.repeats)
            for name, function in functions.items()
        }

    current_ms = float(timings["current_advanced"]["median_ms"])
    for result in timings.values():
        result["speedup_vs_current"] = current_ms / float(result["median_ms"])

    report = {
        "shape": {
            "experts": args.experts,
            "hidden": args.hidden,
            "intermediate_per_tp_rank": args.intermediate,
            "top_k": args.top_k,
            "dtype": str(dtype),
            "device": str(device),
        },
        "config": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "seed": args.seed,
            "weight_scale": weight_scale,
        },
        "parity": parity,
        "timings": timings,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
