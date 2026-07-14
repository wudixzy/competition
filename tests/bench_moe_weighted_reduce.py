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


ReduceFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def reduce_existing(expert_out: torch.Tensor,
                    weights: torch.Tensor) -> torch.Tensor:
    return (expert_out * weights.unsqueeze(-1)).sum(0, keepdim=True)


def reduce_matmul(expert_out: torch.Tensor,
                  weights: torch.Tensor) -> torch.Tensor:
    return torch.matmul(weights.unsqueeze(0), expert_out)


def reduce_mv(expert_out: torch.Tensor,
              weights: torch.Tensor) -> torch.Tensor:
    return torch.mv(expert_out.transpose(0, 1), weights).unsqueeze(0)


def reduce_linear(expert_out: torch.Tensor,
                  weights: torch.Tensor) -> torch.Tensor:
    return F.linear(weights.unsqueeze(0), expert_out.transpose(0, 1))


def reduce_einsum(expert_out: torch.Tensor,
                  weights: torch.Tensor) -> torch.Tensor:
    return torch.einsum("kh,k->h", expert_out, weights).unsqueeze(0)


def measure(case: Callable[[], torch.Tensor], warmup: int, iterations: int,
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


def summarize(cases: dict[str, Callable[[], torch.Tensor]], warmup: int,
              iterations: int, repeats: int) -> dict:
    reference = cases["existing"]()
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

    baseline_ms = results["existing"]["median_ms"]
    for result in results.values():
        result["speedup_vs_existing"] = baseline_ms / result["median_ms"]
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
    eids = topk_ids[0]
    w13_sel = w13[eids]
    w2_sel = w2[eids]
    gate_up = F.linear(
        hidden_states, w13_sel.reshape(-1, args.hidden_size))
    gate, up = gate_up.view(args.top_k, -1).chunk(2, dim=-1)
    act = F.silu(gate) * up
    expert_out = torch.bmm(w2_sel, act.unsqueeze(-1)).squeeze(-1)

    reducers: dict[str, ReduceFn] = {
        "existing": reduce_existing,
        "matmul": reduce_matmul,
        "mv": reduce_mv,
        "linear": reduce_linear,
        "einsum": reduce_einsum,
    }
    reduce_cases = {
        name: (lambda fn=fn: fn(expert_out, weights))
        for name, fn in reducers.items()
    }

    def full_forward(reduce_fn: ReduceFn) -> torch.Tensor:
        current_topk_logits, current_topk_ids = torch.topk(
            router_logits.float(), args.top_k, dim=-1)
        current_weights = torch.softmax(
            current_topk_logits, dim=-1)[0].to(dtype)
        current_w13 = w13[current_topk_ids[0]]
        current_w2 = w2[current_topk_ids[0]]
        current_gate_up = F.linear(
            hidden_states, current_w13.reshape(-1, args.hidden_size))
        current_gate, current_up = current_gate_up.view(
            args.top_k, -1).chunk(2, dim=-1)
        current_act = F.silu(current_gate) * current_up
        current_expert_out = torch.bmm(
            current_w2, current_act.unsqueeze(-1)).squeeze(-1)
        return reduce_fn(current_expert_out, current_weights).to(dtype)

    full_cases = {
        name: (lambda fn=fn: full_forward(fn))
        for name, fn in reducers.items()
    }

    def preweight_act_forward() -> torch.Tensor:
        current_topk_logits, current_topk_ids = torch.topk(
            router_logits.float(), args.top_k, dim=-1)
        current_weights = torch.softmax(
            current_topk_logits, dim=-1)[0].to(dtype)
        current_w13 = w13[current_topk_ids[0]]
        current_w2 = w2[current_topk_ids[0]]
        current_gate_up = F.linear(
            hidden_states, current_w13.reshape(-1, args.hidden_size))
        current_gate, current_up = current_gate_up.view(
            args.top_k, -1).chunk(2, dim=-1)
        current_act = F.silu(current_gate) * current_up
        current_act = current_act * current_weights.unsqueeze(-1)
        return torch.bmm(
            current_w2, current_act.unsqueeze(-1)).squeeze(-1).sum(
                0, keepdim=True).to(dtype)

    full_cases["preweight_act"] = preweight_act_forward
    report = {
        "config": vars(args) | {"out": str(args.out)},
        "metadata": {
            "w13_shape": list(w13.shape),
            "w2_shape": list(w2.shape),
            "expert_out_shape": list(expert_out.shape),
            "allocated_bytes": torch.cuda.memory_allocated(device),
        },
        "reduce_only": summarize(
            reduce_cases, args.warmup, args.iterations, args.repeats),
        "full_path": summarize(
            full_cases, args.warmup, args.iterations, args.repeats),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
