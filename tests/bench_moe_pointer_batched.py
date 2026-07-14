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
    spec = importlib.util.spec_from_file_location(
        "corex_moe_pointer_batched", path)
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
    parser.add_argument("--extension", type=Path, required=True)
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
    extension = load_extension(args.extension)
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
    gate_up = torch.empty((top_k, 2 * intermediate), device=device,
                          dtype=dtype)
    expert_out = torch.empty((top_k, hidden), device=device, dtype=dtype)
    pointer_workspace = torch.empty(3 * top_k, device=device,
                                    dtype=torch.int64)

    def route() -> tuple[torch.Tensor, torch.Tensor]:
        logits, ids = torch.topk(router_logits.float(), top_k, dim=-1)
        return torch.softmax(logits, dim=-1).to(dtype), ids

    def finish(current_gate_up: torch.Tensor, weights: torch.Tensor,
               ids: torch.Tensor, fp32_accumulate: bool) -> torch.Tensor:
        gate, up = current_gate_up.chunk(2, dim=-1)
        activation = F.silu(gate) * up
        extension.selected_gemv_out(
            activation, ids[0], w2, expert_out, pointer_workspace,
            True, fp32_accumulate)
        return (expert_out * weights[0].unsqueeze(-1)).sum(
            0, keepdim=True).to(dtype)

    def reference() -> torch.Tensor:
        weights, ids = route()
        selected_w13 = w13[ids[0]]
        selected_w2 = w2[ids[0]]
        current_gate_up = F.linear(
            hidden_states, selected_w13.reshape(-1, hidden)).view(top_k, -1)
        gate, up = current_gate_up.chunk(2, dim=-1)
        activation = F.silu(gate) * up
        current_expert_out = torch.bmm(
            selected_w2, activation.unsqueeze(-1)).squeeze(-1)
        return (current_expert_out * weights[0].unsqueeze(-1)).sum(
            0, keepdim=True).to(dtype)

    def candidate(fp32_accumulate: bool) -> torch.Tensor:
        weights, ids = route()
        extension.selected_gemv_out(
            hidden_states, ids[0], w13, gate_up, pointer_workspace,
            False, fp32_accumulate)
        return finish(gate_up, weights, ids, fp32_accumulate)

    cases: dict[str, Case] = {
        "reference": reference,
        "pointer_hgemm": lambda: candidate(False),
        "pointer_fp32": lambda: candidate(True),
    }
    reference_output = reference()
    torch.cuda.synchronize()
    results = {}
    for name, case in cases.items():
        output = case()
        torch.cuda.synchronize()
        difference = (output.float() - reference_output.float()).abs()
        trials = measure(case, args.warmup, args.iterations, args.repeats)
        results[name] = {
            "exact": bool(torch.equal(output, reference_output)),
            "close": bool(torch.allclose(
                output, reference_output, rtol=1e-3, atol=1e-3)),
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
            "median_ms": statistics.median(trials),
            "p10_ms": percentile(trials, 10),
            "p90_ms": percentile(trials, 90),
            "trials_ms": trials,
        }

    baseline = results["reference"]["median_ms"]
    for result in results.values():
        result["speedup_vs_reference"] = baseline / result["median_ms"]

    report = {
        "device": torch.cuda.get_device_name(device),
        "config": vars(args) | {
            "extension": str(args.extension), "out": str(args.out)},
        "results": results,
    }
    report["ok"] = bool(results["pointer_hgemm"]["close"]
                        or results["pointer_fp32"]["close"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
