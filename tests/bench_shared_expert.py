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
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def benchmark(function: Callable[[], torch.Tensor], warmup: int,
              iterations: int, repeats: int) -> dict[str, object]:
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
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float16
    scale = 0.02
    hidden = torch.randn(1, args.hidden, device=device, dtype=dtype)
    w13 = torch.randn(
        2 * args.intermediate, args.hidden, device=device, dtype=dtype,
    ) * scale
    w2 = torch.randn(
        args.hidden, args.intermediate, device=device, dtype=dtype,
    ) * scale
    gate_weight = torch.randn(
        1, args.hidden, device=device, dtype=dtype,
    ) * scale
    fused_weight = torch.cat((w13, gate_weight), dim=0).contiguous()
    activation_input = torch.randn(
        1, args.intermediate, device=device, dtype=dtype,
    )

    def activate(gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        return F.silu(gate) * up

    def baseline() -> torch.Tensor:
        activation = activate(F.linear(hidden, w13))
        output = F.linear(activation, w2)
        score = torch.sigmoid(F.linear(hidden, gate_weight))
        return output * score

    def fused_projection_post_gate() -> torch.Tensor:
        projected = F.linear(hidden, fused_weight)
        activation = activate(projected[:, :-1].contiguous())
        output = F.linear(activation, w2)
        return output * torch.sigmoid(projected[:, -1:])

    def separate_projection_pre_gate() -> torch.Tensor:
        activation = activate(F.linear(hidden, w13))
        score = torch.sigmoid(F.linear(hidden, gate_weight))
        return F.linear(activation * score, w2)

    def fused_projection_pre_gate() -> torch.Tensor:
        projected = F.linear(hidden, fused_weight)
        activation = activate(projected[:, :-1].contiguous())
        return F.linear(
            activation * torch.sigmoid(projected[:, -1:]), w2,
        )

    functions = {
        "baseline": baseline,
        "fused_projection_post_gate": fused_projection_post_gate,
        "separate_projection_pre_gate": separate_projection_pre_gate,
        "fused_projection_pre_gate": fused_projection_pre_gate,
        "gate_projection_only": lambda: F.linear(hidden, gate_weight),
        "gate_up_projection_only": lambda: F.linear(hidden, w13),
        "down_projection_only": lambda: F.linear(activation_input, w2),
    }

    with torch.no_grad():
        reference = baseline()
        candidates = {
            name: function() for name, function in functions.items()
            if name in {
                "fused_projection_post_gate",
                "separate_projection_pre_gate",
                "fused_projection_pre_gate",
            }
        }
        if not all(torch.isfinite(value).all().item()
                   for value in (reference, *candidates.values())):
            raise RuntimeError("non-finite shared-expert output")
        parity = {}
        for name, value in candidates.items():
            delta = (reference - value).abs()
            parity[name] = {
                "max_abs": float(delta.max().item()),
                "mean_abs": float(delta.mean().item()),
                "max_rel": float(
                    (delta / reference.abs().clamp_min(1e-6)).max().item()),
            }
        timings = {
            name: benchmark(function, args.warmup, args.iterations,
                            args.repeats)
            for name, function in functions.items()
        }

    baseline_ms = float(timings["baseline"]["median_ms"])
    for result in timings.values():
        result["speedup_vs_baseline"] = (
            baseline_ms / float(result["median_ms"]))

    report = {
        "shape": {
            "hidden": args.hidden,
            "intermediate_per_tp_rank": args.intermediate,
            "dtype": str(dtype),
            "device": str(device),
        },
        "config": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "seed": args.seed,
            "weight_scale": scale,
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
