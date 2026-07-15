#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch


Case = Callable[[], tuple[torch.Tensor, torch.Tensor]]


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


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
    parser.add_argument("--value-heads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    dtype = torch.float16
    batch = 1
    key_heads = 4
    value_heads = args.value_heads
    if value_heads % key_heads:
        parser.error("--value-heads must be divisible by four key heads")
    head_dim = 128
    expand_ratio = value_heads // key_heads
    scale = head_dim ** -0.5

    q = torch.randn(
        (batch, 1, key_heads, head_dim), device=device, dtype=dtype,
        generator=generator)
    k = torch.randn(
        (batch, 1, key_heads, head_dim), device=device, dtype=dtype,
        generator=generator)
    v = torch.randn(
        (batch, value_heads, head_dim), device=device, dtype=dtype,
        generator=generator).float()
    decay = torch.full(
        (batch, value_heads), 0.99, device=device, dtype=torch.float32)
    beta = torch.full(
        (batch, value_heads), 0.5, device=device, dtype=torch.float32)
    initial_state = torch.randn(
        (batch, value_heads, head_dim, head_dim), device=device,
        dtype=torch.float32, generator=generator) * 0.01

    def recurrent(q_t: torch.Tensor,
                  k_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = initial_state.clone()
        state.mul_(decay[:, :, None, None])
        state_flat = state.view(-1, head_dim, head_dim)
        bh = state_flat.shape[0]
        kv_mem = torch.bmm(
            k_t.view(bh, 1, head_dim), state_flat,
        ).view(batch, value_heads, head_dim)
        delta = (v - kv_mem) * beta[:, :, None]
        state_flat.baddbmm_(
            k_t.view(bh, head_dim, 1),
            delta.view(bh, 1, head_dim),
        )
        output = torch.bmm(
            q_t.view(bh, 1, head_dim), state_flat,
        ).view(batch, value_heads, head_dim)
        return output, state

    def baseline() -> tuple[torch.Tensor, torch.Tensor]:
        q_expanded = q.repeat_interleave(expand_ratio, dim=2)
        k_expanded = k.repeat_interleave(expand_ratio, dim=2)
        q_t = l2norm(q_expanded.squeeze(1)).float() * scale
        k_t = l2norm(k_expanded.squeeze(1)).float()
        return recurrent(q_t, k_t)

    def candidate() -> tuple[torch.Tensor, torch.Tensor]:
        q_t = l2norm(q.squeeze(1)).repeat_interleave(
            expand_ratio, dim=1).float() * scale
        k_t = l2norm(k.squeeze(1)).repeat_interleave(
            expand_ratio, dim=1).float()
        return recurrent(q_t, k_t)

    reference_output, reference_state = baseline()
    torch.cuda.synchronize()
    results = {}
    for name, case in {"baseline": baseline, "candidate": candidate}.items():
        output, state = case()
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(output).all()
                      and torch.isfinite(state).all())
        output_diff = (output - reference_output).abs()
        state_diff = (state - reference_state).abs()
        trials = measure(case, args.warmup, args.iterations, args.repeats)
        results[name] = {
            "finite": finite,
            "output_exact": bool(finite and torch.equal(
                output, reference_output)),
            "state_exact": bool(finite and torch.equal(state, reference_state)),
            "output_max_abs": float(output_diff.max()),
            "state_max_abs": float(state_diff.max()),
            "median_ms": statistics.median(trials),
            "p10_ms": percentile(trials, 10),
            "p90_ms": percentile(trials, 90),
            "trials_ms": trials,
        }

    baseline_ms = results["baseline"]["median_ms"]
    for result in results.values():
        result["speedup_vs_baseline"] = baseline_ms / result["median_ms"]

    report = {
        "device": torch.cuda.get_device_name(device),
        "config": vars(args) | {"out": str(args.out)},
        "shapes": {
            "q_k_before_expand": list(q.shape),
            "q_k_after_expand": [batch, 1, value_heads, head_dim],
            "temporal_state": list(initial_state.shape),
        },
        "results": results,
    }
    report["ok"] = bool(all(
        result["finite"]
        and result["output_exact"]
        and result["state_exact"]
        for result in results.values()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
