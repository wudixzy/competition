#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def current(q: torch.Tensor, k: torch.Tensor, ratio: int, scale: float):
    q_expanded = q.repeat_interleave(ratio, dim=2)
    k_expanded = k.repeat_interleave(ratio, dim=2)
    return (
        l2norm(q_expanded.squeeze(1)).float() * scale,
        l2norm(k_expanded.squeeze(1)).float(),
    )


def norm_then_half_expand(
    q: torch.Tensor, k: torch.Tensor, ratio: int, scale: float
):
    return (
        l2norm(q.squeeze(1)).repeat_interleave(ratio, dim=1).float() * scale,
        l2norm(k.squeeze(1)).repeat_interleave(ratio, dim=1).float(),
    )


def norm_then_float_expand(
    q: torch.Tensor, k: torch.Tensor, ratio: int, scale: float
):
    return (
        l2norm(q.squeeze(1)).float().repeat_interleave(ratio, dim=1) * scale,
        l2norm(k.squeeze(1)).float().repeat_interleave(ratio, dim=1),
    )


def recurrent_step(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    state_source: torch.Tensor,
    v_t: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = state_source.clone()
    state.mul_(decay[:, :, None, None])
    batch, heads, key_dim, value_dim = state.shape
    bh = batch * heads
    flat = state.view(bh, key_dim, value_dim)
    kv_mem = torch.bmm(k_t.view(bh, 1, key_dim), flat).view(
        batch, heads, value_dim
    )
    delta = (v_t - kv_mem) * beta[:, :, None]
    flat.baddbmm_(
        k_t.view(bh, key_dim, 1),
        delta.view(bh, 1, value_dim),
    )
    out = torch.bmm(q_t.view(bh, 1, key_dim), flat).view(
        batch, heads, value_dim
    )
    return out, state


def bench(fn, warmups: int, repeats: int, iterations: int) -> list[float]:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000 / iterations)
    return samples


def summary(samples: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-label", default="unknown")
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--prep-iterations", type=int, default=500)
    parser.add_argument("--full-iterations", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(20260715)
    device = torch.device("cuda:0")
    batch, key_heads, ratio, key_dim, value_dim = 1, 4, 3, 128, 128
    value_heads = key_heads * ratio
    scale = key_dim**-0.5
    q = torch.randn(batch, 1, key_heads, key_dim, device=device,
                    dtype=torch.float16)
    k = torch.randn_like(q)
    state = torch.randn(batch, value_heads, key_dim, value_dim,
                        device=device, dtype=torch.float32) * 0.01
    v = torch.randn(batch, value_heads, value_dim, device=device,
                    dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(batch, value_heads, device=device))
    decay = torch.sigmoid(torch.randn(batch, value_heads, device=device))

    variants = {
        "current": current,
        "norm_then_half_expand": norm_then_half_expand,
        "norm_then_float_expand": norm_then_float_expand,
    }
    prepared = {name: fn(q, k, ratio, scale) for name, fn in variants.items()}
    reference_q, reference_k = prepared["current"]
    parity = {}
    for name, (q_t, k_t) in prepared.items():
        ref_out, ref_state = recurrent_step(
            reference_q, reference_k, state, v, beta, decay
        )
        out, final_state = recurrent_step(q_t, k_t, state, v, beta, decay)
        parity[name] = {
            "q_max_abs": (q_t - reference_q).abs().max().item(),
            "k_max_abs": (k_t - reference_k).abs().max().item(),
            "out_max_abs": (out - ref_out).abs().max().item(),
            "state_max_abs": (final_state - ref_state).abs().max().item(),
            "finite": bool(torch.isfinite(out).all().item()
                           and torch.isfinite(final_state).all().item()),
        }

    prep_results = {}
    full_results = {}
    for name, fn in variants.items():
        prep_results[name] = summary(bench(
            lambda fn=fn: fn(q, k, ratio, scale),
            args.warmups, args.repeats, args.prep_iterations,
        ))
        full_results[name] = summary(bench(
            lambda fn=fn: recurrent_step(
                *fn(q, k, ratio, scale), state, v, beta, decay
            ),
            args.warmups, args.repeats, args.full_iterations,
        ))

    baseline_prep = prep_results["current"]["median_ms"]
    baseline_full = full_results["current"]["median_ms"]
    for name in variants:
        prep_results[name]["speedup_vs_current"] = (
            baseline_prep / prep_results[name]["median_ms"]
        )
        full_results[name]["speedup_vs_current"] = (
            baseline_full / full_results[name]["median_ms"]
        )

    print(json.dumps({
        "device": args.device_label,
        "shape": {
            "batch": batch,
            "key_heads": key_heads,
            "value_heads": value_heads,
            "key_dim": key_dim,
            "value_dim": value_dim,
            "dtype": "float16",
        },
        "parity": parity,
        "prep": prep_results,
        "full": full_results,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
