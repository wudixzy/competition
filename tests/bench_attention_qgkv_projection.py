#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F


def bench(fn, *, warmups: int, repeats: int, iterations: int) -> list[float]:
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


def summarize(samples: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-rank", type=int, required=True)
    parser.add_argument("--device-label", required=True)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    if not 0 <= args.tp_rank < 4:
        parser.error("--tp-rank must be in [0, 3]")

    torch.manual_seed(20260715)
    device = torch.device("cuda:0")
    hidden = 2048
    total_q_heads = 16
    total_kv_heads = 2
    head_dim = 256
    tp_size = 4
    local_qg_dim = total_q_heads * head_dim * 2 // tp_size
    full_kv_dim = total_kv_heads * head_dim
    selected_kv_dim = head_dim
    kv_head = args.tp_rank // (tp_size // total_kv_heads)
    kv_start = kv_head * head_dim

    wq = torch.randn(local_qg_dim, hidden, device=device,
                     dtype=torch.float16) * 0.01
    wk = torch.randn(full_kv_dim, hidden, device=device,
                     dtype=torch.float16) * 0.01
    wv = torch.randn(full_kv_dim, hidden, device=device,
                     dtype=torch.float16) * 0.01
    wk_selected = wk[kv_start:kv_start + selected_kv_dim].contiguous()
    wv_selected = wv[kv_start:kv_start + selected_kv_dim].contiguous()
    fused_weight = torch.cat([wq, wk_selected, wv_selected], dim=0)

    def current(x):
        qg = F.linear(x, wq)
        k_full = F.linear(x, wk)
        v_full = F.linear(x, wv)
        return (
            qg,
            k_full[:, kv_start:kv_start + selected_kv_dim],
            v_full[:, kv_start:kv_start + selected_kv_dim],
        )

    def selected_three(x):
        return (
            F.linear(x, wq),
            F.linear(x, wk_selected),
            F.linear(x, wv_selected),
        )

    def fused_selected(x):
        projected = F.linear(x, fused_weight)
        return torch.split(
            projected,
            [local_qg_dim, selected_kv_dim, selected_kv_dim],
            dim=-1,
        )

    results = {}
    for tokens, iterations in ((1, 300), (64, 30)):
        x = torch.randn(tokens, hidden, device=device, dtype=torch.float16)
        reference = current(x)
        variants = {
            "current_replicated_three": current,
            "selected_three": selected_three,
            "fused_selected": fused_selected,
        }
        parity = {}
        timings = {}
        for name, fn in variants.items():
            actual = fn(x)
            parity[name] = {
                "qg_max_abs": (actual[0] - reference[0]).abs().max().item(),
                "k_max_abs": (actual[1] - reference[1]).abs().max().item(),
                "v_max_abs": (actual[2] - reference[2]).abs().max().item(),
                "finite": all(bool(torch.isfinite(value).all().item())
                              for value in actual),
            }
            timings[name] = summarize(bench(
                lambda fn=fn: fn(x),
                warmups=args.warmups,
                repeats=args.repeats,
                iterations=iterations,
            ))
        baseline = timings["current_replicated_three"]["median_ms"]
        for timing in timings.values():
            timing["speedup_vs_current"] = baseline / timing["median_ms"]
        results[f"t{tokens}"] = {
            "iterations": iterations,
            "parity": parity,
            "timings": timings,
        }

    print(json.dumps({
        "device": args.device_label,
        "tp_rank": args.tp_rank,
        "kv_head": kv_head,
        "shape": {
            "hidden": hidden,
            "total_q_heads": total_q_heads,
            "virtual_gated_q_heads": total_q_heads * 2,
            "total_kv_heads": total_kv_heads,
            "head_dim": head_dim,
            "local_qg_dim": local_qg_dim,
            "full_kv_dim": full_kv_dim,
            "selected_kv_dim": selected_kv_dim,
            "fused_output_dim": local_qg_dim + 2 * selected_kv_dim,
            "dtype": "float16",
        },
        "results": results,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
