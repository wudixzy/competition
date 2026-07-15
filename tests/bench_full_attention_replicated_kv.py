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
    parser.add_argument("--tokens", default="1,64")
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--local-qg-size", type=int, default=2048)
    parser.add_argument("--replicated-kv-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=30)
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
    sizes = (args.local_qg_size, args.replicated_kv_size,
             args.replicated_kv_size)
    qg_weight, k_weight, v_weight = [
        torch.randn((size, args.hidden_size), device=device, dtype=dtype,
                    generator=generator) * 0.02
        for size in sizes
    ]
    kv_weight = torch.cat((k_weight, v_weight), dim=0)
    qgkv_weight = torch.cat((qg_weight, k_weight, v_weight), dim=0)

    reports = {}
    all_exact = True
    for tokens in [int(value) for value in args.tokens.split(",")]:
        hidden = torch.randn(
            (tokens, args.hidden_size), device=device, dtype=dtype,
            generator=generator)

        def separate() -> tuple[torch.Tensor, ...]:
            return (F.linear(hidden, qg_weight),
                    F.linear(hidden, k_weight),
                    F.linear(hidden, v_weight))

        def merged_kv() -> tuple[torch.Tensor, ...]:
            qg = F.linear(hidden, qg_weight)
            kv = F.linear(hidden, kv_weight)
            k, v = torch.split(
                kv, (args.replicated_kv_size,
                     args.replicated_kv_size), dim=-1)
            return qg, k, v

        def merged_all_upper_bound() -> tuple[torch.Tensor, ...]:
            projected = F.linear(hidden, qgkv_weight)
            return torch.split(projected, sizes, dim=-1)

        cases: dict[str, Case] = {
            "separate": separate,
            "merged_kv": merged_kv,
            "merged_all_upper_bound": merged_all_upper_bound,
        }
        reference = separate()
        checks = {}
        for name, case in cases.items():
            output = case()
            torch.cuda.synchronize()
            shards = []
            for expected, actual in zip(reference, output):
                difference = (actual.float() - expected.float()).abs()
                shards.append({
                    "exact": bool(torch.equal(actual, expected)),
                    "max_abs": float(difference.max()),
                    "mean_abs": float(difference.mean()),
                })
            checks[name] = shards
        exact = all(item["exact"] for item in checks["merged_kv"])
        all_exact = all_exact and exact

        timings = {}
        for name, case in cases.items():
            trials = measure(
                case, args.warmup, args.iterations, args.repeats)
            timings[name] = {
                "median_ms": statistics.median(trials),
                "p10_ms": percentile(trials, 10),
                "p90_ms": percentile(trials, 90),
                "trials_ms": trials,
            }
        baseline = timings["separate"]["median_ms"]
        for result in timings.values():
            result["speedup_vs_separate"] = baseline / result["median_ms"]
        reports[str(tokens)] = {
            "checks": checks,
            "merged_kv_exact": exact,
            "timings": timings,
        }

    report = {
        "device": torch.cuda.get_device_name(device),
        "authoritative_config": {
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "tensor_parallel_size": 4,
        },
        "config": vars(args) | {"out": str(args.out)},
        "weight_shapes": {
            "qg_local": list(qg_weight.shape),
            "k_replicated": list(k_weight.shape),
            "v_replicated": list(v_weight.shape),
            "kv_merged": list(kv_weight.shape),
        },
        "tokens": reports,
        "ok": all_exact,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if all_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
