#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch

from vllm.model_executor.layers.linear import ReplicatedLinear


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
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    hidden_size, local_qg_dim, kv_dim = 2048, 2048, 512
    sizes = (local_qg_dim, kv_dim, kv_dim)
    dtype = torch.float16

    qg_weight, k_weight, v_weight = [
        torch.randn((size, hidden_size), device=device, dtype=dtype,
                    generator=generator) * 0.02
        for size in sizes
    ]
    packed_weight = torch.cat((qg_weight, k_weight, v_weight), dim=0)

    def make_layer(output_size: int, prefix: str) -> ReplicatedLinear:
        return ReplicatedLinear(
            hidden_size, output_size, bias=False, params_dtype=dtype,
            quant_config=None, prefix=prefix).to(device)

    qg_layer = make_layer(local_qg_dim, "probe.qg")
    k_layer = make_layer(kv_dim, "probe.k")
    v_layer = make_layer(kv_dim, "probe.v")
    packed_layer = make_layer(sum(sizes), "probe.qgkv")
    qg_layer.weight_loader(qg_layer.weight, qg_weight)
    k_layer.weight_loader(k_layer.weight, k_weight)
    v_layer.weight_loader(v_layer.weight, v_weight)
    packed_layer.weight_loader(packed_layer.weight, packed_weight)

    reports = {}
    all_exact = True
    for tokens in [int(value) for value in args.tokens.split(",")]:
        hidden = torch.randn(
            (tokens, hidden_size), device=device, dtype=dtype,
            generator=generator)

        def separate() -> tuple[torch.Tensor, ...]:
            qg, _ = qg_layer(hidden)
            key, _ = k_layer(hidden)
            value, _ = v_layer(hidden)
            return qg, key, value

        def packed() -> tuple[torch.Tensor, ...]:
            projected, _ = packed_layer(hidden)
            return torch.split(projected, sizes, dim=-1)

        expected = separate()
        actual = packed()
        torch.cuda.synchronize()
        checks = []
        for reference, candidate in zip(expected, actual):
            difference = (candidate.float() - reference.float()).abs()
            checks.append({
                "exact": bool(torch.equal(candidate, reference)),
                "max_abs": float(difference.max()),
                "mean_abs": float(difference.mean()),
            })
        exact = all(check["exact"] for check in checks)
        all_exact = all_exact and exact

        timings = {}
        for name, case in {"separate": separate, "packed": packed}.items():
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
            "exact": exact,
            "checks": checks,
            "timings": timings,
        }

    report = {
        "device": torch.cuda.get_device_name(device),
        "config": vars(args) | {"out": str(args.out)},
        "shapes": {
            "qg_local": list(qg_weight.shape),
            "k_replicated": list(k_weight.shape),
            "v_replicated": list(v_weight.shape),
            "packed_rank_local": list(packed_weight.shape),
        },
        "tokens": reports,
        "ok": all_exact,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
