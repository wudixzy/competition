#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


LINEAR_LAYERS = 30
LOCAL_CONV_DIM = 2048
CONV_STATE_TOKENS = 3
LOCAL_VALUE_HEADS = 8
HEAD_DIM = 128


def main() -> int:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--min-saving-ms", type=float, default=15.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.repeats) <= 0:
        parser.error("warmup, iterations, and repeats must be positive")
    if args.min_saving_ms < 0:
        parser.error("--min-saving-ms must be non-negative")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    generator = torch.Generator(device="cpu").manual_seed(20260721)
    saved_conv = torch.randn(
        LINEAR_LAYERS, LOCAL_CONV_DIM, CONV_STATE_TOKENS,
        dtype=torch.float32, generator=generator)
    saved_temporal = torch.randn(
        LINEAR_LAYERS, LOCAL_VALUE_HEADS, HEAD_DIM, HEAD_DIM,
        dtype=torch.float32, generator=generator)
    live_conv = torch.empty_like(saved_conv, device=device)
    live_temporal = torch.empty_like(saved_temporal, device=device)

    def temporary_gpu_copy() -> None:
        live_conv.copy_(
            saved_conv.to(device=live_conv.device, dtype=live_conv.dtype),
            non_blocking=True)
        live_temporal.copy_(
            saved_temporal.to(
                device=live_temporal.device, dtype=live_temporal.dtype),
            non_blocking=True)

    def direct_copy() -> None:
        live_conv.copy_(saved_conv, non_blocking=True)
        live_temporal.copy_(saved_temporal, non_blocking=True)

    temporary_gpu_copy()
    torch.cuda.synchronize(device)
    expected_conv = live_conv.cpu()
    expected_temporal = live_temporal.cpu()
    live_conv.zero_()
    live_temporal.zero_()
    direct_copy()
    torch.cuda.synchronize(device)
    exact = (
        torch.equal(live_conv.cpu(), expected_conv)
        and torch.equal(live_temporal.cpu(), expected_temporal))

    def measure(operation) -> float:
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(args.iterations):
            operation()
        torch.cuda.synchronize(device)
        return (time.perf_counter() - started) * 1000.0 / args.iterations

    for _ in range(args.warmup):
        temporary_gpu_copy()
        direct_copy()
    torch.cuda.synchronize(device)

    trials = {"temporary_gpu_copy": [], "direct_copy": []}
    for repeat in range(args.repeats):
        order = (
            (("temporary_gpu_copy", temporary_gpu_copy),
             ("direct_copy", direct_copy))
            if repeat % 2 == 0 else
            (("direct_copy", direct_copy),
             ("temporary_gpu_copy", temporary_gpu_copy)))
        for name, operation in order:
            trials[name].append(measure(operation))

    baseline_ms = statistics.median(trials["temporary_gpu_copy"])
    direct_ms = statistics.median(trials["direct_copy"])
    saving_ms = baseline_ms - direct_ms
    result = {
        "device": str(device),
        "dtype": "float32",
        "shapes": {
            "conv": list(saved_conv.shape),
            "temporal": list(saved_temporal.shape),
        },
        "state_bytes": saved_conv.nbytes + saved_temporal.nbytes,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "exact": exact,
        "temporary_gpu_copy_median_ms": baseline_ms,
        "direct_copy_median_ms": direct_ms,
        "absolute_saving_ms": saving_ms,
        "speedup": baseline_ms / direct_ms if direct_ms > 0 else None,
        "min_saving_ms": args.min_saving_ms,
        "qualified": exact and saving_ms >= args.min_saving_ms,
        "trials_ms": trials,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
