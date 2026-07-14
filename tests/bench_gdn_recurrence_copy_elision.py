#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch


def recurrence(attn: torch.Tensor, *, clone_submatrix: bool) -> torch.Tensor:
    result = attn.clone()
    chunk_size = result.shape[-1]
    for index in range(1, chunk_size):
        row = result[..., index, :index].clone()
        submatrix = result[..., :index, :index]
        if clone_submatrix:
            submatrix = submatrix.clone()
        result[..., index, :index] = (
            row + (row.unsqueeze(-1) * submatrix).sum(-2))
    return result


def measure(fn, *, warmups: int, repeats: int) -> dict[str, object]:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
        del result
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def peak_bytes(fn) -> int:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    result = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    del result
    torch.cuda.empty_cache()
    return int(peak)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-label", required=True)
    parser.add_argument("--tokens", default="64,256,1024,8192,16384")
    parser.add_argument("--heads-per-rank", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    token_counts = [int(value) for value in args.tokens.split(",")]
    torch.manual_seed(20260714)
    device = torch.device("cuda:0")
    results = {}

    with torch.no_grad():
        for tokens in token_counts:
            chunks = (tokens + args.chunk_size - 1) // args.chunk_size
            matrix_batch = args.heads_per_rank * chunks
            source = torch.tril(
                torch.randn(
                    matrix_batch,
                    args.chunk_size,
                    args.chunk_size,
                    dtype=torch.float32,
                    device=device,
                ) * 0.01,
                diagonal=-1,
            )
            current_fn = lambda: recurrence(source, clone_submatrix=True)
            candidate_fn = lambda: recurrence(source, clone_submatrix=False)

            current = current_fn()
            candidate = candidate_fn()
            parity = {
                "bitwise_equal": bool(torch.equal(current, candidate)),
                "max_abs": float((current - candidate).abs().max().item()),
                "all_finite": bool(torch.isfinite(candidate).all().item()),
            }
            del current, candidate

            current_peak = peak_bytes(current_fn)
            candidate_peak = peak_bytes(candidate_fn)
            current_timing = measure(
                current_fn, warmups=args.warmups, repeats=args.repeats)
            candidate_timing = measure(
                candidate_fn, warmups=args.warmups, repeats=args.repeats)
            speedup = (
                current_timing["median_ms"] / candidate_timing["median_ms"])
            results[str(tokens)] = {
                "chunks": chunks,
                "matrix_batch": matrix_batch,
                "parity": parity,
                "current": {
                    **current_timing,
                    "peak_bytes": current_peak,
                },
                "candidate": {
                    **candidate_timing,
                    "peak_bytes": candidate_peak,
                    "speedup": speedup,
                    "peak_bytes_saved": current_peak - candidate_peak,
                },
            }
            del source
            torch.cuda.empty_cache()

    print(json.dumps({
        "device": args.device_label,
        "heads_per_rank": args.heads_per_rank,
        "chunk_size": args.chunk_size,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
