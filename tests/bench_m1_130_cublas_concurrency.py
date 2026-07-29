#!/usr/bin/env python3
"""Fixed-shape BI100 probe for independent FP32 QK/PV stream overlap."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Callable


SCHEMA = "bi100-m1-130-cublas-concurrency-cell-v1"
HEADS = 4
HEAD_DIM = 256
KEY_TOKENS = 512
SEED = 20260729
WARMUPS = 5
TRIALS = 20
RELATIVE_L2_LIMIT = 1e-7
MAX_ABS_LIMIT = 1e-5
MIN_CELL_SPEEDUP = 1.05
CASES = {
    "q8176": 8176,
    "q5616": 5616,
}


def _error_metrics(actual: Any, expected: Any) -> dict[str, float]:
    actual_cpu = actual.detach().cpu().double()
    expected_cpu = expected.detach().cpu().double()
    difference_tensor = actual_cpu - expected_cpu
    difference = difference_tensor.norm().item()
    denominator = expected_cpu.norm().item()
    if denominator == 0:
        relative_l2 = 0.0 if difference == 0 else math.inf
    else:
        relative_l2 = difference / denominator
    return {
        "relative_l2": relative_l2,
        "max_abs": float(difference_tensor.abs().max().item()),
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summarize(values: list[float]) -> dict[str, float]:
    return {
        "minimum_ms": min(values),
        "p10_ms": _percentile(values, 0.10),
        "median_ms": statistics.median(values),
        "p90_ms": _percentile(values, 0.90),
        "maximum_ms": max(values),
    }


def _measure(function: Callable[[], None], torch: Any) -> list[float]:
    for _ in range(WARMUPS):
        function()
        torch.cuda.synchronize()
    values = []
    for _ in range(TRIALS):
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        function()
        torch.cuda.synchronize()
        elapsed_ns = time.perf_counter_ns() - started
        values.append(elapsed_ns / 1_000_000.0)
    return values


def _valid_metric(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CoreX GPU is required")
    if hasattr(torch.backends, "cuda") and hasattr(
            torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(1)

    query_tokens = CASES[args.case]
    generator = torch.Generator(device="cuda")
    generator.manual_seed(SEED)

    qk_left = torch.randn(
        (HEADS, KEY_TOKENS, HEAD_DIM),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    qk_right = torch.randn(
        (HEADS, HEAD_DIM, query_tokens),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    pv_left = torch.randn(
        (HEADS, HEAD_DIM, KEY_TOKENS),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    pv_right = torch.randn(
        (HEADS, KEY_TOKENS, query_tokens),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    sequential_qk = torch.empty(
        (HEADS, KEY_TOKENS, query_tokens),
        dtype=torch.float32,
        device="cuda",
    )
    sequential_pv = torch.empty(
        (HEADS, HEAD_DIM, query_tokens),
        dtype=torch.float32,
        device="cuda",
    )
    concurrent_qk = torch.empty_like(sequential_qk)
    concurrent_pv = torch.empty_like(sequential_pv)

    qk_stream = torch.cuda.Stream()
    pv_stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    def qk_only() -> None:
        torch.bmm(qk_left, qk_right, out=sequential_qk)

    def pv_only() -> None:
        torch.bmm(pv_left, pv_right, out=sequential_pv)

    def sequential() -> None:
        torch.bmm(qk_left, qk_right, out=sequential_qk)
        torch.bmm(pv_left, pv_right, out=sequential_pv)

    def concurrent() -> None:
        with torch.cuda.stream(qk_stream):
            torch.bmm(qk_left, qk_right, out=concurrent_qk)
        with torch.cuda.stream(pv_stream):
            torch.bmm(pv_left, pv_right, out=concurrent_pv)

    qk_trials = _measure(qk_only, torch)
    pv_trials = _measure(pv_only, torch)
    if args.visible_physical_gpu % 2 == 0:
        sequential_trials = _measure(sequential, torch)
        concurrent_trials = _measure(concurrent, torch)
    else:
        concurrent_trials = _measure(concurrent, torch)
        sequential_trials = _measure(sequential, torch)

    sequential()
    concurrent()
    torch.cuda.synchronize()
    qk_error = _error_metrics(concurrent_qk, sequential_qk)
    pv_error = _error_metrics(concurrent_pv, sequential_pv)
    finite = bool(
        torch.isfinite(sequential_qk).all().item()
        and torch.isfinite(sequential_pv).all().item()
        and torch.isfinite(concurrent_qk).all().item()
        and torch.isfinite(concurrent_pv).all().item()
    )

    qk_summary = _summarize(qk_trials)
    pv_summary = _summarize(pv_trials)
    sequential_summary = _summarize(sequential_trials)
    concurrent_summary = _summarize(concurrent_trials)
    sequential_ms = sequential_summary["median_ms"]
    concurrent_ms = concurrent_summary["median_ms"]
    speedup = sequential_ms / concurrent_ms
    independent_sum_ms = (
        qk_summary["median_ms"] + pv_summary["median_ms"])
    ideal_parallel_ms = max(
        qk_summary["median_ms"], pv_summary["median_ms"])
    overlap_opportunity_ms = sequential_ms - ideal_parallel_ms
    overlap_efficiency = (
        (sequential_ms - concurrent_ms) / overlap_opportunity_ms
        if overlap_opportunity_ms > 0
        else 0.0
    )

    reasons = []
    if not finite:
        reasons.append("one or more outputs are not finite")
    for label, metrics in (("qk", qk_error), ("pv", pv_error)):
        if (
            not _valid_metric(metrics["relative_l2"])
            or metrics["relative_l2"] > RELATIVE_L2_LIMIT
        ):
            reasons.append(
                f"{label} relative-L2 exceeds {RELATIVE_L2_LIMIT}")
        if (
            not _valid_metric(metrics["max_abs"])
            or metrics["max_abs"] > MAX_ABS_LIMIT
        ):
            reasons.append(
                f"{label} maximum absolute error exceeds {MAX_ABS_LIMIT}")
    if not _valid_metric(speedup) or speedup < MIN_CELL_SPEEDUP:
        reasons.append(
            f"concurrent speedup {speedup:.6f} is below "
            f"{MIN_CELL_SPEEDUP:.2f}x")

    qualified = not reasons
    return {
        "schema": SCHEMA,
        "version": 1,
        "qualified": qualified,
        "source_revision": args.source_revision,
        "instance": args.instance,
        "visible_physical_gpu": args.visible_physical_gpu,
        "device_name": torch.cuda.get_device_name(0),
        "case": args.case,
        "shape": {
            "heads": HEADS,
            "query_tokens": query_tokens,
            "key_tokens": KEY_TOKENS,
            "head_dim": HEAD_DIM,
            "dtype": "float32",
        },
        "seed": SEED,
        "numerical": {
            "finite": finite,
            "qk_concurrent_vs_sequential": qk_error,
            "pv_concurrent_vs_sequential": pv_error,
        },
        "timing": {
            "warmups": WARMUPS,
            "trials": TRIALS,
            "qk_only": qk_summary,
            "pv_only": pv_summary,
            "sequential": sequential_summary,
            "concurrent": concurrent_summary,
            "independent_sum_ms": independent_sum_ms,
            "ideal_parallel_ms": ideal_parallel_ms,
            "sequential_over_concurrent_speedup": speedup,
            "overlap_efficiency": overlap_efficiency,
        },
        "thresholds": {
            "relative_l2": RELATIVE_L2_LIMIT,
            "max_abs": MAX_ABS_LIMIT,
            "minimum_cell_speedup": MIN_CELL_SPEEDUP,
        },
        "reasons": reasons,
        "decision": {
            "double_buffer_pipeline_authorized": False,
            "tp4_service_authorized": False,
            "main_or_yaml_change_authorized": False,
        },
        "privacy": {
            "contains_raw_tensors": False,
            "contains_model_outputs": False,
            "contains_credentials": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=tuple(CASES))
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--visible-physical-gpu", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "case": report["case"],
        "qualified": report["qualified"],
        "speedup": report["timing"][
            "sequential_over_concurrent_speedup"],
        "reasons": report["reasons"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
