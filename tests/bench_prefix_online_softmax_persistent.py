#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from pathlib import Path
from typing import Callable

import torch


TILE_SIZE = 512
QUERY_HEADS = 6


def load_extension(path: Path):
    spec = importlib.util.spec_from_file_location(
        "corex_prefix_online_softmax_persistent", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension from {path}")
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


def reference_update(
    scores: torch.Tensor,
    running_max: torch.Tensor,
    running_sum: torch.Tensor,
) -> torch.Tensor:
    block_max = scores.amax(dim=-1)
    new_max = torch.maximum(running_max, block_max)
    scores.sub_(new_max.unsqueeze(-1)).exp_()
    correction = torch.exp(running_max - new_max)
    running_max.copy_(new_max)
    running_sum.mul_(correction).add_(scores.sum(dim=-1))
    return correction


def make_inputs(
    query_len: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = QUERY_HEADS * query_len
    generator = torch.Generator(device=device).manual_seed(seed + query_len)
    scores = torch.randn(
        (rows, TILE_SIZE), device=device, dtype=torch.float32,
        generator=generator) * 0.25
    running_max = torch.randn(
        (rows,), device=device, dtype=torch.float32,
        generator=generator) * 0.05
    running_sum = torch.rand(
        (rows,), device=device, dtype=torch.float32,
        generator=generator) * 128.0 + 1.0
    midpoint = rows // 2
    running_max[:midpoint] = float("-inf")
    running_sum[:midpoint] = 0.0
    return scores, running_max, running_sum


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    delta = (reference - candidate).abs()
    denominator = torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
    return {
        "max_abs": float(delta.max().item()),
        "relative_l2": float(
            (torch.linalg.vector_norm(delta.float()) / denominator).item()),
    }


def measure(
    operation: Callable,
    base: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    warmup: int,
    repeats: int,
) -> list[float]:
    trials = []
    for trial in range(warmup + repeats):
        inputs = tuple(tensor.clone() for tensor in base)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation(*inputs)
        end.record()
        end.synchronize()
        if trial >= warmup:
            trials.append(float(start.elapsed_time(end)))
    return trials


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-lengths", type=int, nargs="+",
                        default=[456, 8192])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--speedup-gate", type=float, default=1.5)
    parser.add_argument("--max-abs-gate", type=float, default=1e-3)
    parser.add_argument("--relative-l2-gate", type=float, default=1e-5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("warmup must be nonnegative and repeats must be positive")
    if not args.query_lengths or min(args.query_lengths) <= 0:
        parser.error("query lengths must be positive")

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    extension = load_extension(args.extension)
    cases = {}

    for query_len in args.query_lengths:
        base = make_inputs(query_len, device, args.seed)
        reference_inputs = tuple(tensor.clone() for tensor in base)
        candidate_inputs = tuple(tensor.clone() for tensor in base)
        reference_correction = reference_update(*reference_inputs)
        candidate_correction = extension.update(*candidate_inputs)
        torch.cuda.synchronize()

        parity = {
            "finite": all(bool(torch.isfinite(tensor).all()) for tensor in (
                reference_inputs[0], candidate_inputs[0],
                reference_inputs[1], candidate_inputs[1],
                reference_inputs[2], candidate_inputs[2],
                reference_correction, candidate_correction,
            )),
            "exp_scores": compare(reference_inputs[0], candidate_inputs[0]),
            "running_max": compare(reference_inputs[1], candidate_inputs[1]),
            "running_sum": compare(reference_inputs[2], candidate_inputs[2]),
            "correction": compare(reference_correction, candidate_correction),
        }
        baseline_trials = measure(
            reference_update, base, args.warmup, args.repeats)
        candidate_trials = measure(
            extension.update, base, args.warmup, args.repeats)
        baseline_ms = statistics.median(baseline_trials)
        candidate_ms = statistics.median(candidate_trials)
        cases[str(query_len)] = {
            "rows": QUERY_HEADS * query_len,
            "parity": parity,
            "baseline": {
                "median_ms": baseline_ms,
                "p10_ms": percentile(baseline_trials, 10),
                "p90_ms": percentile(baseline_trials, 90),
                "trials_ms": baseline_trials,
            },
            "candidate": {
                "median_ms": candidate_ms,
                "p10_ms": percentile(candidate_trials, 10),
                "p90_ms": percentile(candidate_trials, 90),
                "trials_ms": candidate_trials,
            },
            "speedup": baseline_ms / candidate_ms,
        }

    parity_ok = all(
        case["parity"]["finite"]
        and all(
            metric["max_abs"] <= args.max_abs_gate
            and metric["relative_l2"] <= args.relative_l2_gate
            for name, metric in case["parity"].items()
            if name != "finite")
        for case in cases.values())
    speed_ok = all(
        case["speedup"] >= args.speedup_gate for case in cases.values())
    report = {
        "experiment": "M1-37-persistent-online-softmax-capability",
        "qualified": bool(parity_ok and speed_ok),
        "parity_ok": parity_ok,
        "speed_ok": speed_ok,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "config": {
            "tile_size": TILE_SIZE,
            "persistent_blocks": 1024,
            "threads": 256,
            "query_heads": QUERY_HEADS,
            "query_lengths": args.query_lengths,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "speedup_gate": args.speedup_gate,
            "max_abs_gate": args.max_abs_gate,
            "relative_l2_gate": args.relative_l2_gate,
        },
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
