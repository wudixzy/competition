#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import time
from pathlib import Path
from typing import Callable

import torch


FIXED_SEEDS = (20260715, 20260727)
RANDOM_SCALES = (0.001, 0.05, 0.5, 2.0)


def load_extension(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def l2norm(value: torch.Tensor) -> torch.Tensor:
    return value * torch.rsqrt(
        (value * value).sum(dim=-1, keepdim=True) + 1.0e-6)


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    delta = actual.float() - expected.float()
    squared_error = float(delta.square().sum())
    squared_reference = float(expected.float().square().sum())
    return {
        "exact": bool(torch.equal(actual, expected)),
        "finite": bool(torch.isfinite(actual).all()),
        "max_abs": float(delta.abs().max()),
        "relative_l2": math.sqrt(
            squared_error / max(squared_reference, 1.0e-30)),
    }


def measure(case: Callable[[], torch.Tensor], warmup: int, iterations: int,
            repeats: int) -> dict:
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
    ordered = sorted(trials)
    return {
        "median_ms": statistics.median(trials),
        "p10_ms": ordered[max(0, int(0.1 * (len(ordered) - 1)))],
        "p90_ms": ordered[min(
            len(ordered) - 1, int(0.9 * (len(ordered) - 1)))],
        "trials_ms": trials,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qk-map-extension", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--sequence-steps", type=int, default=500)
    parser.add_argument("--sequence-seed", type=int, default=20260729)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    qk_map = load_extension("corex_gdn_qk_map", args.qk_map_extension)

    batch = 1
    key_heads = 4
    value_heads = 8
    head_dim = 128

    def reference(raw_qk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query, key = torch.split(raw_qk, key_heads, dim=1)
        normalized_query = l2norm(query)
        normalized_key = l2norm(key)
        mapped = qk_map.qk_map(
            normalized_query, normalized_key, value_heads)
        return torch.stack((normalized_query, normalized_key)), mapped

    def candidate(raw_qk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = l2norm(raw_qk)
        normalized_query, normalized_key = torch.split(
            normalized, key_heads, dim=1)
        if not normalized_query.is_contiguous() or not normalized_key.is_contiguous():
            raise RuntimeError("combined q/k slices must remain contiguous")
        mapped = qk_map.qk_map(
            normalized_query, normalized_key, value_heads)
        return torch.stack((normalized_query, normalized_key)), mapped

    fixed = {}
    timing_raw = None
    for seed in FIXED_SEEDS:
        generator = torch.Generator(device=device).manual_seed(seed)
        raw = torch.randn(
            (batch, key_heads * 2, head_dim),
            device=device,
            dtype=torch.float16,
            generator=generator,
        ) * 0.05
        reference_norm, reference_mapped = reference(raw)
        candidate_norm, candidate_mapped = candidate(raw)
        fixed[str(seed)] = {
            "normalized": compare(candidate_norm, reference_norm),
            "mapped": compare(candidate_mapped, reference_mapped),
        }
        timing_raw = raw

    assert timing_raw is not None

    def reference_timed() -> torch.Tensor:
        return reference(timing_raw)[1]

    def candidate_timed() -> torch.Tensor:
        return candidate(timing_raw)[1]

    timings = {
        "reference": measure(
            reference_timed, args.warmup, args.iterations, args.repeats),
        "candidate": measure(
            candidate_timed, args.warmup, args.iterations, args.repeats),
    }
    reference_ms = float(timings["reference"]["median_ms"])
    candidate_ms = float(timings["candidate"]["median_ms"])
    timings["candidate"]["speedup_vs_reference"] = (
        reference_ms / candidate_ms)
    timings["candidate"]["saving_ms"] = reference_ms - candidate_ms
    timings["candidate"]["projected_30_layer_saving_ms"] = (
        (reference_ms - candidate_ms) * 30)

    generator = torch.Generator(
        device=device).manual_seed(args.sequence_seed)
    normalized_error = 0.0
    normalized_reference = 0.0
    mapped_error = 0.0
    mapped_reference = 0.0
    normalized_exact_steps = 0
    mapped_exact_steps = 0
    finite_steps = 0
    max_normalized_abs = 0.0
    max_mapped_abs = 0.0
    max_normalized_relative_l2 = 0.0
    max_mapped_relative_l2 = 0.0
    for step in range(args.sequence_steps):
        scale = RANDOM_SCALES[step % len(RANDOM_SCALES)]
        raw = torch.randn(
            (batch, key_heads * 2, head_dim),
            device=device,
            dtype=torch.float16,
            generator=generator,
        ) * scale
        reference_norm, reference_mapped = reference(raw)
        candidate_norm, candidate_mapped = candidate(raw)
        norm_delta = candidate_norm.float() - reference_norm.float()
        map_delta = candidate_mapped.float() - reference_mapped.float()
        norm_error = float(norm_delta.square().sum())
        norm_reference = float(reference_norm.float().square().sum())
        map_error = float(map_delta.square().sum())
        map_reference = float(reference_mapped.float().square().sum())
        normalized_error += norm_error
        normalized_reference += norm_reference
        mapped_error += map_error
        mapped_reference += map_reference
        normalized_exact_steps += int(torch.equal(
            candidate_norm, reference_norm))
        mapped_exact_steps += int(torch.equal(
            candidate_mapped, reference_mapped))
        finite_steps += int(
            torch.isfinite(candidate_norm).all()
            and torch.isfinite(candidate_mapped).all())
        max_normalized_abs = max(
            max_normalized_abs, float(norm_delta.abs().max()))
        max_mapped_abs = max(
            max_mapped_abs, float(map_delta.abs().max()))
        max_normalized_relative_l2 = max(
            max_normalized_relative_l2,
            math.sqrt(norm_error / max(norm_reference, 1.0e-30)),
        )
        max_mapped_relative_l2 = max(
            max_mapped_relative_l2,
            math.sqrt(map_error / max(map_reference, 1.0e-30)),
        )

    report = {
        "schema": "bi100-gdn-combined-qk-norm-v1",
        "shape": {
            "batch": batch,
            "key_heads": key_heads,
            "value_heads": value_heads,
            "head_dim": head_dim,
            "dtype": str(torch.float16),
        },
        "config": {
            "device": args.device,
            "fixed_seeds": list(FIXED_SEEDS),
            "random_scales": list(RANDOM_SCALES),
            "sequence_seed": args.sequence_seed,
            "sequence_steps": args.sequence_steps,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
        },
        "fixed": fixed,
        "sequence": {
            "steps": args.sequence_steps,
            "finite_steps": finite_steps,
            "normalized_exact_steps": normalized_exact_steps,
            "mapped_exact_steps": mapped_exact_steps,
            "normalized_relative_l2": math.sqrt(
                normalized_error / max(normalized_reference, 1.0e-30)),
            "mapped_relative_l2": math.sqrt(
                mapped_error / max(mapped_reference, 1.0e-30)),
            "max_normalized_relative_l2": max_normalized_relative_l2,
            "max_mapped_relative_l2": max_mapped_relative_l2,
            "max_normalized_abs": max_normalized_abs,
            "max_mapped_abs": max_mapped_abs,
        },
        "timings": timings,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
