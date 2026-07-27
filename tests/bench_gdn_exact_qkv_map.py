#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Callable

import torch


FIXED_SEEDS = (20260715, 20260727)
RANDOM_SCALES = (0.001, 0.05, 0.5, 2.0)


def load_extension(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "corex_gdn_qkv_map", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def l2norm(value: torch.Tensor) -> torch.Tensor:
    return value * torch.rsqrt(
        (value * value).sum(dim=-1, keepdim=True) + 1.0e-6)


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
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


def _time_block(case: Callable[[], Any], iterations: int) -> float:
    started = time.perf_counter()
    for _ in range(iterations):
        case()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0 / iterations


def summarize(trials: list[float]) -> dict[str, Any]:
    ordered = sorted(trials)
    return {
        "median_ms": statistics.median(trials),
        "p10_ms": ordered[max(0, int(0.1 * (len(ordered) - 1)))],
        "p90_ms": ordered[min(
            len(ordered) - 1, int(0.9 * (len(ordered) - 1)))],
        "trials_ms": trials,
    }


def measure(case: Callable[[], Any], warmup: int, iterations: int,
            repeats: int) -> dict[str, Any]:
    for _ in range(warmup):
        case()
    torch.cuda.synchronize()
    return summarize([
        _time_block(case, iterations) for _ in range(repeats)
    ])


def measure_pair(
    reference: Callable[[], Any],
    candidate: Callable[[], Any],
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        reference()
        candidate()
    torch.cuda.synchronize()
    reference_trials: list[float] = []
    candidate_trials: list[float] = []
    for repeat in range(repeats):
        if repeat % 2 == 0:
            reference_trials.append(_time_block(reference, iterations))
            candidate_trials.append(_time_block(candidate, iterations))
        else:
            candidate_trials.append(_time_block(candidate, iterations))
            reference_trials.append(_time_block(reference, iterations))
    paired_speedups = [
        reference_ms / candidate_ms
        for reference_ms, candidate_ms
        in zip(reference_trials, candidate_trials, strict=True)
    ]
    return {
        "reference": summarize(reference_trials),
        "candidate": summarize(candidate_trials),
        "paired_speedups": paired_speedups,
        "paired_speedup_median": statistics.median(paired_speedups),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--sequence-steps", type=int, default=1000)
    parser.add_argument("--sequence-seed", type=int, default=20260731)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1 or args.repeats < 3:
        parser.error("warmup and iterations must be positive; repeats >= 3")
    if args.sequence_steps < 1:
        parser.error("sequence-steps must be positive")

    extension_path = args.extension.resolve(strict=True)
    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    extension = load_extension(extension_path)

    batch = 1
    key_heads = 4
    value_heads = 8
    head_dim = 128

    def prepare(
        seed: int,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device=device).manual_seed(seed)
        raw_qk = torch.randn(
            (batch, 2 * key_heads, head_dim),
            device=device,
            dtype=torch.float16,
            generator=generator,
        ) * scale
        value = torch.randn(
            (batch, value_heads, head_dim),
            device=device,
            dtype=torch.float16,
            generator=generator,
        ) * scale
        normalized = l2norm(raw_qk)
        query, key = torch.split(normalized, key_heads, dim=1)
        if not query.is_contiguous() or not key.is_contiguous():
            raise RuntimeError("normalized q/k slices must remain contiguous")
        return query, key, value

    def reference(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return extension.qk_map(query, key, value_heads), value.float()

    def candidate(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        return extension.qkv_map(query, key, value)

    fixed: dict[str, Any] = {}
    timing_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    for seed in FIXED_SEEDS:
        query, key, value = prepare(seed, 0.05)
        mapped, value_fp32 = reference(query, key, value)
        combined = candidate(query, key, value)
        fixed[str(seed)] = {
            "mapped_qk": compare(combined[:2], mapped),
            "value": compare(combined[2], value_fp32),
            "complete": compare(
                combined, torch.cat((mapped, value_fp32.unsqueeze(0)), dim=0)),
        }
        timing_inputs = query, key, value

    assert timing_inputs is not None
    timing_query, timing_key, timing_value = timing_inputs

    def timed_qk_only() -> torch.Tensor:
        return extension.qk_map(timing_query, timing_key, value_heads)

    def timed_value_only() -> torch.Tensor:
        return timing_value.float()

    def timed_reference() -> tuple[torch.Tensor, torch.Tensor]:
        return reference(timing_query, timing_key, timing_value)

    def timed_candidate() -> torch.Tensor:
        return candidate(timing_query, timing_key, timing_value)

    pair = measure_pair(
        timed_reference,
        timed_candidate,
        args.warmup,
        args.iterations,
        args.repeats,
    )
    reference_ms = float(pair["reference"]["median_ms"])
    candidate_ms = float(pair["candidate"]["median_ms"])
    saving_ms = reference_ms - candidate_ms
    timings = {
        "qk_map_only": measure(
            timed_qk_only, args.warmup, args.iterations, args.repeats),
        "value_cast_only": measure(
            timed_value_only, args.warmup, args.iterations, args.repeats),
        "pair": pair,
        "candidate_speedup": reference_ms / candidate_ms,
        "candidate_saving_ms": saving_ms,
        "projected_30_layer_saving_ms": 30.0 * saving_ms,
    }

    total_squared_error = 0.0
    total_squared_reference = 0.0
    exact_steps = 0
    finite_steps = 0
    max_abs = 0.0
    max_relative_l2 = 0.0
    generator = torch.Generator(
        device=device).manual_seed(args.sequence_seed)
    for step in range(args.sequence_steps):
        scale = RANDOM_SCALES[step % len(RANDOM_SCALES)]
        raw_qk = torch.randn(
            (batch, 2 * key_heads, head_dim),
            device=device,
            dtype=torch.float16,
            generator=generator,
        ) * scale
        value = torch.randn(
            (batch, value_heads, head_dim),
            device=device,
            dtype=torch.float16,
            generator=generator,
        ) * scale
        normalized = l2norm(raw_qk)
        query, key = torch.split(normalized, key_heads, dim=1)
        mapped, value_fp32 = reference(query, key, value)
        expected = torch.cat((mapped, value_fp32.unsqueeze(0)), dim=0)
        actual = candidate(query, key, value)
        delta = actual.float() - expected.float()
        squared_error = float(delta.square().sum())
        squared_reference = float(expected.float().square().sum())
        total_squared_error += squared_error
        total_squared_reference += squared_reference
        relative_l2 = math.sqrt(
            squared_error / max(squared_reference, 1.0e-30))
        exact_steps += int(torch.equal(actual, expected))
        finite_steps += int(torch.isfinite(actual).all())
        max_abs = max(max_abs, float(delta.abs().max()))
        max_relative_l2 = max(max_relative_l2, relative_l2)

    report = {
        "schema": "bi100-gdn-exact-qkv-map-v1",
        "version": 1,
        "shape": {
            "batch": batch,
            "key_heads": key_heads,
            "value_heads": value_heads,
            "head_dim": head_dim,
            "dtype": str(torch.float16),
        },
        "artifact": {
            "path": str(extension_path),
            "sha256": sha256(extension_path),
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
            "exact_steps": exact_steps,
            "finite_steps": finite_steps,
            "relative_l2": math.sqrt(
                total_squared_error
                / max(total_squared_reference, 1.0e-30)),
            "max_relative_l2": max_relative_l2,
            "max_abs": max_abs,
        },
        "timings": timings,
        "production_integration_attempted": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
