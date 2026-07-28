#!/usr/bin/env python3
"""Numerical and timing gate for the BI100 WMMA QK tile."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
from pathlib import Path

import torch


def load_extension(path: Path):
    name = path.name.split(".", 1)[0]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "qk"):
        raise RuntimeError("extension does not expose qk")
    return module


def measure(function, warmup: int, repeats: int) -> tuple[float, list[float]]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return statistics.median(values), values


def relative_l2(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(
        reference.float() - candidate.float())
    denominator = torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
    return float((numerator / denominator).item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tiles", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    if args.tiles <= 0 or args.warmup < 0 or args.repeats <= 0:
        parser.error("tiles/repeats must be positive and warmup nonnegative")

    device = torch.device(args.device)
    extension = load_extension(args.extension)
    scale = 1.0 / math.sqrt(256)
    cases = []
    for case_index, magnitude in enumerate((0.5, 1.0, 2.0)):
        generator = torch.Generator(device=device).manual_seed(
            args.seed + case_index)
        query = torch.randn(
            (args.tiles, 16, 256), generator=generator,
            device=device, dtype=torch.float16) * magnitude
        key = torch.randn(
            (args.tiles, 32, 256), generator=generator,
            device=device, dtype=torch.float16) * magnitude
        value = torch.randn(
            (args.tiles, 32, 256), generator=generator,
            device=device, dtype=torch.float16)

        reference_scores = torch.bmm(
            query.float(), key.float().transpose(1, 2))
        candidate_scores = extension.qk(query, key)
        reference_output = torch.bmm(
            torch.softmax(reference_scores * scale, dim=-1), value.float(),
        ).half()
        candidate_output = torch.bmm(
            torch.softmax(candidate_scores * scale, dim=-1), value.float(),
        ).half()
        score_difference = (reference_scores - candidate_scores).abs()
        output_difference = (
            reference_output.float() - candidate_output.float()).abs()
        cases.append({
            "magnitude": magnitude,
            "scores_finite": bool(torch.isfinite(candidate_scores).all()),
            "score_max_abs": float(score_difference.max().item()),
            "score_mean_abs": float(score_difference.mean().item()),
            "output_finite": bool(torch.isfinite(candidate_output).all()),
            "output_max_abs": float(output_difference.max().item()),
            "output_mean_abs": float(output_difference.mean().item()),
            "output_relative_l2": relative_l2(
                reference_output, candidate_output),
        })

    generator = torch.Generator(device=device).manual_seed(args.seed + 100)
    query = torch.randn(
        (args.tiles, 16, 256), generator=generator,
        device=device, dtype=torch.float16)
    key = torch.randn(
        (args.tiles, 32, 256), generator=generator,
        device=device, dtype=torch.float16)
    reference_ms, reference_trials = measure(
        lambda: torch.bmm(query.float(), key.float().transpose(1, 2)),
        args.warmup, args.repeats)
    candidate_ms, candidate_trials = measure(
        lambda: extension.qk(query, key), args.warmup, args.repeats)

    numerical_ok = all(
        case["scores_finite"] and case["output_finite"]
        and case["output_max_abs"] <= 1e-3
        and case["output_relative_l2"] <= 1e-5
        for case in cases)
    speedup = reference_ms / candidate_ms
    report = {
        "experiment": "M1-28-corex-wmma-qk-capability",
        "configuration": {
            "tiles": args.tiles,
            "shape": [16, 32, 256],
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "cases": cases,
        "timing": {
            "reference_fp32_bmm_median_ms": reference_ms,
            "candidate_wmma_median_ms": candidate_ms,
            "speedup": speedup,
            "reference_trials_ms": reference_trials,
            "candidate_trials_ms": candidate_trials,
        },
        "gate": {
            "numerical_ok": numerical_ok,
            "speed_ok": speedup >= 1.5,
            "continuation": numerical_ok and speedup >= 1.5,
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["gate"]["continuation"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
