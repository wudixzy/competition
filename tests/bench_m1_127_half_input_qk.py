#!/usr/bin/env python3
"""Fixed production-shape capability gate for half-input QK GemmEx."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
from typing import Any, Callable


SCHEMA = "bi100-m1-127-half-input-qk-capability-v1"
MODULE_NAME = "corex_half_input_qk_gemm"
HEADS = 4
HEAD_DIM = 256
KEY_TOKENS = 512
SCALE = HEAD_DIM ** -0.5
SEED = 20260729
MAGNITUDES = (0.5, 1.0, 2.0)
WARMUPS = 5
TRIALS = 20
SAMPLED_QUERIES = 16
RELATIVE_L2_LIMIT = 1e-5
MAX_ABS_LIMIT = 1e-3
MIN_SPEEDUP = 1.25
CASES = {
    "q8176": 8176,
    "q5616": 5616,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_l2(actual: Any, expected: Any) -> float:
    difference = (actual.double() - expected.double()).norm().item()
    denominator = expected.double().norm().item()
    if denominator == 0:
        return 0.0 if difference == 0 else math.inf
    return difference / denominator


def _load_extension(path: Path, expected_sha256: str) -> tuple[Any, dict[str, Any]]:
    resolved = path.resolve(strict=True)
    actual_sha256 = _sha256(resolved)
    if actual_sha256 != expected_sha256:
        raise RuntimeError("extension SHA-256 differs")
    spec = importlib.util.spec_from_file_location(MODULE_NAME, resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot create extension module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("qk_sgemm", "qk_half_input"):
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"extension is missing {name}")
    return module, {
        "size_bytes": resolved.stat().st_size,
        "sha256": actual_sha256,
    }


def _measure(function: Callable[[], Any]) -> tuple[list[float], Any]:
    import torch

    for _ in range(WARMUPS):
        value = function()
        torch.cuda.synchronize()
        del value
    trials = []
    result = None
    for _ in range(TRIALS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function()
        end.record()
        torch.cuda.synchronize()
        trials.append(float(start.elapsed_time(end)))
    assert result is not None
    return trials, result


def _oracle_metrics(query: Any, key: Any, output: Any) -> dict[str, float]:
    import torch

    indices = torch.linspace(
        0, query.shape[1] - 1, SAMPLED_QUERIES,
        dtype=torch.int64, device=query.device)
    sampled_query = query.index_select(1, indices).cpu().double()
    key_cpu = key.cpu().double()
    oracle = torch.matmul(sampled_query * SCALE, key_cpu.transpose(0, 1))
    sampled_output = output.index_select(1, indices).cpu().double()
    return {
        "relative_l2": _relative_l2(sampled_output, oracle),
        "max_abs": float((sampled_output - oracle).abs().max().item()),
    }


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
    query_tokens = CASES[args.case]
    extension, artifact = _load_extension(
        args.extension, args.expected_extension_sha256)
    numerical = []
    timing_inputs = None
    for index, magnitude in enumerate(MAGNITUDES):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(SEED + index)
        query = (
            torch.randn(
                (HEADS, query_tokens, HEAD_DIM),
                dtype=torch.float16,
                device="cuda",
                generator=generator,
            )
            * magnitude
        ).contiguous()
        key = (
            torch.randn(
                (KEY_TOKENS, HEAD_DIM),
                dtype=torch.float16,
                device="cuda",
                generator=generator,
            )
            * magnitude
        ).contiguous()
        scaled_query = query.float().mul(SCALE).contiguous()
        key_float = key.float().contiguous()
        control = extension.qk_sgemm(scaled_query, key_float)
        candidate = extension.qk_half_input(query, key, SCALE)
        torch.cuda.synchronize()
        candidate_vs_control_l2 = _relative_l2(candidate, control)
        candidate_vs_control_max_abs = float(
            (candidate - control).abs().max().item())
        control_oracle = _oracle_metrics(query, key, control)
        candidate_oracle = _oracle_metrics(query, key, candidate)
        numerical.append({
            "magnitude": magnitude,
            "finite": bool(
                torch.isfinite(control).all().item()
                and torch.isfinite(candidate).all().item()),
            "candidate_vs_control_relative_l2": candidate_vs_control_l2,
            "candidate_vs_control_max_abs": candidate_vs_control_max_abs,
            "control_vs_fp64_sample": control_oracle,
            "candidate_vs_fp64_sample": candidate_oracle,
        })
        if magnitude == 1.0:
            timing_inputs = (query, key, scaled_query, key_float)
        del control, candidate

    assert timing_inputs is not None
    query, key, scaled_query, key_float = timing_inputs
    control_call = lambda: extension.qk_sgemm(scaled_query, key_float)
    candidate_call = lambda: extension.qk_half_input(query, key, SCALE)
    if args.visible_physical_gpu % 2 == 0:
        control_trials, control_output = _measure(control_call)
        candidate_trials, candidate_output = _measure(candidate_call)
    else:
        candidate_trials, candidate_output = _measure(candidate_call)
        control_trials, control_output = _measure(control_call)
    timing_l2 = _relative_l2(candidate_output, control_output)
    control_median = statistics.median(control_trials)
    candidate_median = statistics.median(candidate_trials)
    speedup = control_median / candidate_median

    reasons = []
    for row in numerical:
        label = f"magnitude={row['magnitude']}"
        if row["finite"] is not True:
            reasons.append(f"{label}: output is not finite")
        for field, limit in (
            ("candidate_vs_control_relative_l2", RELATIVE_L2_LIMIT),
            ("candidate_vs_control_max_abs", MAX_ABS_LIMIT),
        ):
            value = row[field]
            if not _valid_metric(value) or value > limit:
                reasons.append(f"{label}: {field} exceeds {limit}")
        candidate_oracle = row["candidate_vs_fp64_sample"]
        if (
            not _valid_metric(candidate_oracle["relative_l2"])
            or candidate_oracle["relative_l2"] > RELATIVE_L2_LIMIT
            or not _valid_metric(candidate_oracle["max_abs"])
            or candidate_oracle["max_abs"] > MAX_ABS_LIMIT
        ):
            reasons.append(f"{label}: sampled FP64 oracle gate failed")
    if not _valid_metric(timing_l2) or timing_l2 > RELATIVE_L2_LIMIT:
        reasons.append("timed output relative L2 gate failed")
    if not _valid_metric(speedup) or speedup < MIN_SPEEDUP:
        reasons.append(f"QK speedup {speedup:.6f} is below {MIN_SPEEDUP:.2f}x")

    return {
        "schema": SCHEMA,
        "version": 1,
        "qualified": not reasons,
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
        },
        "seed": SEED,
        "magnitudes": list(MAGNITUDES),
        "extension": artifact,
        "numerical": numerical,
        "timing": {
            "warmups": WARMUPS,
            "trials": TRIALS,
            "control_ms": control_median,
            "candidate_ms": candidate_median,
            "control_over_candidate_speedup": speedup,
            "timed_candidate_vs_control_relative_l2": timing_l2,
        },
        "thresholds": {
            "relative_l2": RELATIVE_L2_LIMIT,
            "max_abs": MAX_ABS_LIMIT,
            "minimum_qk_speedup": MIN_SPEEDUP,
        },
        "reasons": reasons,
        "decision": {
            "full_pipeline_integration_authorized": not reasons,
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
    parser.add_argument("--extension", required=True, type=Path)
    parser.add_argument("--expected-extension-sha256", required=True)
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
        "speedup": report["timing"]["control_over_candidate_speedup"],
        "reasons": report["reasons"],
    }, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
