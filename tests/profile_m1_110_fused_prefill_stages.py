#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from bench_m1_55_production_prefill import (
    CASES,
    HEAD_DIM,
    _make_inputs,
    _measure,
    _relative_l2,
    reference_forward,
)


SCHEMA = "bi100-m1-110-fused-prefill-stage-profile-v1"
STAGE_NAMES = (
    "init",
    "gather",
    "qk",
    "mask",
    "softmax",
    "pv",
    "merge",
    "finalize",
)
PROFILE_TRIALS = 3
MAX_RELATIVE_L2 = 1e-5
MAX_PROFILE_EVENT_PERTURBATION = 0.15
MAX_REPRESENTATIVE_RUNTIME_DELTA = 0.05
MAX_PROFILE_ROW_CLOSURE_ABS_MS = 1e-6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_extension(
    path: Path,
    *,
    module_name: str,
    expected_sha256: str,
    require_profile: bool,
) -> tuple[Any, dict[str, Any]]:
    resolved = path.resolve(strict=True)
    digest = sha256_file(resolved)
    if digest != expected_sha256:
        raise RuntimeError(
            f"{module_name} SHA-256 mismatch: expected "
            f"{expected_sha256}, got {digest}")
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension from {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "forward", None)):
        raise RuntimeError(f"{module_name} does not expose forward")
    if require_profile and not callable(getattr(module, "profile", None)):
        raise RuntimeError(f"{module_name} does not expose profile")
    return module, {
        "path": str(resolved),
        "sha256": digest,
        "size_bytes": resolved.stat().st_size,
    }


def median_rows(rows: list[list[float]]) -> list[float]:
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("profile rows must be a non-empty rectangular matrix")
    return [
        statistics.median(row[index] for row in rows)
        for index in range(len(rows[0]))
    ]


def finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def profile_row_closure_residuals(
    rows: list[list[float]],
    stage_count: int,
) -> list[float]:
    if stage_count <= 0:
        raise ValueError("stage_count must be positive")
    if not rows or any(len(row) <= stage_count for row in rows):
        raise ValueError(
            "profile rows must include every stage and an event total")
    return [
        sum(row[:stage_count]) - row[stage_count]
        for row in rows
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "M1-110 requires exactly one visible CoreX GPU")
    production, production_artifact = load_extension(
        args.production_extension,
        module_name="corex_fused_paged_prefill",
        expected_sha256=args.expected_production_sha256,
        require_profile=False,
    )
    profiler, profiler_artifact = load_extension(
        args.profile_extension,
        module_name="corex_fused_paged_prefill_profile",
        expected_sha256=args.expected_profile_sha256,
        require_profile=True,
    )
    context_len, query_len, kind = CASES[args.case]
    if kind != "production":
        raise RuntimeError("M1-110 accepts production cases only")
    inputs = _make_inputs(context_len, query_len)
    scale = HEAD_DIM ** -0.5

    production_timing, production_result = _measure(
        lambda: tuple(production.forward(*inputs, context_len, scale)))
    representative_timing, representative_result = _measure(
        lambda: tuple(profiler.forward(*inputs, context_len, scale)))

    profile_rows = []
    profile_host_ms = []
    profiled_result = None
    for _ in range(PROFILE_TRIALS):
        started = time.perf_counter()
        values = tuple(profiler.profile(*inputs, context_len, scale))
        profile_host_ms.append((time.perf_counter() - started) * 1000.0)
        if len(values) != 3:
            raise RuntimeError("profile extension returned an invalid result")
        profiled_result = values[:2]
        profile_rows.append([float(value) for value in values[2].tolist()])
    assert profiled_result is not None

    expected_intervals = (
        2
        + 6 * (
            math.ceil(context_len / 2048)
            + math.ceil(query_len / 2048)
        )
    )
    row_width = len(STAGE_NAMES) + 3
    reasons = []
    if any(len(row) != row_width for row in profile_rows):
        reasons.append("profile timing row width is invalid")
        medians = [math.nan] * row_width
        row_closure_residuals = [math.nan] * len(profile_rows)
    else:
        medians = median_rows(profile_rows)
        row_closure_residuals = profile_row_closure_residuals(
            profile_rows, len(STAGE_NAMES))

    stage_ms = dict(zip(STAGE_NAMES, medians[:len(STAGE_NAMES)]))
    event_total_ms = medians[len(STAGE_NAMES)]
    validation_host_ms = medians[len(STAGE_NAMES) + 1]
    interval_count = medians[len(STAGE_NAMES) + 2]
    if (
        not finite_nonnegative(interval_count)
        or not math.isclose(interval_count, expected_intervals)
    ):
        reasons.append(
            f"profile interval count differs from {expected_intervals}")
    if not all(finite_nonnegative(value) for value in stage_ms.values()):
        reasons.append("one or more stage timings are invalid")
    if (
        not finite_nonnegative(event_total_ms)
        or event_total_ms <= 0.0
    ):
        reasons.append("event total timing is invalid")
    if (
        not all(math.isfinite(value) for value in row_closure_residuals)
        or any(
            abs(value) > MAX_PROFILE_ROW_CLOSURE_ABS_MS
            for value in row_closure_residuals
        )
    ):
        reasons.append("a profile trial does not close to its event total")
    if not finite_nonnegative(validation_host_ms):
        reasons.append("validation host timing is invalid")

    production_output, production_lse = production_result
    representative_output, representative_lse = representative_result
    profile_output, profile_lse = profiled_result
    exact_representative = bool(
        torch.equal(production_output, representative_output)
        and torch.equal(production_lse, representative_lse)
    )
    exact_profile = bool(
        torch.equal(representative_output, profile_output)
        and torch.equal(representative_lse, profile_lse)
    )
    if not exact_representative:
        reasons.append("profile build forward differs from production")
    if not exact_profile:
        reasons.append("event instrumentation changes output")

    reference_output, reference_lse = reference_forward(
        *inputs, context_len, scale)
    output_relative_l2 = _relative_l2(profile_output, reference_output)
    lse_relative_l2 = _relative_l2(profile_lse, reference_lse)
    finite = bool(
        torch.isfinite(profile_output).all().item()
        and torch.isfinite(profile_lse).all().item()
    )
    if (
        not finite
        or output_relative_l2 > MAX_RELATIVE_L2
        or lse_relative_l2 > MAX_RELATIVE_L2
    ):
        reasons.append("profiled output failed the numerical gate")

    production_ms = production_timing["cuda_median_ms"]
    representative_ms = representative_timing["cuda_median_ms"]
    representative_delta = representative_ms / production_ms - 1.0
    event_perturbation = event_total_ms / representative_ms - 1.0
    profile_host_median_ms = statistics.median(profile_host_ms)
    profile_host_perturbation = (
        profile_host_median_ms
        / representative_timing["host_median_ms"]
        - 1.0
    )
    if abs(representative_delta) > MAX_REPRESENTATIVE_RUNTIME_DELTA:
        reasons.append("profile build is not representative of production")
    if abs(event_perturbation) > MAX_PROFILE_EVENT_PERTURBATION:
        reasons.append("event instrumentation perturbation exceeds 15%")

    stage_median_sum_ms = sum(stage_ms.values())
    median_closure_residual_ms = stage_median_sum_ms - event_total_ms
    stage_share = {
        name: value / stage_median_sum_ms
        for name, value in stage_ms.items()
    } if stage_median_sum_ms > 0 else {}
    ranked_stages = sorted(
        stage_ms,
        key=lambda name: stage_ms[name],
        reverse=True,
    )
    return {
        "schema": SCHEMA,
        "source_commit": args.source_commit,
        "instance": args.instance,
        "visible_physical_gpu": args.visible_physical_gpu,
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "case": args.case,
        "context_len": context_len,
        "query_len": query_len,
        "total_kv_len": context_len + query_len,
        "profile_trials": PROFILE_TRIALS,
        "expected_intervals": expected_intervals,
        "production_extension": production_artifact,
        "profile_extension": profiler_artifact,
        "production_timing": production_timing,
        "profile_build_forward_timing": representative_timing,
        "profile_host_trials_ms": profile_host_ms,
        "profile_host_median_ms": profile_host_median_ms,
        "profile_rows_ms": profile_rows,
        "profile_row_closure_residuals_ms": row_closure_residuals,
        "stage_median_ms": stage_ms,
        "stage_median_sum_ms": stage_median_sum_ms,
        "stage_share": stage_share,
        "ranked_stages": ranked_stages,
        "event_total_median_ms": event_total_ms,
        "median_closure_residual_ms": median_closure_residual_ms,
        "validation_host_median_ms": validation_host_ms,
        "interval_count_median": interval_count,
        "representative_runtime_delta": representative_delta,
        "event_perturbation": event_perturbation,
        "profile_host_perturbation": profile_host_perturbation,
        "numerical": {
            "finite": finite,
            "production_profile_build_exact": exact_representative,
            "unprofiled_profiled_exact": exact_profile,
            "output_relative_l2": output_relative_l2,
            "lse_relative_l2": lse_relative_l2,
        },
        "thresholds": {
            "maximum_relative_l2": MAX_RELATIVE_L2,
            "maximum_profile_event_perturbation":
                MAX_PROFILE_EVENT_PERTURBATION,
            "maximum_representative_runtime_delta":
                MAX_REPRESENTATIVE_RUNTIME_DELTA,
            "maximum_profile_row_closure_abs_ms":
                MAX_PROFILE_ROW_CLOSURE_ABS_MS,
        },
        "qualified": not reasons,
        "reasons": reasons,
        "decision": {
            "deeper_fusion_design_selection_authorized": not reasons,
            "tp4_service_authorized": False,
            "main_or_yaml_change_authorized": False,
            "official_score_claim_authorized": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        required=True,
        choices=tuple(
            name for name, (_, _, kind) in CASES.items()
            if kind == "production"),
    )
    parser.add_argument("--production-extension", required=True, type=Path)
    parser.add_argument("--profile-extension", required=True, type=Path)
    parser.add_argument("--expected-production-sha256", required=True)
    parser.add_argument("--expected-profile-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--visible-physical-gpu", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "case": report["case"],
        "qualified": report["qualified"],
        "ranked_stages": report["ranked_stages"],
        "stage_share": report["stage_share"],
        "event_perturbation": report["event_perturbation"],
        "reasons": report["reasons"],
    }, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
