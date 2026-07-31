#!/usr/bin/env python3
"""Paired BI100 screen of M1-157 FP16-input QK against M1-109."""

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

import bench_m1_55_production_prefill as production


SCHEMA = "bi100-m1-157-fp16-qk-ab-cell-v1"
QUERY_LEN = 8176
WARMUPS = 1
TRIALS = 5
MIN_CELL_SPEEDUP = 0.98
CASES = {
    "p90_total_16k_q8176": (8_192, QUERY_LEN),
    "p90_total_32k_q8176": (24_576, QUERY_LEN),
    "p90_total_64k_q8176": (57_344, QUERY_LEN),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_extension(
    path: Path,
    expected_sha256: str,
    module_name: str,
) -> tuple[Any, dict[str, Any]]:
    resolved = path.resolve(strict=True)
    actual_sha256 = _sha256(resolved)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{module_name} SHA-256 mismatch: expected "
            f"{expected_sha256}, got {actual_sha256}"
        )
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension spec from {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "forward", None)):
        raise RuntimeError(f"{module_name} does not expose forward")
    return module, {
        "path": str(resolved),
        "sha256": actual_sha256,
        "size_bytes": resolved.stat().st_size,
        "module_name": module_name,
    }


def _time_once(
    function: Callable[[], tuple[Any, Any]],
) -> tuple[float, float, tuple[Any, Any]]:
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    host_start = time.perf_counter()
    start.record()
    result = function()
    end.record()
    torch.cuda.synchronize()
    return (
        float(start.elapsed_time(end)),
        (time.perf_counter() - host_start) * 1000.0,
        result,
    )


def _paired_measure(
    baseline: Callable[[], tuple[Any, Any]],
    candidate: Callable[[], tuple[Any, Any]],
) -> tuple[dict[str, Any], tuple[Any, Any], tuple[Any, Any]]:
    import torch

    for function in (baseline, candidate):
        result = function()
        torch.cuda.synchronize()
        del result

    trials = {
        "baseline": {"cuda": [], "host": []},
        "candidate": {"cuda": [], "host": []},
    }
    outputs: dict[str, tuple[Any, Any]] = {}
    functions = {"baseline": baseline, "candidate": candidate}
    for trial in range(TRIALS):
        order = (
            ("baseline", "candidate")
            if trial % 2 == 0
            else ("candidate", "baseline")
        )
        for name in order:
            cuda_ms, host_ms, outputs[name] = _time_once(functions[name])
            trials[name]["cuda"].append(cuda_ms)
            trials[name]["host"].append(host_ms)

    timings = {}
    for name in ("baseline", "candidate"):
        timings[name] = {
            "cuda_trials_ms": trials[name]["cuda"],
            "cuda_median_ms": statistics.median(trials[name]["cuda"]),
            "host_trials_ms": trials[name]["host"],
            "host_median_ms": statistics.median(trials[name]["host"]),
        }
    timings["speedup"] = (
        timings["baseline"]["cuda_median_ms"]
        / timings["candidate"]["cuda_median_ms"]
    )
    return timings, outputs["baseline"], outputs["candidate"]


def _comparison(actual: tuple[Any, Any], expected: tuple[Any, Any]) -> dict[str, Any]:
    import torch

    actual_output, actual_lse = actual
    expected_output, expected_lse = expected
    return {
        "finite": bool(
            torch.isfinite(actual_output).all().item()
            and torch.isfinite(actual_lse).all().item()
        ),
        "output_max_abs": float(
            (actual_output.float() - expected_output.float())
            .abs()
            .max()
            .item()
        ),
        "output_relative_l2": production._relative_l2(
            actual_output, expected_output
        ),
        "lse_relative_l2": production._relative_l2(
            actual_lse, expected_lse
        ),
    }


def _finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def evaluate(result: Any) -> dict[str, Any]:
    reasons: list[str] = []
    if not isinstance(result, dict):
        return {"qualified": False, "reasons": ["result must be an object"]}
    if result.get("schema") != SCHEMA:
        reasons.append("cell schema differs")
    case = result.get("case")
    if case not in CASES:
        reasons.append("case is outside the fixed P90 grid")
        return {"qualified": False, "reasons": reasons}
    context_len, query_len = CASES[case]
    for name, expected in (
        ("context_len", context_len),
        ("query_len", query_len),
        ("warmups", WARMUPS),
        ("trials", TRIALS),
        ("seed", production.SEED),
    ):
        if result.get(name) != expected:
            reasons.append(f"{name} differs")
    numerical = result.get("numerical")
    if not isinstance(numerical, dict):
        reasons.append("numerical report is missing")
    else:
        for comparison in ("candidate_vs_reference", "candidate_vs_baseline"):
            values = numerical.get(comparison)
            if not isinstance(values, dict):
                reasons.append(f"{comparison} report is missing")
                continue
            if values.get("finite") is not True:
                reasons.append(f"{comparison} is nonfinite")
            for field, limit in (
                ("output_relative_l2", production.RELATIVE_L2_LIMIT),
                ("lse_relative_l2", production.RELATIVE_L2_LIMIT),
                ("output_max_abs", production.MAX_ABS_LIMIT),
            ):
                value = values.get(field)
                if not _finite_nonnegative(value) or float(value) > limit:
                    reasons.append(
                        f"{comparison}.{field} exceeds {limit:g}"
                    )
    timings = result.get("timings")
    speedup = timings.get("speedup") if isinstance(timings, dict) else None
    if not _finite_nonnegative(speedup):
        reasons.append("speedup is invalid")
    elif float(speedup) < MIN_CELL_SPEEDUP:
        reasons.append(
            f"speedup is below the {MIN_CELL_SPEEDUP:.2f}x no-regression gate"
        )
    for side in ("baseline", "candidate"):
        values = timings.get(side) if isinstance(timings, dict) else None
        trials = values.get("cuda_trials_ms") if isinstance(values, dict) else None
        if (
            not isinstance(trials, list)
            or len(trials) != TRIALS
            or not all(_finite_nonnegative(value) and value > 0 for value in trials)
        ):
            reasons.append(f"{side} timing trials are invalid")
    if result.get("authorization") != {
        "operator_screen_only": True,
        "tp4_service_authorized": False,
        "main_or_yaml_change_authorized": False,
    }:
        reasons.append("authorization must fail closed")
    return {"qualified": not reasons, "reasons": reasons}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("M1-157 requires one visible healthy BI100")
    context_len, query_len = CASES[args.case]
    baseline, baseline_artifact = _load_extension(
        args.baseline_extension,
        args.expected_baseline_sha256,
        args.baseline_module_name,
    )
    candidate, candidate_artifact = _load_extension(
        args.candidate_extension,
        args.expected_candidate_sha256,
        args.candidate_module_name,
    )
    inputs = production._make_inputs(context_len, query_len)
    scale = production.HEAD_DIM**-0.5
    baseline_call = lambda: tuple(
        baseline.forward(*inputs, context_len, scale)
    )
    candidate_call = lambda: tuple(
        candidate.forward(*inputs, context_len, scale)
    )
    timings, baseline_result, candidate_result = _paired_measure(
        baseline_call, candidate_call
    )
    reference_result = production.reference_forward(
        *inputs, context_len, scale
    )
    result = {
        "schema": SCHEMA,
        "version": 1,
        "source_revision": args.source_revision,
        "runtime_identity": args.runtime_identity,
        "instance": args.instance,
        "visible_physical_gpu": args.visible_physical_gpu,
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "case": args.case,
        "context_len": context_len,
        "query_len": query_len,
        "total_kv_len": context_len + query_len,
        "seed": production.SEED,
        "warmups": WARMUPS,
        "trials": TRIALS,
        "baseline_extension": baseline_artifact,
        "candidate_extension": candidate_artifact,
        "timings": timings,
        "numerical": {
            "baseline_vs_reference": _comparison(
                baseline_result, reference_result
            ),
            "candidate_vs_reference": _comparison(
                candidate_result, reference_result
            ),
            "candidate_vs_baseline": _comparison(
                candidate_result, baseline_result
            ),
        },
        "authorization": {
            "operator_screen_only": True,
            "tp4_service_authorized": False,
            "main_or_yaml_change_authorized": False,
        },
    }
    result["evaluation"] = evaluate(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=tuple(CASES))
    parser.add_argument("--baseline-extension", required=True, type=Path)
    parser.add_argument("--candidate-extension", required=True, type=Path)
    parser.add_argument("--expected-baseline-sha256", required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument(
        "--baseline-module-name",
        default="corex_fused_paged_prefill",
    )
    parser.add_argument(
        "--candidate-module-name",
        default="corex_fused_paged_prefill_fp16_qk",
    )
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--runtime-identity", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--visible-physical-gpu", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            result,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "case": result["case"],
                "qualified": result["evaluation"]["qualified"],
                "speedup": result["timings"]["speedup"],
                "candidate_reference_l2": result["numerical"][
                    "candidate_vs_reference"
                ]["output_relative_l2"],
                "reasons": result["evaluation"]["reasons"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["evaluation"]["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
