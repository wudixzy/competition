#!/usr/bin/env python3
"""Retest the closed M1-55 kernel only on its proposed production domain."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

from bench_m1_55_production_prefill import (
    HEAD_DIM,
    MAX_ABS_LIMIT,
    RELATIVE_L2_LIMIT,
    _make_inputs,
    _measure,
    _relative_l2,
    reference_forward,
)


SCHEMA = "bi100-m1-111-query-tiled-production-retest-v1"
CANDIDATE_ORIGIN_COMMIT = "a30b6e7212286cd613c946b1ca02d8972a198863"
MIN_LONG_SPEEDUP = 1.5
MIN_SHORT_SPEEDUP = 0.98
CASES = {
    "production_dense_q8176": (0, 8_176, "short"),
    "production_32k_q8176": (24_576, 8_176, "short"),
    "production_65k_q8176": (65_536, 8_176, "long"),
    "production_128k_q8176": (122_880, 8_176, "long"),
    "production_235k_q5616": (229_376, 5_616, "long"),
    "boundary_262k_q8192": (253_952, 8_192, "long"),
}


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
    return module, {
        "path": str(resolved),
        "sha256": digest,
        "size_bytes": resolved.stat().st_size,
    }


def numerical(actual: tuple[Any, Any], expected: tuple[Any, Any]) -> dict:
    import torch

    output, lse = actual
    expected_output, expected_lse = expected
    return {
        "finite": bool(
            torch.isfinite(output).all().item()
            and torch.isfinite(lse).all().item()
        ),
        "output_relative_l2": _relative_l2(output, expected_output),
        "lse_relative_l2": _relative_l2(lse, expected_lse),
        "output_max_abs": float(
            (output.float() - expected_output.float()).abs().max().item()
        ),
        "output_exact": bool(torch.equal(output, expected_output)),
        "lse_exact": bool(torch.equal(lse, expected_lse)),
    }


def numerical_reasons(label: str, result: dict) -> list[str]:
    reasons = []
    if result.get("finite") is not True:
        reasons.append(f"{label} output is not finite")
    for field, limit in (
        ("output_relative_l2", RELATIVE_L2_LIMIT),
        ("lse_relative_l2", RELATIVE_L2_LIMIT),
        ("output_max_abs", MAX_ABS_LIMIT),
    ):
        value = result.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            reasons.append(f"{label}.{field} is invalid")
        elif float(value) > limit:
            reasons.append(
                f"{label}.{field}={value:.9g} exceeds {limit:.9g}")
    return reasons


def run(args: argparse.Namespace) -> dict:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "M1-111 requires exactly one visible CoreX GPU")
    context_len, query_len, case_class = CASES[args.case]
    if query_len < 4_096:
        raise RuntimeError("M1-111 production domain requires query >= 4096")

    baseline, baseline_artifact = load_extension(
        args.baseline_extension,
        module_name="corex_fused_paged_prefill",
        expected_sha256=args.expected_baseline_sha256,
    )
    candidate, candidate_artifact = load_extension(
        args.candidate_extension,
        module_name="corex_query_tiled_paged_prefill_retest",
        expected_sha256=args.expected_candidate_sha256,
    )
    inputs = _make_inputs(context_len, query_len)
    scale = HEAD_DIM ** -0.5
    calls = {
        "baseline": lambda: tuple(
            baseline.forward(*inputs, context_len, scale)),
        "candidate": lambda: tuple(
            candidate.forward(*inputs, context_len, scale)),
    }
    timings = {}
    results = {}
    for label in args.order.split(","):
        timings[label], results[label] = _measure(calls[label])
    reference = reference_forward(*inputs, context_len, scale)
    torch.cuda.synchronize()

    baseline_numerical = numerical(results["baseline"], reference)
    candidate_numerical = numerical(results["candidate"], reference)
    candidate_vs_baseline = numerical(
        results["candidate"], results["baseline"])
    speedup = (
        timings["baseline"]["cuda_median_ms"]
        / timings["candidate"]["cuda_median_ms"]
    )
    reasons = []
    reasons.extend(numerical_reasons("baseline", baseline_numerical))
    reasons.extend(numerical_reasons("candidate", candidate_numerical))
    required_speedup = (
        MIN_LONG_SPEEDUP if case_class == "long" else MIN_SHORT_SPEEDUP)
    if not math.isfinite(speedup) or speedup < required_speedup:
        reasons.append(
            f"baseline/candidate speedup {speedup:.6f} is below "
            f"{required_speedup:.2f}x")

    return {
        "schema": SCHEMA,
        "source_commit": args.source_commit,
        "candidate_origin_commit": CANDIDATE_ORIGIN_COMMIT,
        "instance": args.instance,
        "visible_physical_gpu": args.visible_physical_gpu,
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "case": args.case,
        "case_class": case_class,
        "context_len": context_len,
        "query_len": query_len,
        "total_kv_len": context_len + query_len,
        "physical_block_permutation": context_len > 0,
        "order": args.order,
        "baseline_extension": baseline_artifact,
        "candidate_extension": candidate_artifact,
        "timings": {
            "baseline": timings["baseline"],
            "candidate": timings["candidate"],
            "baseline_over_candidate": speedup,
        },
        "numerical": {
            "baseline_vs_reference": baseline_numerical,
            "candidate_vs_reference": candidate_numerical,
            "candidate_vs_baseline": candidate_vs_baseline,
        },
        "thresholds": {
            "maximum_relative_l2": RELATIVE_L2_LIMIT,
            "maximum_absolute_error": MAX_ABS_LIMIT,
            "minimum_long_speedup": MIN_LONG_SPEEDUP,
            "minimum_short_speedup": MIN_SHORT_SPEEDUP,
            "minimum_supported_query_len": 4_096,
        },
        "qualified": not reasons,
        "reasons": reasons,
        "decision": {
            "runtime_integration_authorized": not reasons,
            "tp4_service_authorized": False,
            "main_or_yaml_change_authorized": False,
            "official_score_claim_authorized": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=tuple(CASES))
    parser.add_argument("--baseline-extension", required=True, type=Path)
    parser.add_argument("--candidate-extension", required=True, type=Path)
    parser.add_argument("--expected-baseline-sha256", required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--visible-physical-gpu", required=True, type=int)
    parser.add_argument(
        "--order",
        required=True,
        choices=("baseline,candidate", "candidate,baseline"),
    )
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
        "speedup": report["timings"]["baseline_over_candidate"],
        "candidate_output_relative_l2": report["numerical"][
            "candidate_vs_reference"]["output_relative_l2"],
        "reasons": report["reasons"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
