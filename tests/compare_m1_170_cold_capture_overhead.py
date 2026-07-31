#!/usr/bin/env python3
"""Attribute cold-prefill overhead to admission64 state capture."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any

try:
    from tests import bench_m1_104_admission64_policy_matrix as matrix
except ImportError:
    import bench_m1_104_admission64_policy_matrix as matrix


SCHEMA = "bi100-m1-170-cold-capture-overhead-comparison-v2"
OUTPUT_FIELDS = (
    "first_token_sha256",
    "output_sha256",
    "content_sha256",
    "reasoning_sha256",
    "tool_calls_sha256",
    "finish_reason",
    "completion_tokens",
)


def _load(path: Path, policy: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema") != matrix.SCHEMA
        or value.get("version") != matrix.VERSION
        or value.get("policy") != policy
        or value.get("qualified_measurement") is not True
        or value.get("reasons") != []
        or value.get("request_count") != matrix.REQUEST_COUNT
        or value.get("fixed", {}).get("salt_order") != "identity-first"
        or value.get("fixed", {}).get("tool_count") != 0
    ):
        raise ValueError(f"{path} measurement contract differs")
    records = value.get("requests")
    if not isinstance(records, list) or len(records) != matrix.REQUEST_COUNT:
        raise ValueError(f"{path} request matrix differs")
    return value


def _identity(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record.get(name) for name in (
        "request_id",
        "target_prompt_tokens",
        "pair",
        "phase",
        "salt_sha256",
        "rendered_tokens_local",
        "seed",
    ))


def _positive(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def _percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def compare(control: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    control_rows = control["requests"]
    candidate_rows = candidate["requests"]
    if [_identity(row) for row in control_rows] != [
            _identity(row) for row in candidate_rows]:
        raise ValueError("control/candidate request identities differ")
    if control.get("request_manifest_sha256") != candidate.get(
            "request_manifest_sha256"):
        raise ValueError("control/candidate request manifests differ")

    output_matches = 0
    first_token_matches = 0
    cold_output_matches = 0
    cold_first_token_matches = 0
    by_shape: dict[str, Any] = {}
    all_control_cold: list[float] = []
    all_candidate_cold: list[float] = []
    control_cold_cached_tokens = 0
    candidate_cold_cached_tokens = 0
    for target in matrix.SHAPES:
        control_ttft: list[float] = []
        candidate_ttft: list[float] = []
        control_cached = 0
        candidate_cached = 0
        for left, right in zip(control_rows, candidate_rows):
            if left.get("target_prompt_tokens") != target:
                continue
            exact = all(left.get(field) == right.get(field)
                        for field in OUTPUT_FIELDS)
            output_matches += int(exact)
            first_token_matches += int(
                left.get("first_token_sha256")
                == right.get("first_token_sha256"))
            if left.get("phase") != "cold":
                continue
            cold_output_matches += int(exact)
            cold_first_token_matches += int(
                left.get("first_token_sha256")
                == right.get("first_token_sha256"))
            left_ttft = _positive(left.get("ttft_s"), "control.ttft_s")
            right_ttft = _positive(right.get("ttft_s"), "candidate.ttft_s")
            control_ttft.append(left_ttft)
            candidate_ttft.append(right_ttft)
            all_control_cold.append(left_ttft)
            all_candidate_cold.append(right_ttft)
            control_cached += int(left.get("cached_tokens") or 0)
            candidate_cached += int(right.get("cached_tokens") or 0)
        if len(control_ttft) != len(matrix.PAIRS):
            raise ValueError(f"target={target} cold matrix differs")
        control_cold_cached_tokens += control_cached
        candidate_cold_cached_tokens += candidate_cached
        control_median = statistics.median(control_ttft)
        candidate_median = statistics.median(candidate_ttft)
        by_shape[str(target)] = {
            "request_count": len(control_ttft),
            "admission64_ttft_median_s": control_median,
            "off_ttft_median_s": candidate_median,
            "admission64_overhead_fraction": (
                control_median / candidate_median - 1.0),
            "admission64_cached_tokens": control_cached,
            "off_cached_tokens": candidate_cached,
        }

    cold_count = len(matrix.SHAPES) * len(matrix.PAIRS)
    cold_isolated = (
        control_cold_cached_tokens == 0
        and candidate_cold_cached_tokens == 0
    )
    qualified = cold_isolated
    return {
        "schema": SCHEMA,
        "version": 2,
        "qualified_analysis": qualified,
        "reasons": [
            reason for condition, reason in (
                (not cold_isolated,
                 "cold rows contain cached tokens and cannot attribute "
                 "capture overhead"),
            ) if condition
        ],
        "control_policy": "admission64",
        "candidate_policy": "off",
        "request_manifest_sha256": control["request_manifest_sha256"],
        "cold": {
            "request_count": cold_count,
            "admission64_ttft_median_s": statistics.median(all_control_cold),
            "off_ttft_median_s": statistics.median(all_candidate_cold),
            "admission64_ttft_p90_s": _percentile(all_control_cold, 90),
            "off_ttft_p90_s": _percentile(all_candidate_cold, 90),
            "admission64_overhead_fraction_median": (
                statistics.median(all_control_cold)
                / statistics.median(all_candidate_cold) - 1.0),
            "admission64_overhead_fraction_p90": (
                _percentile(all_control_cold, 90)
                / _percentile(all_candidate_cold, 90) - 1.0),
        },
        "cold_isolation": {
            "qualified": cold_isolated,
            "admission64_cold_cached_tokens": control_cold_cached_tokens,
            "off_cold_cached_tokens": candidate_cold_cached_tokens,
        },
        "cross_policy_numeric_observation": {
            "first_token_identity_matches": first_token_matches,
            "first_token_identity_rate": (
                first_token_matches / matrix.REQUEST_COUNT),
            "complete_output_identity_matches": output_matches,
            "complete_output_identity_rate": (
                output_matches / matrix.REQUEST_COUNT),
            "cold_first_token_identity_matches": cold_first_token_matches,
            "cold_first_token_identity_rate": (
                cold_first_token_matches / cold_count),
            "cold_complete_output_identity_matches": cold_output_matches,
            "cold_complete_output_identity_rate": (
                cold_output_matches / cold_count),
            "strict_output_identity_qualified": (
                output_matches == matrix.REQUEST_COUNT),
            "teacher_forced_logits_evaluated": False,
            "relative_l2_evaluated": False,
            "strict_cross_policy_output_required_for_timing": False,
        },
        "by_shape": by_shape,
        "scope": {
            "diagnostic_four_layer_model": True,
            "cache_off_is_diagnostic_only": True,
            "tp4_evaluated": False,
            "semantic_quality_evaluated": False,
            "production_promotion_authorized": False,
            "yaml_or_main_change_authorized": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = compare(
        _load(args.control, "admission64"),
        _load(args.candidate, "off"),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qualified_analysis"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
