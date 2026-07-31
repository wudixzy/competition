#!/usr/bin/env python3
"""Summarize the two order-balanced M1-170 cold timing screens."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


COMPARISON_SCHEMA = "bi100-m1-170-cold-capture-overhead-comparison-v2"
RUNNER_SCHEMA = "bi100-m1-170-cold-capture-overhead-runner-v2"
SCHEMA = "bi100-m1-170-order-balanced-summary-v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _validate_comparison(value: dict[str, Any], label: str) -> None:
    if (
        value.get("schema") != COMPARISON_SCHEMA
        or value.get("version") != 2
        or value.get("qualified_analysis") is not True
        or value.get("reasons") != []
        or value.get("cold_isolation", {}).get("qualified") is not True
        or value.get("cold_isolation", {}).get(
            "admission64_cold_cached_tokens") != 0
        or value.get("cold_isolation", {}).get("off_cold_cached_tokens") != 0
    ):
        raise ValueError(f"{label} comparison is not a qualified cold screen")


def _validate_runner(
    value: dict[str, Any], label: str, expected_order: list[str]
) -> None:
    gates = value.get("gates")
    if (
        value.get("schema") != RUNNER_SCHEMA
        or value.get("version") != 2
        or value.get("qualified_development_screen") is not True
        or value.get("returncode") != 0
        or value.get("arm_order") != expected_order
        or value.get("bench_tool_count") != 0
        or not isinstance(gates, dict)
        or not gates
        or any(result != 0 for result in gates.values())
    ):
        raise ValueError(f"{label} runner contract differs")


def _balanced(left: float, right: float) -> float:
    left_ratio = 1.0 + float(left)
    right_ratio = 1.0 + float(right)
    if left_ratio <= 0 or right_ratio <= 0:
        raise ValueError("overhead ratio must be positive")
    return math.sqrt(left_ratio * right_ratio) - 1.0


def summarize(
    forward: dict[str, Any],
    reverse: dict[str, Any],
    forward_runner: dict[str, Any],
    reverse_runner: dict[str, Any],
) -> dict[str, Any]:
    _validate_comparison(forward, "forward")
    _validate_comparison(reverse, "reverse")
    _validate_runner(
        forward_runner, "forward", ["admission64", "off"])
    _validate_runner(
        reverse_runner, "reverse", ["off", "admission64"])
    if forward.get("request_manifest_sha256") != reverse.get(
            "request_manifest_sha256"):
        raise ValueError("request manifests differ")
    if forward_runner.get("source_revision") != reverse_runner.get(
            "source_revision"):
        raise ValueError("source revisions differ")

    forward_median = float(
        forward["cold"]["admission64_overhead_fraction_median"])
    reverse_median = float(
        reverse["cold"]["admission64_overhead_fraction_median"])
    forward_p90 = float(
        forward["cold"]["admission64_overhead_fraction_p90"])
    reverse_p90 = float(
        reverse["cold"]["admission64_overhead_fraction_p90"])
    by_shape = {}
    for shape in sorted(forward["by_shape"], key=int):
        if shape not in reverse["by_shape"]:
            raise ValueError(f"reverse result is missing shape={shape}")
        first = float(
            forward["by_shape"][shape]["admission64_overhead_fraction"])
        second = float(
            reverse["by_shape"][shape]["admission64_overhead_fraction"])
        by_shape[shape] = {
            "forward_overhead_fraction": first,
            "reverse_overhead_fraction": second,
            "order_range": [min(first, second), max(first, second)],
            "order_balanced_geometric_overhead_fraction": _balanced(
                first, second),
        }

    numeric = {}
    for field in (
        "first_token_identity_rate",
        "complete_output_identity_rate",
        "cold_first_token_identity_rate",
        "cold_complete_output_identity_rate",
    ):
        left = float(forward["cross_policy_numeric_observation"][field])
        right = float(reverse["cross_policy_numeric_observation"][field])
        numeric[field] = {"forward": left, "reverse": right}

    return {
        "schema": SCHEMA,
        "version": 1,
        "qualified_order_balanced_timing": True,
        "source_revision": forward_runner["source_revision"],
        "request_manifest_sha256": forward["request_manifest_sha256"],
        "cold_request_count_per_order": forward["cold"]["request_count"],
        "median": {
            "forward_overhead_fraction": forward_median,
            "reverse_overhead_fraction": reverse_median,
            "order_range": [
                min(forward_median, reverse_median),
                max(forward_median, reverse_median),
            ],
            "order_balanced_geometric_overhead_fraction": _balanced(
                forward_median, reverse_median),
        },
        "p90": {
            "forward_overhead_fraction": forward_p90,
            "reverse_overhead_fraction": reverse_p90,
            "order_range": [
                min(forward_p90, reverse_p90),
                max(forward_p90, reverse_p90),
            ],
            "order_balanced_geometric_overhead_fraction": _balanced(
                forward_p90, reverse_p90),
        },
        "by_shape": by_shape,
        "cross_policy_numeric_observation": numeric,
        "scope": {
            "diagnostic_four_layer_model": True,
            "tp4_evaluated": False,
            "semantic_quality_evaluated": False,
            "statistical_significance_claimed": False,
            "production_promotion_authorized": False,
            "yaml_or_main_change_authorized": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward", type=Path, required=True)
    parser.add_argument("--reverse", type=Path, required=True)
    parser.add_argument("--forward-runner", type=Path, required=True)
    parser.add_argument("--reverse-runner", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(
        _load(args.forward),
        _load(args.reverse),
        _load(args.forward_runner),
        _load(args.reverse_runner),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
