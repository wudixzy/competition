#!/usr/bin/env python3
"""Compare the fixed M1-169 TP1 control and candidate measurements."""

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


SCHEMA = "bi100-m1-169-tail64-nofinal-tp1-comparison-v1"
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


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def _relative_improvement(control: float, candidate: float) -> float:
    return (control - candidate) / control


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
    by_shape: dict[str, Any] = {}
    for target in matrix.SHAPES:
        target_rows: dict[str, list[float]] = {
            "control_cold": [],
            "control_warm": [],
            "candidate_cold": [],
            "candidate_warm": [],
        }
        cached = {name: 0 for name in (
            "control_cold", "control_warm",
            "candidate_cold", "candidate_warm")}
        for left, right in zip(control_rows, candidate_rows):
            if left["target_prompt_tokens"] != target:
                continue
            phase = left["phase"]
            left_key = f"control_{phase}"
            right_key = f"candidate_{phase}"
            target_rows[left_key].append(
                _finite(left.get("ttft_s"), f"{left_key}.ttft_s"))
            target_rows[right_key].append(
                _finite(right.get("ttft_s"), f"{right_key}.ttft_s"))
            cached[left_key] += int(left.get("cached_tokens") or 0)
            cached[right_key] += int(right.get("cached_tokens") or 0)
            if all(left.get(name) == right.get(name)
                   for name in OUTPUT_FIELDS):
                output_matches += 1
        medians = {
            name: statistics.median(values)
            for name, values in target_rows.items()
        }
        by_shape[str(target)] = {
            "ttft_median_s": medians,
            "ttft_relative_improvement": {
                phase: _relative_improvement(
                    medians[f"control_{phase}"],
                    medians[f"candidate_{phase}"],
                )
                for phase in matrix.PHASES
            },
            "cached_tokens": cached,
        }

    control_aggregate = control["aggregate"]
    candidate_aggregate = candidate["aggregate"]
    return {
        "schema": SCHEMA,
        "version": 1,
        "qualified_analysis": True,
        "control_policy": "admission64",
        "candidate_policy": "tail64_nofinal",
        "request_count": matrix.REQUEST_COUNT,
        "request_manifest_sha256": control["request_manifest_sha256"],
        "cache_transparency": {
            "control_cold_warm_exact": True,
            "candidate_cold_warm_exact": True,
            "cross_policy_output_identity_matches": output_matches,
            "cross_policy_output_identity_rate": (
                output_matches / matrix.REQUEST_COUNT),
            "cross_policy_exact_required_for_analysis": False,
        },
        "aggregate": {
            "control": control_aggregate,
            "candidate": candidate_aggregate,
            "candidate_relative_improvement": {
                "ttft_p90": _relative_improvement(
                    _finite(control_aggregate["ttft_p90_s"], "control.ttft"),
                    _finite(candidate_aggregate["ttft_p90_s"], "candidate.ttft"),
                ),
                "weighted": (
                    _finite(candidate_aggregate["weighted"], "candidate.weighted")
                    / _finite(control_aggregate["weighted"], "control.weighted")
                    - 1.0
                ),
            },
        },
        "by_shape": by_shape,
        "scope": {
            "diagnostic_four_layer_model": True,
            "tp4_evaluated": False,
            "semantic_quality_evaluated": False,
            "official_workload_evaluated": False,
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
        _load(args.candidate, "tail64_nofinal"),
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
