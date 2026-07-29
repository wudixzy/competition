#!/usr/bin/env python3
"""Qualify four-way real-activation replay evidence for the funnel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any


REPORT_SCHEMA = "bi100-fused-prefill-activation-replay-v1"
RESULT_SCHEMA = "bi100-fused-prefill-activation-replay-qualification-v1"
CONTRACT_SCHEMA = "bi100-experiment-funnel-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            json.dump(value, stream, ensure_ascii=True, indent=2,
                      sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _finite_positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value > 0
    )


def qualify(
    reports: list[Any],
    contract: Any,
    *,
    profile: str,
) -> dict[str, Any]:
    invalid_reasons = []
    numeric_reasons = []
    performance_reasons = []
    coverage_reasons = []
    if (
        not isinstance(contract, dict)
        or contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("version") != 1
    ):
        invalid_reasons.append("experiment funnel contract is invalid")
        stages = []
    else:
        stages = contract.get("stages")
    l2 = next(
        (
            stage for stage in stages
            if isinstance(stage, dict) and stage.get("id") == "L2"
        ),
        {},
    )
    capture = l2.get("capture") or {}
    screen = l2.get("continuation_screen") or {}
    required_ranks = capture.get("required_tp_ranks")
    required_buckets = capture.get("required_context_buckets")
    required_ordinals = capture.get("required_full_attention_call_ordinals")
    minimum_speedup = screen.get("minimum_median_candidate_speedup")
    maximum_regression = screen.get(
        "maximum_single_case_regression_fraction")
    if (
        required_ranks != [0, 1, 2, 3]
        or required_buckets != [24576, 57344, 122880]
        or required_ordinals != [0, 4, 9]
        or minimum_speedup != 1.05
        or maximum_regression != 0.02
    ):
        invalid_reasons.append("L2 funnel contract differs")

    accepted = []
    identities = set()
    ranks = set()
    artifact_shas = set()
    for index, report in enumerate(reports):
        if (
            not isinstance(report, dict)
            or report.get("schema") != REPORT_SCHEMA
            or report.get("version") != 1
            or report.get("all_numeric_qualified") is not True
        ):
            invalid_reasons.append(
                f"report {index} identity or status is invalid")
            continue
        rank = report.get("rank")
        records = report.get("records")
        artifact_sha = (
            report.get("candidate_extension") or {}).get("sha256")
        identity = (
            report.get("capture_source_revision"),
            report.get("candidate_source_revision"),
            report.get("runtime_identity"),
            report.get("instance"),
        )
        if (
            rank not in {0, 1, 2, 3}
            or rank in ranks
            or not isinstance(records, list)
            or not records
            or not isinstance(artifact_sha, str)
            or len(artifact_sha) != 64
        ):
            invalid_reasons.append(f"report {index} structure is invalid")
            continue
        ranks.add(rank)
        identities.add(identity)
        artifact_shas.add(artifact_sha)
        for record in records:
            numeric = record.get("numeric") or {}
            speedup = record.get("candidate_speedup")
            if (
                record.get("rank") != rank
                or numeric.get("finite") is not True
                or numeric.get("lse_finite") is not True
                or numeric.get("qualified") is not True
            ):
                numeric_reasons.append(
                    f"rank {rank} has an unqualified numeric record")
            if not _finite_positive(speedup):
                invalid_reasons.append(
                    f"rank {rank} has invalid replay timing")
            else:
                accepted.append(record)
    if ranks != {0, 1, 2, 3}:
        coverage_reasons.append("replay does not cover all four TP ranks")
    if len(identities) != 1:
        invalid_reasons.append("replay reports have different run identities")
    if len(artifact_shas) != 1:
        invalid_reasons.append("replay reports use different artifacts")

    observed = {
        (
            record.get("rank"),
            record.get("bucket_min_context_tokens"),
            record.get("call_ordinal"),
        )
        for record in accepted
    }
    if profile == "qualification":
        expected = {
            (rank, bucket, ordinal)
            for rank in required_ranks
            for bucket in required_buckets
            for ordinal in required_ordinals
        }
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        if missing:
            coverage_reasons.append(
                f"qualification profile is missing {len(missing)} cases")
        if extra:
            coverage_reasons.append(
                f"qualification profile has {len(extra)} unexpected cases")
    elif profile == "smoke":
        for rank in required_ranks:
            if not any(row[0] == rank for row in observed):
                coverage_reasons.append(
                    f"smoke profile has no case for rank {rank}")
    else:
        invalid_reasons.append("unknown replay qualification profile")

    speedups = [
        float(record["candidate_speedup"]) for record in accepted
        if _finite_positive(record.get("candidate_speedup"))
    ]
    median_speedup = statistics.median(speedups) if speedups else None
    minimum_case_speedup = min(speedups) if speedups else None
    if (
        median_speedup is None
        or median_speedup < float(minimum_speedup or math.inf)
    ):
        performance_reasons.append(
            "median replay speedup is below the continuation screen")
    minimum_allowed = 1.0 - float(maximum_regression or 0.0)
    if (
        minimum_case_speedup is None
        or minimum_case_speedup < minimum_allowed
    ):
        performance_reasons.append(
            "a replay case exceeds the allowed regression")

    execution_valid = not invalid_reasons and not numeric_reasons
    stage_qualified = bool(
        execution_valid
        and not performance_reasons
        and not coverage_reasons)
    return {
        "schema": RESULT_SCHEMA,
        "version": 1,
        "profile": profile,
        "execution_valid": execution_valid,
        "stage_qualified": stage_qualified,
        "invalid_reasons": invalid_reasons,
        "numeric_reasons": numeric_reasons,
        "performance_reasons": performance_reasons,
        "coverage_reasons": coverage_reasons,
        "report_count": len(reports),
        "record_count": len(accepted),
        "ranks": sorted(ranks),
        "median_candidate_speedup": median_speedup,
        "minimum_case_speedup": minimum_case_speedup,
        "contract_sha256": None,
        "authorization": {
            "short_tp4_authorized": (
                stage_qualified and profile == "qualification"),
            "long_context_authorized": False,
            "main_or_yaml_change_authorized": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=("smoke", "qualification"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reports = [
        json.loads(path.read_text(encoding="ascii"))
        for path in args.report
    ]
    contract = json.loads(args.contract.read_text(encoding="ascii"))
    result = qualify(reports, contract, profile=args.profile)
    result["contract_sha256"] = _sha256(args.contract)
    _atomic_write(args.out, result)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if result["stage_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
