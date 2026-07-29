#!/usr/bin/env python3
"""Classify teacher-forced evidence without treating drift as capability loss."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


REPORT_SCHEMA = "bi100-teacher-forced-distribution-classification-v2"
COMPARISON_SCHEMA = "bi100-teacher-forced-topk-comparison-v1"
CONTRACT_SCHEMA = "bi100-layered-quality-gate-contract-v2"
Json = dict[str, Any]


def _atomic_write(path: Path, value: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
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


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_comparison(value: Any, mode: str, label: str) -> list[str]:
    reasons = []
    if not isinstance(value, dict):
        return [f"{label}: report root must be an object"]
    if (
        value.get("schema") != COMPARISON_SCHEMA
        or value.get("version") != 1
        or value.get("comparison_mode") != mode
        or value.get("status") == "invalid"
        or value.get("gpu_count") != 4
        or value.get("tensor_parallel_size") != 4
        or value.get("max_model_len") != 262144
        or value.get("top_k") != 5
        or value.get("case_count") != 5
        or value.get("sampled_positions") != 320
    ):
        reasons.append(f"{label}: comparison identity is invalid")
    metrics = value.get("metrics")
    required_metrics = {
        "top1_agreement",
        "top1_mismatch_count",
        "mutually_uncovered_top1_mismatches",
        "high_margin_guard_nats",
        "high_margin_top1_mismatches",
        "teacher_token_logprob_absolute_delta_max",
        "teacher_token_logprob_absolute_delta_p99",
        "shared_topk_logprob_absolute_delta_p99",
        "mean_teacher_token_nll_regression",
    }
    if (
        not isinstance(metrics, dict)
        or set(metrics) != required_metrics
        or any(not _finite_number(metrics.get(name))
               for name in required_metrics)
    ):
        reasons.append(f"{label}: metrics are invalid")
        metrics_valid = False
    else:
        metrics_valid = True
        count_fields = (
            "top1_mismatch_count",
            "mutually_uncovered_top1_mismatches",
            "high_margin_top1_mismatches",
        )
        if any(
            not isinstance(metrics[name], int)
            or isinstance(metrics[name], bool)
            or not 0 <= metrics[name] <= 320
            for name in count_fields
        ):
            reasons.append(f"{label}: metric counts are invalid")
            metrics_valid = False
        if (
            not 0.0 <= float(metrics["top1_agreement"]) <= 1.0
            or float(metrics["high_margin_guard_nats"]) < 0.0
            or any(
                float(metrics[name]) < 0.0
                for name in (
                    "teacher_token_logprob_absolute_delta_max",
                    "teacher_token_logprob_absolute_delta_p99",
                    "shared_topk_logprob_absolute_delta_p99",
                )
            )
            or (
                float(metrics[
                    "teacher_token_logprob_absolute_delta_p99"])
                > float(metrics[
                    "teacher_token_logprob_absolute_delta_max"])
            )
        ):
            reasons.append(f"{label}: metric domains are invalid")
            metrics_valid = False
        if (
            isinstance(metrics["top1_mismatch_count"], int)
            and not isinstance(metrics["top1_mismatch_count"], bool)
            and abs(
                float(metrics["top1_agreement"]) * 320
                - (320 - metrics["top1_mismatch_count"])
            ) > 1.0e-9
        ):
            reasons.append(f"{label}: top-1 count and rate disagree")
            metrics_valid = False
        if (
            isinstance(metrics["top1_mismatch_count"], int)
            and not isinstance(metrics["top1_mismatch_count"], bool)
            and any(
                isinstance(metrics[name], int)
                and not isinstance(metrics[name], bool)
                and metrics[name] > metrics["top1_mismatch_count"]
                for name in (
                    "mutually_uncovered_top1_mismatches",
                    "high_margin_top1_mismatches",
                )
            )
        ):
            reasons.append(f"{label}: mismatch subsets are inconsistent")
            metrics_valid = False
    cases = value.get("cases")
    expected_lengths = [4096, 32768, 65536, 131072, 235000]
    if (
        not isinstance(cases, list)
        or any(not isinstance(case, dict) for case in cases)
        or sorted(case.get("prompt_tokens") for case in cases)
        != expected_lengths
        or any(case.get("sampled_positions") != 64 for case in cases)
    ):
        reasons.append(f"{label}: prompt-case matrix is invalid")
    else:
        case_matches = 0
        case_uncovered = 0
        case_domains_valid = True
        for case in cases:
            agreement = case.get("top1_agreement")
            uncovered = case.get("mutually_uncovered_top1_mismatches")
            if (
                set(case) != {
                    "id", "prompt_tokens", "sampled_positions",
                    "top1_agreement",
                    "mutually_uncovered_top1_mismatches",
                }
                or not isinstance(agreement, (int, float))
                or isinstance(agreement, bool)
                or not math.isfinite(float(agreement))
                or not 0.0 <= float(agreement) <= 1.0
                or not isinstance(uncovered, int)
                or isinstance(uncovered, bool)
                or not 0 <= uncovered <= 64
                or abs(float(agreement) * 64 - round(float(agreement) * 64))
                > 1.0e-9
            ):
                case_domains_valid = False
                continue
            case_matches += round(float(agreement) * 64)
            case_uncovered += uncovered
        if not case_domains_valid:
            reasons.append(f"{label}: prompt-case metric domains are invalid")
        elif metrics_valid and (
            case_matches != 320 - metrics["top1_mismatch_count"]
            or case_uncovered
            != metrics["mutually_uncovered_top1_mismatches"]
        ):
            reasons.append(f"{label}: prompt-case and aggregate metrics disagree")
    return reasons


def _validate_contract(value: Any) -> list[str]:
    if (
        not isinstance(value, dict)
        or value.get("schema") != CONTRACT_SCHEMA
        or value.get("version") != 2
        or not isinstance(value.get("teacher_forced_distribution"), dict)
    ):
        return ["v2 contract identity is invalid"]
    distribution = value["teacher_forced_distribution"]
    if (
        distribution.get("top_k") != 5
        or distribution.get("required_prompt_tokens")
        != [4096, 32768, 65536, 131072, 235000]
        or distribution.get("minimum_positions_per_case") != 64
        or distribution.get("candidate_exceeds_tight_equivalence_action")
        != "escalate"
        or distribution.get("candidate_exceedance_is_operator_failure")
        is not False
        or distribution.get("candidate_exceedance_is_capability_failure")
        is not False
    ):
        return ["teacher-forced contract identity is invalid"]
    repeat = distribution.get("control_repeat")
    tight = distribution.get("tight_equivalence")
    repeat_fields = {
        "minimum_top1_agreement",
        "maximum_actual_logprob_absolute_delta_p99",
        "maximum_shared_logprob_absolute_delta_p99",
        "maximum_absolute_mean_teacher_nll_delta",
        "maximum_mutually_uncovered_top1_mismatches",
    }
    tight_fields = {
        "minimum_top1_agreement",
        "maximum_actual_logprob_absolute_delta",
        "maximum_actual_logprob_absolute_delta_p99",
        "maximum_shared_logprob_absolute_delta_p99",
        "maximum_mean_actual_nll_regression",
        "maximum_high_margin_top1_mismatches",
        "maximum_mutually_uncovered_top1_mismatches",
    }
    if (
        not isinstance(repeat, dict)
        or set(repeat) != repeat_fields
        or not isinstance(tight, dict)
        or set(tight) != tight_fields
        or any(not _finite_number(item) for item in repeat.values())
        or any(not _finite_number(item) for item in tight.values())
        or not 0.0 <= float(repeat["minimum_top1_agreement"]) <= 1.0
        or not 0.0 <= float(tight["minimum_top1_agreement"]) <= 1.0
        or any(float(item) < 0.0 for name, item in repeat.items()
               if name != "minimum_top1_agreement")
        or any(float(item) < 0.0 for name, item in tight.items()
               if name != "minimum_top1_agreement")
    ):
        return ["teacher-forced thresholds are invalid"]
    return []


def _violations(metrics: Json, thresholds: Json, *, repeat: bool) -> list[str]:
    reasons = []
    if metrics["top1_agreement"] < thresholds["minimum_top1_agreement"]:
        reasons.append("top-1 agreement is outside the envelope")
    if (
        metrics["teacher_token_logprob_absolute_delta_p99"]
        > thresholds["maximum_actual_logprob_absolute_delta_p99"]
    ):
        reasons.append("teacher-token p99 drift is outside the envelope")
    if (
        metrics["shared_topk_logprob_absolute_delta_p99"]
        > thresholds["maximum_shared_logprob_absolute_delta_p99"]
    ):
        reasons.append("shared-top-k p99 drift is outside the envelope")
    if (
        metrics["mutually_uncovered_top1_mismatches"]
        > thresholds["maximum_mutually_uncovered_top1_mismatches"]
    ):
        reasons.append("mutually uncovered top-1 choices exceed the envelope")
    if repeat:
        if (
            abs(metrics["mean_teacher_token_nll_regression"])
            > thresholds["maximum_absolute_mean_teacher_nll_delta"]
        ):
            reasons.append("control-repeat mean NLL drift is outside the envelope")
        return reasons
    if (
        metrics["teacher_token_logprob_absolute_delta_max"]
        > thresholds["maximum_actual_logprob_absolute_delta"]
    ):
        reasons.append("teacher-token maximum drift is outside the envelope")
    if (
        metrics["mean_teacher_token_nll_regression"]
        > thresholds["maximum_mean_actual_nll_regression"]
    ):
        reasons.append("mean teacher-token NLL regression exceeds the envelope")
    if (
        metrics["high_margin_top1_mismatches"]
        > thresholds["maximum_high_margin_top1_mismatches"]
    ):
        reasons.append("high-margin top-1 choices exceed the envelope")
    return reasons


def classify(candidate: Any, control_repeat: Any, contract: Any) -> Json:
    validation_reasons = []
    validation_reasons.extend(_validate_comparison(
        candidate, "candidate", "candidate"))
    validation_reasons.extend(_validate_comparison(
        control_repeat, "control-repeat", "control_repeat"))
    validation_reasons.extend(_validate_contract(contract))

    base = {
        "schema": REPORT_SCHEMA,
        "version": 2,
        "candidate_source_revision": (
            candidate.get("source_revision")
            if isinstance(candidate, dict) else None),
        "control_repeat_source_revision": (
            control_repeat.get("source_revision")
            if isinstance(control_repeat, dict) else None),
        "case_count": (
            candidate.get("case_count")
            if isinstance(candidate, dict) else None),
        "sampled_positions": (
            candidate.get("sampled_positions")
            if isinstance(candidate, dict) else None),
        "promotion_authorized": False,
        "operator_numerics_decided": False,
        "capability_noninferiority_decided": False,
        "privacy": {
            "contains_private_token_identity": False,
            "contains_token_ids": False,
            "contains_prompts": False,
            "contains_model_outputs": False,
            "contains_credentials": False,
        },
    }
    if validation_reasons:
        return {
            **base,
            "status": "invalid",
            "classification": "invalid",
            "validation_reasons": validation_reasons,
            "control_repeat_reasons": [],
            "candidate_drift_reasons": [],
            "next_required_evidence": [],
        }

    distribution = contract["teacher_forced_distribution"]
    repeat_reasons = _violations(
        control_repeat["metrics"], distribution["control_repeat"],
        repeat=True)
    if repeat_reasons:
        return {
            **base,
            "status": "invalid",
            "classification": "measurement-not-repeatable",
            "validation_reasons": [],
            "control_repeat_reasons": repeat_reasons,
            "candidate_drift_reasons": [],
            "next_required_evidence": [
                "repair-or-explain-control-repeatability",
                "rerun-unchanged-control-repeat",
            ],
        }

    drift_reasons = _violations(
        candidate["metrics"], distribution["tight_equivalence"],
        repeat=False)
    if drift_reasons:
        return {
            **base,
            "status": "escalate",
            "classification": "distribution-drift-requires-adjudication",
            "validation_reasons": [],
            "control_repeat_reasons": [],
            "candidate_drift_reasons": drift_reasons,
            "next_required_evidence": [
                "same-real-activation-operator-shadow-reference",
                "paired-powered-task-capability-noninferiority",
            ],
        }
    return {
        **base,
        "status": "pass",
        "classification": "tight-distribution-equivalence",
        "validation_reasons": [],
        "control_repeat_reasons": [],
        "candidate_drift_reasons": [],
        "next_required_evidence": [
            "same-real-activation-operator-shadow-reference",
            "standard-capability-regression-matrix",
        ],
    }


def exit_code(status: str) -> int:
    return {"pass": 0, "escalate": 0, "invalid": 2}[status]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("control_repeat", type=Path)
    parser.add_argument(
        "--contract", type=Path,
        default=Path("quality/layered_quality_gate.v2.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = classify(
        json.loads(args.candidate.read_text(encoding="utf-8")),
        json.loads(args.control_repeat.read_text(encoding="utf-8")),
        json.loads(args.contract.read_text(encoding="utf-8")),
    )
    _atomic_write(args.out, result)
    return exit_code(result["status"])


if __name__ == "__main__":
    raise SystemExit(main())
