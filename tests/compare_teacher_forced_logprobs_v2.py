#!/usr/bin/env python3
"""A/A-calibrated v2 teacher-forced distribution comparison."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import compare_teacher_forced_logprobs as legacy
import validate_bi100_metrics_contract as metrics_contract


SCHEMA = "bi100-teacher-forced-distribution-v2"
TARGETS = (4096, 16384, 32768, 65536)
BOOTSTRAP_SAMPLES = 20000
BOOTSTRAP_SEED = 20260904


def _bootstrap_upper(values: list[float]) -> float:
    if not values:
        raise ValueError("paired NLL sample is empty")
    generator = random.Random(BOOTSTRAP_SEED)
    count = len(values)
    means = sorted(
        sum(values[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    return means[min(len(means) - 1, int(0.95 * len(means)))]


def _pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_cases = legacy._case_map(left)
    right_cases = legacy._case_map(right)
    if set(left_cases) != set(right_cases):
        raise ValueError("teacher-forced case identities differ")
    teacher_deltas: list[float] = []
    shared_deltas: list[float] = []
    nll_differences: list[float] = []
    margins: list[float] = []
    flip_margins: list[float] = []
    top1_matches = 0
    mutually_covered = 0
    first_divergent: int | None = None
    position_ordinal = 0
    for case_id in sorted(left_cases, key=lambda item: left_cases[item]["prompt_tokens"]):
        left_case = left_cases[case_id]
        right_case = right_cases[case_id]
        if left_case["prompt_tokens"] != right_case["prompt_tokens"]:
            raise ValueError("teacher-forced prompt lengths differ")
        left_positions = legacy._position_map(left_case)
        right_positions = legacy._position_map(right_case)
        if set(left_positions) != set(right_positions):
            raise ValueError("teacher-forced sampled positions differ")
        for position in sorted(left_positions):
            lhs = left_positions[position]
            rhs = right_positions[position]
            if lhs["actual_token_key"] != rhs["actual_token_key"]:
                raise ValueError("teacher token identity differs")
            left_values = legacy._logprob_map(lhs)
            right_values = legacy._logprob_map(rhs)
            teacher = lhs["actual_token_key"]
            if teacher not in right_values:
                raise ValueError("candidate omitted teacher token")
            teacher_deltas.append(abs(right_values[teacher] - left_values[teacher]))
            nll_differences.append(left_values[teacher] - right_values[teacher])
            shared = left_values.keys() & right_values.keys()
            if not shared:
                raise ValueError("top-k observations have no shared token")
            shared_deltas.extend(
                abs(right_values[token] - left_values[token]) for token in shared)
            left_top = lhs["top_logprobs"][0]["token_key"]
            right_top = rhs["top_logprobs"][0]["token_key"]
            margin = (
                float(lhs["top_logprobs"][0]["logprob"])
                - float(lhs["top_logprobs"][1]["logprob"])
            )
            margins.append(margin)
            if left_top == right_top:
                top1_matches += 1
            else:
                if first_divergent is None:
                    first_divergent = position_ordinal
                flip_margins.append(margin)
                if left_top in right_values and right_top in left_values:
                    mutually_covered += 1
            position_ordinal += 1
    flips = len(flip_margins)
    return {
        "sampled_positions": position_ordinal,
        "top1_agreement": top1_matches / position_ordinal,
        "mutual_topk_coverage": mutually_covered / flips if flips else 1.0,
        "teacher_token_logprob_delta": legacy._percentile(teacher_deltas, 0.99),
        "shared_token_logprob_delta": legacy._percentile(shared_deltas, 0.99),
        "paired_nll_difference": sum(nll_differences) / len(nll_differences),
        "paired_nll_one_sided_95_upper_ci": _bootstrap_upper(nll_differences),
        "first_divergent_token": first_divergent if first_divergent is not None else -1,
        "baseline_top1_margin": legacy._percentile(margins, 0.99),
        "flip_margins": flip_margins,
    }


def compare(
    control_a: Any,
    control_b: Any,
    candidate: Any,
    contract: Any,
) -> dict[str, Any]:
    reasons: list[str] = []
    reasons.extend(legacy._validate_arm(control_a, "control", "control_a"))
    reasons.extend(legacy._validate_arm(control_b, "control", "control_b"))
    reasons.extend(legacy._validate_arm(candidate, "candidate", "candidate"))
    if not reasons:
        reasons.extend(legacy._identity_reasons(
            control_a, control_b, "control-repeat"))
        reasons.extend(legacy._identity_reasons(
            control_a, candidate, "candidate"))
        observed = tuple(sorted(case["prompt_tokens"] for case in control_a["cases"]))
        if observed != TARGETS:
            reasons.append("short teacher-forced target population differs")
    try:
        metrics_contract.validate_contract(contract, "layered")
    except metrics_contract.ContractError as exc:
        reasons.append(str(exc))
    if reasons:
        return {
            "schema": SCHEMA,
            "version": 2,
            "status": "invalid",
            "classification": "invalid",
            "reasons": reasons,
        }
    try:
        aa = _pair(control_a, control_b)
        observed = _pair(control_a, candidate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return {
            "schema": SCHEMA,
            "version": 2,
            "status": "invalid",
            "classification": "invalid",
            "reasons": [str(exc)],
        }
    rules = contract["teacher_forced_distribution"]
    high_margin_threshold = max(
        rules["high_margin_threshold"]["absolute_floor_nats"],
        rules["high_margin_threshold"][
            "aa_p99_shared_logprob_delta_multiplier"]
        * aa["shared_token_logprob_delta"],
    )
    candidate_metrics = {
        key: value for key, value in observed.items() if key != "flip_margins"
    }
    candidate_metrics["high_margin_flips"] = sum(
        margin > high_margin_threshold for margin in observed["flip_margins"])
    aa_metrics = {
        "shared_logprob_delta_p99": aa["shared_token_logprob_delta"],
        "paired_nll_upper_ci": aa["paired_nll_one_sided_95_upper_ci"],
    }
    decision = metrics_contract.classify_distribution(
        candidate_metrics, aa_metrics, contract)
    return {
        "schema": SCHEMA,
        "version": 2,
        "status": decision["status"],
        "classification": decision["classification"],
        "source_revision": control_a["source_revision"],
        "targets": list(TARGETS),
        "sampled_positions": observed["sampled_positions"],
        "aa": {
            **aa_metrics,
            "top1_agreement": aa["top1_agreement"],
            "mutual_topk_coverage": aa["mutual_topk_coverage"],
            "teacher_token_logprob_delta": aa["teacher_token_logprob_delta"],
            "paired_nll_difference": aa["paired_nll_difference"],
        },
        "candidate": candidate_metrics,
        "decision": decision,
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
        },
        "privacy": {
            "contains_token_keys": False,
            "contains_token_ids": False,
            "contains_prompts": False,
            "contains_model_outputs": False,
            "contains_credentials": False,
        },
        "promotion_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("control_a", type=Path)
    parser.add_argument("control_b", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        json.loads(args.control_a.read_text(encoding="utf-8")),
        json.loads(args.control_b.read_text(encoding="utf-8")),
        json.loads(args.candidate.read_text(encoding="utf-8")),
        json.loads(args.contract.read_text(encoding="utf-8")),
    )
    legacy._atomic_write(args.out, result)
    return {"pass": 0, "inconclusive": 0, "invalid": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
