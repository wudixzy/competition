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
QUICK_HIGH_MARGIN_THRESHOLD_NATS = 0.1
QUICK_SEVERE_MEAN_NLL_REGRESSION_NATS = 0.05


def _cluster_bootstrap_upper(clusters: list[list[float]]) -> float:
    """Hierarchical bootstrap with request/length as the independent unit."""
    if not clusters or any(not cluster for cluster in clusters):
        raise ValueError("paired NLL cluster sample is empty")
    generator = random.Random(BOOTSTRAP_SEED)
    cluster_count = len(clusters)
    means = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled_cluster_means = []
        for _ in range(cluster_count):
            cluster = clusters[generator.randrange(cluster_count)]
            sampled_cluster_means.append(sum(
                cluster[generator.randrange(len(cluster))]
                for _ in range(len(cluster))) / len(cluster))
        means.append(sum(sampled_cluster_means) / cluster_count)
    means.sort()
    return means[min(len(means) - 1, int(0.95 * len(means)))]


def _v2_arm_reasons(value: Any, label: str) -> list[str]:
    reasons = legacy._validate_arm(
        value, "candidate" if label == "candidate" else "control", label)
    if reasons or not isinstance(value, dict):
        return reasons
    for field in ("source_revision", "runtime_identity", "instance",
                  "model_path"):
        if not isinstance(value.get(field), str) or not value[field]:
            reasons.append(f"{label}: {field} is empty")
    cases = value.get("cases")
    if isinstance(cases, list):
        for case in cases:
            if (not isinstance(case, dict)
                    or case.get("cached_tokens") != 0
                    or case.get("request") != {
                        "http_status": 200,
                        "stream": False,
                        "response_complete": True,
                        "usage_complete": True,
                        "finish_reason": "length",
                    }
                    or len(case.get("positions") or []) != 64):
                reasons.append(
                    f"{label}: request/cache/64-position contract differs")
                break
    return reasons


def _pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_cases = legacy._case_map(left)
    right_cases = legacy._case_map(right)
    if set(left_cases) != set(right_cases):
        raise ValueError("teacher-forced case identities differ")
    teacher_deltas: list[float] = []
    shared_deltas: list[float] = []
    nll_clusters: list[list[float]] = []
    nll_by_length: list[dict[str, Any]] = []
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
        case_nll: list[float] = []
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
            case_nll.append(left_values[teacher] - right_values[teacher])
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
        nll_clusters.append(case_nll)
        nll_by_length.append({
            "prompt_tokens": left_case["prompt_tokens"],
            "sampled_positions": len(case_nll),
            "mean_nll_difference": sum(case_nll) / len(case_nll),
        })
    flips = len(flip_margins)
    cluster_means = [sum(cluster) / len(cluster) for cluster in nll_clusters]
    return {
        "sampled_positions": position_ordinal,
        "top1_agreement": top1_matches / position_ordinal,
        "mutual_topk_coverage": mutually_covered / flips if flips else 1.0,
        "teacher_token_logprob_delta": legacy._percentile(teacher_deltas, 0.99),
        "shared_token_logprob_delta": legacy._percentile(shared_deltas, 0.99),
        "paired_nll_difference": sum(cluster_means) / len(cluster_means),
        "paired_nll_one_sided_95_upper_ci": _cluster_bootstrap_upper(
            nll_clusters),
        "nll_by_length": nll_by_length,
        "nll_cluster_count": len(nll_clusters),
        "first_divergent_token": first_divergent if first_divergent is not None else -1,
        "baseline_top1_margin": legacy._percentile(margins, 0.99),
        "flip_margins": flip_margins,
    }


def quick_screen(control_a: Any, candidate: Any) -> dict[str, Any]:
    """Conservative two-arm screen used before paying for control B."""
    reasons = []
    reasons.extend(_v2_arm_reasons(control_a, "control_a"))
    reasons.extend(_v2_arm_reasons(candidate, "candidate"))
    if not reasons:
        reasons.extend(legacy._identity_reasons(
            control_a, candidate, "candidate"))
        observed = tuple(sorted(
            case["prompt_tokens"] for case in control_a["cases"]))
        if observed != TARGETS:
            reasons.append("short teacher-forced target population differs")
    if reasons:
        return {
            "schema": "bi100-m1-178-teacher-forced-quick-screen-v1",
            "version": 1,
            "status": "invalid",
            "classification": "invalid_evidence",
            "reasons": reasons,
            "control_b_authorized": False,
        }
    try:
        observed = _pair(control_a, candidate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return {
            "schema": "bi100-m1-178-teacher-forced-quick-screen-v1",
            "version": 1,
            "status": "invalid",
            "classification": "invalid_evidence",
            "reasons": [str(exc)],
            "control_b_authorized": False,
        }
    high_margin_flips = sum(
        margin > QUICK_HIGH_MARGIN_THRESHOLD_NATS
        for margin in observed["flip_margins"])
    severe_nll = (
        observed["paired_nll_difference"]
        > QUICK_SEVERE_MEAN_NLL_REGRESSION_NATS)
    if high_margin_flips or severe_nll:
        status = "inconclusive"
        classification = "distribution_drift_requires_adjudication"
        reasons = []
        if high_margin_flips:
            reasons.append("absolute-floor high-margin flip observed")
        if severe_nll:
            reasons.append("mean NLL regression exceeds quick-stop threshold")
    else:
        status = "pass"
        classification = "quick_screen_pass"
        reasons = []
    metrics = {key: value for key, value in observed.items()
               if key != "flip_margins"}
    metrics["high_margin_flips"] = high_margin_flips
    return {
        "schema": "bi100-m1-178-teacher-forced-quick-screen-v1",
        "version": 1,
        "status": status,
        "classification": classification,
        "reasons": reasons,
        "candidate": metrics,
        "thresholds": {
            "high_margin_nats": QUICK_HIGH_MARGIN_THRESHOLD_NATS,
            "severe_mean_nll_regression_nats": (
                QUICK_SEVERE_MEAN_NLL_REGRESSION_NATS),
        },
        "bootstrap": {
            "method": "hierarchical_request_cluster_within_position",
            "cluster_unit": "teacher_forced_request_length",
            "cluster_count": len(TARGETS),
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
        },
        "control_b_authorized": status == "pass",
    }


def compare(
    control_a: Any,
    control_b: Any,
    candidate: Any,
    contract: Any,
) -> dict[str, Any]:
    reasons: list[str] = []
    reasons.extend(_v2_arm_reasons(control_a, "control_a"))
    reasons.extend(_v2_arm_reasons(control_b, "control_b"))
    reasons.extend(_v2_arm_reasons(candidate, "candidate"))
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
        "runtime_identity": control_a["runtime_identity"],
        "instance": control_a["instance"],
        "model_path": control_a["model_path"],
        "targets": list(TARGETS),
        "sampled_positions": observed["sampled_positions"],
        "workload_identity": {
            "case_ids": [case["id"] for case in control_a["cases"]],
            "prompt_tokens": [case["prompt_tokens"]
                              for case in control_a["cases"]],
            "sampled_positions_per_case": [len(case["positions"])
                                           for case in control_a["cases"]],
        },
        "arm_binding": {
            "control_a": "control",
            "control_b": "control",
            "candidate": "candidate",
        },
        "aa": {
            **aa_metrics,
            "sampled_positions": aa["sampled_positions"],
            "top1_agreement": aa["top1_agreement"],
            "mutual_topk_coverage": aa["mutual_topk_coverage"],
            "teacher_token_logprob_delta": aa["teacher_token_logprob_delta"],
            "paired_nll_difference": aa["paired_nll_difference"],
            "nll_by_length": aa["nll_by_length"],
            "nll_cluster_count": aa["nll_cluster_count"],
        },
        "candidate": candidate_metrics,
        "decision": decision,
        "bootstrap": {
            "method": "hierarchical_request_cluster_within_position",
            "cluster_unit": "teacher_forced_request_length",
            "cluster_count": len(TARGETS),
            "positions_per_cluster": 64,
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
