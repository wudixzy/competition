#!/usr/bin/env python3
"""Compare M1-109/M1-162/M1-109 fixed-strata teacher-forced evidence."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import compare_teacher_forced_logprobs as legacy
import validate_bi100_metrics_contract as metrics_contract


SCHEMA = "bi100-m1-179-incremental-distribution-v1"
OBSERVATION_SCHEMA = "bi100-teacher-forced-topk-observation-v2"
TARGETS = (4096, 16384, 32768, 65536)
POSITIONS_PER_TARGET = 64
BOOTSTRAP_SAMPLES = 20000
BOOTSTRAP_SEED = 20260905
EXPECTED_VARIANTS = {
    "control_a": "m1_109_fp32_qk",
    "candidate": "m1_162_fp16_qk",
    "control_b": "m1_109_fp32_qk",
}


def _observation_reasons(value: Any, label: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{label}: observation root must be an object"]
    expected_variant = EXPECTED_VARIANTS[label]
    base = copy.deepcopy(value)
    variant = base.pop("fused_variant", None)
    extension = base.pop("extension_identity", None)
    base["schema"] = legacy.OBSERVATION_SCHEMA
    base["version"] = legacy.VERSION
    expected_mode = "candidate" if label == "candidate" else "control"
    reasons = legacy._validate_arm(base, expected_mode, label)
    if (value.get("schema") != OBSERVATION_SCHEMA
            or value.get("version") != 2
            or variant != expected_variant):
        reasons.append(f"{label}: fused variant binding is invalid")
    if (not isinstance(extension, dict)
            or set(extension) != {
                "module_path", "runtime_loaded_module", "sha256"}
            or not isinstance(extension.get("module_path"), str)
            or not extension["module_path"]
            or extension.get("runtime_loaded_module") != extension.get(
                "module_path")
            or not legacy._is_digest(extension.get("sha256"))):
        reasons.append(f"{label}: extension runtime identity is invalid")
    optimization = value.get("optimization")
    if (not isinstance(optimization, dict)
            or optimization.get("fused_prefill") != "1"):
        reasons.append(f"{label}: fused prefill was not enabled")
    cases = value.get("cases")
    if isinstance(cases, list):
        observed_targets = tuple(sorted(
            case.get("prompt_tokens") for case in cases
            if isinstance(case, dict)))
        if observed_targets != TARGETS:
            reasons.append(f"{label}: fixed target population differs")
        if any(case.get("cached_tokens") != 0
               or len(case.get("positions") or []) != POSITIONS_PER_TARGET
               for case in cases if isinstance(case, dict)):
            reasons.append(f"{label}: cache/position contract differs")
    for field in ("source_revision", "runtime_identity", "instance",
                  "model_path"):
        if not isinstance(value.get(field), str) or not value[field]:
            reasons.append(f"{label}: {field} is empty")
    return reasons


def _cross_arm_reasons(arms: dict[str, dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    reference = arms["control_a"]
    for label in ("candidate", "control_b"):
        value = arms[label]
        for field in ("source_revision", "runtime_identity", "instance",
                      "model_path", "gpu_count", "tensor_parallel_size",
                      "max_model_len", "top_k"):
            if not reference.get(field) or value.get(field) != reference.get(field):
                reasons.append(f"{label}: cross-arm {field} differs")
        left = dict(reference.get("optimization") or {})
        right = dict(value.get("optimization") or {})
        left.pop("fused_prefill", None)
        right.pop("fused_prefill", None)
        if left != right:
            reasons.append(f"{label}: optimization differs beyond variant")
    return reasons


def _percentile(values: list[float], quantile: float) -> float:
    return legacy._percentile(values, quantile)


def _fixed_strata_upper(clusters: list[list[float]], seed: int) -> float:
    """Resample positions within fixed lengths; not a run-to-run interval."""
    if len(clusters) != len(TARGETS) or any(not cluster for cluster in clusters):
        raise ValueError("fixed-strata position samples are incomplete")
    generator = random.Random(seed)
    sampled_means = []
    for _ in range(BOOTSTRAP_SAMPLES):
        stratum_means = []
        for cluster in clusters:
            stratum_means.append(sum(
                cluster[generator.randrange(len(cluster))]
                for _ in range(len(cluster))) / len(cluster))
        sampled_means.append(sum(stratum_means) / len(stratum_means))
    sampled_means.sort()
    return sampled_means[min(
        len(sampled_means) - 1,
        math.ceil(0.95 * len(sampled_means)) - 1,
    )]


def _stratum_upper(values: list[float], seed: int) -> float:
    generator = random.Random(seed)
    means = [sum(values[generator.randrange(len(values))]
                 for _ in range(len(values))) / len(values)
             for _ in range(BOOTSTRAP_SAMPLES)]
    means.sort()
    return means[min(len(means) - 1,
                     math.ceil(0.95 * len(means)) - 1)]


def _pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_cases = legacy._case_map(left)
    right_cases = legacy._case_map(right)
    if set(left_cases) != set(right_cases):
        raise ValueError("teacher-forced case identities differ")
    teacher_deltas: list[float] = []
    shared_deltas: list[float] = []
    clusters: list[list[float]] = []
    by_length: list[dict[str, Any]] = []
    flip_details: list[dict[str, Any]] = []
    margins: list[float] = []
    top1_matches = 0
    mutually_covered = 0
    ordinal = 0
    first_divergent: dict[str, Any] | None = None
    for case_id in sorted(left_cases,
                          key=lambda key: left_cases[key]["prompt_tokens"]):
        lhs_case = left_cases[case_id]
        rhs_case = right_cases[case_id]
        prompt_tokens = lhs_case["prompt_tokens"]
        if prompt_tokens != rhs_case["prompt_tokens"]:
            raise ValueError("teacher-forced prompt lengths differ")
        lhs_positions = legacy._position_map(lhs_case)
        rhs_positions = legacy._position_map(rhs_case)
        if set(lhs_positions) != set(rhs_positions):
            raise ValueError("teacher-forced sampled positions differ")
        nll_values: list[float] = []
        for position in sorted(lhs_positions):
            lhs = lhs_positions[position]
            rhs = rhs_positions[position]
            if lhs["actual_token_key"] != rhs["actual_token_key"]:
                raise ValueError("teacher token identity differs")
            left_values = legacy._logprob_map(lhs)
            right_values = legacy._logprob_map(rhs)
            teacher = lhs["actual_token_key"]
            if teacher not in right_values:
                raise ValueError("right arm omitted teacher token")
            teacher_deltas.append(abs(
                right_values[teacher] - left_values[teacher]))
            nll_values.append(left_values[teacher] - right_values[teacher])
            shared = left_values.keys() & right_values.keys()
            if not shared:
                raise ValueError("top-k observations have no shared token")
            shared_deltas.extend(abs(right_values[token] - left_values[token])
                                 for token in shared)
            lhs_top = lhs["top_logprobs"][0]["token_key"]
            rhs_top = rhs["top_logprobs"][0]["token_key"]
            margin = (float(lhs["top_logprobs"][0]["logprob"])
                      - float(lhs["top_logprobs"][1]["logprob"]))
            margins.append(margin)
            if lhs_top == rhs_top:
                top1_matches += 1
            else:
                detail = {
                    "prompt_tokens": prompt_tokens,
                    "position": position,
                    "sample_ordinal": ordinal,
                    "control_margin_nats": margin,
                }
                flip_details.append(detail)
                if first_divergent is None:
                    first_divergent = detail
                if lhs_top in right_values and rhs_top in left_values:
                    mutually_covered += 1
            ordinal += 1
        clusters.append(nll_values)
        by_length.append({
            "prompt_tokens": prompt_tokens,
            "sampled_positions": len(nll_values),
            "mean_nll_difference_nats": sum(nll_values) / len(nll_values),
            "position_sampling_one_sided_95_upper_nats": _stratum_upper(
                nll_values, BOOTSTRAP_SEED + prompt_tokens),
        })
    flips = len(flip_details)
    cluster_means = [sum(values) / len(values) for values in clusters]
    return {
        "sampled_positions": ordinal,
        "top1_agreement": top1_matches / ordinal,
        "top1_flip_count": flips,
        "mutual_topk_coverage": mutually_covered / flips if flips else 1.0,
        "teacher_token_abs_logprob_delta_p99_nats": _percentile(
            teacher_deltas, 0.99),
        "shared_token_abs_logprob_delta_p99_nats": _percentile(
            shared_deltas, 0.99),
        "paired_mean_nll_difference_nats": sum(cluster_means) / len(
            cluster_means),
        "position_sampling_one_sided_95_upper_nats": _fixed_strata_upper(
            clusters, BOOTSTRAP_SEED),
        "nll_by_length": by_length,
        "first_divergent_position": first_divergent,
        "baseline_top1_margin_p99_nats": _percentile(margins, 0.99),
        "flip_details": flip_details,
        "all_logprobs_finite": True,
    }


def compare(control_a: Any, control_b: Any, candidate: Any,
            contract: Any) -> dict[str, Any]:
    arms = {"control_a": control_a, "candidate": candidate,
            "control_b": control_b}
    reasons: list[str] = []
    for label, value in arms.items():
        reasons.extend(_observation_reasons(value, label))
    if not reasons:
        reasons.extend(_cross_arm_reasons(arms))
    try:
        metrics_contract.validate_contract(contract, "layered")
    except metrics_contract.ContractError as exc:
        reasons.append(str(exc))
    if reasons:
        return {"schema": SCHEMA, "version": 1, "status": "invalid",
                "classification": "invalid_evidence", "reasons": reasons,
                "promotion_authorized": False}
    try:
        aa = _pair(control_a, control_b)
        incremental = _pair(control_a, candidate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return {"schema": SCHEMA, "version": 1, "status": "invalid",
                "classification": "invalid_evidence", "reasons": [str(exc)],
                "promotion_authorized": False}

    rules = contract["teacher_forced_distribution"]
    high_margin_threshold = max(
        rules["high_margin_threshold"]["absolute_floor_nats"],
        rules["high_margin_threshold"][
            "aa_p99_shared_logprob_delta_multiplier"]
        * aa["shared_token_abs_logprob_delta_p99_nats"],
    )
    aa_upper = aa["position_sampling_one_sided_95_upper_nats"]
    nll_threshold = max(
        rules["nll_regression_upper_ci"]["absolute_floor_nats"],
        rules["nll_regression_upper_ci"]["aa_upper_ci_multiplier"]
        * max(0.0, aa_upper),
    )
    aa_high_margin = sum(
        item["control_margin_nats"] > 0.1 for item in aa["flip_details"])
    candidate_high_margin = sum(
        item["control_margin_nats"] > high_margin_threshold
        for item in incremental["flip_details"])
    aa_by_length = {item["prompt_tokens"]: item
                    for item in aa["nll_by_length"]}
    local_nll_regressions = []
    for item in incremental["nll_by_length"]:
        aa_item = aa_by_length[item["prompt_tokens"]]
        local_threshold = max(
            nll_threshold,
            2.0 * max(0.0, aa_item[
                "position_sampling_one_sided_95_upper_nats"]),
        )
        if item["mean_nll_difference_nats"] > local_threshold:
            local_nll_regressions.append({
                **item, "calibrated_threshold_nats": local_threshold})
    aa_large_noise = (
        aa_high_margin > 0
        or aa["shared_token_abs_logprob_delta_p99_nats"] >= 1.0
        or any(abs(item["mean_nll_difference_nats"]) >= 1.0
               for item in aa["nll_by_length"])
    )
    candidate_exceeds = (
        candidate_high_margin > 0
        or incremental["position_sampling_one_sided_95_upper_nats"]
        > nll_threshold
        or bool(local_nll_regressions)
    )
    if aa_large_noise:
        status = "inconclusive"
        classification = "baseline_nondeterminism_or_measurement_noise"
    elif candidate_exceeds:
        status = "inconclusive"
        classification = "incremental_fp16_qk_distribution_drift"
    else:
        status = "pass"
        classification = "incremental_fp16_qk_within_aa_envelope"

    incremental["high_margin_flip_count"] = candidate_high_margin
    aa["absolute_floor_high_margin_flip_count"] = aa_high_margin
    return {
        "schema": SCHEMA,
        "version": 1,
        "status": status,
        "classification": classification,
        "source_revision": control_a["source_revision"],
        "runtime_identity": control_a["runtime_identity"],
        "instance": control_a["instance"],
        "model_path": control_a["model_path"],
        "targets": list(TARGETS),
        "positions_per_target": POSITIONS_PER_TARGET,
        "arm_binding": EXPECTED_VARIANTS,
        "extension_identity": {
            label: value["extension_identity"] for label, value in arms.items()
        },
        "aa": aa,
        "incremental": incremental,
        "thresholds": {
            "high_margin_nats": high_margin_threshold,
            "nll_regression_nats": nll_threshold,
            "aa_shared_logprob_p99_multiplier": 4.0,
            "aa_nll_upper_multiplier": 2.0,
        },
        "decision_basis": {
            "aa_envelope_primary": True,
            "high_margin_flips_primary": True,
            "per_length_nll_primary": True,
            "nonfinite_primary": True,
            "variant_identity_primary": True,
            "aggregate_cancellation_allowed": False,
            "local_nll_regression_lengths": [
                item["prompt_tokens"] for item in local_nll_regressions],
        },
        "bootstrap": {
            "method": "fixed_strata_within_position_resampling",
            "interpretation": "position_sampling_diagnostic",
            "not_a_run_to_run_confidence_interval": True,
            "fixed_strata": list(TARGETS),
            "requests_per_arm_per_stratum": 1,
            "positions_per_stratum": POSITIONS_PER_TARGET,
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
    parser.add_argument("candidate", type=Path)
    parser.add_argument("control_b", type=Path)
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
    return {"pass": 0, "inconclusive": 3, "invalid": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
