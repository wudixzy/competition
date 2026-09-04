#!/usr/bin/env python3
"""Compare privacy-private M1-180 arms and emit a repository-safe summary."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Any

import compare_m1_179_teacher_forced as distribution


SCHEMA = "bi100-m1-180-adjudication-v1"
VERSION = 1
ARMS = {
    "fused_off": "fused_off",
    "m1_109": "m1_109_fp32_qk",
    "m1_162": "m1_162_fp16_qk",
}
STRATA = ("code", "reasoning", "tools", "structured_output",
          "multimodal", "long_context")
BOOTSTRAP_SAMPLES = 20000
BOOTSTRAP_SEED = 20260905
DEVELOPMENT_MARGIN = -0.05


def _finite_number(value: Any, *, nonnegative: bool = False) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value))
            and (not nonnegative or float(value) >= 0))


def _all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return True


def _case_map(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["case_id"]: item
            for item in (value.get("capability") or {}).get("cases") or []
            if isinstance(item, dict) and isinstance(item.get("case_id"), str)}


def arm_reasons(value: Any, label: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{label}: arm is not an object"]
    reasons = []
    capability = value.get("capability")
    cases = capability.get("cases") if isinstance(capability, dict) else None
    if (value.get("schema") != "bi100-m1-180-arm-observation-v1"
            or value.get("version") != 1 or value.get("arm") != label
            or value.get("algorithm_variant") != ARMS[label]
            or not all(isinstance(value.get(field), str) and value[field]
                       for field in ("source_revision", "runtime_identity",
                                     "instance", "model_path", "workload_id"))
            or not isinstance(cases, list)
            or capability.get("strata") != list(STRATA)
            or capability.get("smoke_per_stratum") != 4
            or capability.get("full_per_stratum") != 10):
        reasons.append(f"{label}: arm identity/capability schema differs")
        return reasons
    expected = 24 if (label == "m1_162"
                      and capability.get("extended_triggered") is False) else 60
    if (len(cases) != expected
            or len(_case_map(value)) != expected
            or capability.get("smoke_completed") != 24
            or capability.get("complete") is not (expected == 60)):
        reasons.append(f"{label}: capability population is incomplete")
    expected_ids = {
        f"{stratum}_{ordinal:02d}"
        for stratum in STRATA for ordinal in range(expected // len(STRATA))
    }
    if set(_case_map(value)) != expected_ids:
        reasons.append(f"{label}: capability case identities differ")
    for case in cases:
        if (case.get("stratum") not in STRATA
                or not isinstance(case.get("pass"), bool)
                or case.get("http_status") != 200
                or case.get("response_contract_complete") is not True
                or case.get("all_values_finite") is not True
                or not isinstance(case.get("finish_reason"), str)
                or not _finite_number(case.get("elapsed_s"), nonnegative=True)
                or not isinstance(case.get("prompt_tokens"), int)
                or not isinstance(case.get("completion_tokens"), int)
                or not isinstance(case.get("cached_tokens"), int)):
            reasons.append(f"{label}: invalid case {case.get('case_id')}")
    population = value.get("request_population")
    if (not isinstance(population, dict)
            or population.get("attempted") != population.get("completed")
            or population.get("failed") != 0
            or not isinstance(population.get("completed"), int)):
        reasons.append(f"{label}: request population differs")
    teacher = value.get("teacher_forced")
    candidate_stopped = (label == "m1_162"
                         and capability.get("extended_triggered") is False)
    if candidate_stopped:
        if teacher is not None:
            reasons.append("m1_162: teacher-forced ran after capability stop")
    elif not isinstance(teacher, dict):
        reasons.append(f"{label}: teacher-forced evidence missing")
    else:
        expected_schema = ("bi100-teacher-forced-topk-observation-v1"
                           if label == "fused_off"
                           else "bi100-teacher-forced-topk-observation-v2")
        expected_version = 1 if label == "fused_off" else 2
        teacher_cases = teacher.get("cases")
        if (teacher.get("schema") != expected_schema
                or teacher.get("version") != expected_version
                or teacher.get("source_revision") != value["source_revision"]
                or teacher.get("runtime_identity") != value["runtime_identity"]
                or teacher.get("instance") != value["instance"]
                or teacher.get("model_path") != value["model_path"]
                or teacher.get("optimization", {}).get("fused_prefill")
                != ("0" if label == "fused_off" else "1")
                or not isinstance(teacher_cases, list)
                or len(teacher_cases) != 4
                or not _all_finite(teacher)):
            reasons.append(f"{label}: teacher-forced identity differs")
        elif tuple(sorted(case.get("prompt_tokens")
                          for case in teacher_cases)) != distribution.TARGETS:
            reasons.append(f"{label}: teacher-forced targets differ")
        elif any(case.get("cached_tokens") != 0
                 or len(case.get("positions") or []) != 64
                 for case in teacher_cases):
            reasons.append(f"{label}: teacher-forced cache/positions differ")
        if label != "fused_off" and teacher.get("fused_variant") != ARMS[label]:
            reasons.append(f"{label}: teacher-forced variant differs")
    return reasons


def cross_arm_reasons(arms: dict[str, dict[str, Any]]) -> list[str]:
    reasons = []
    reference = arms["fused_off"]
    for label in ("m1_109", "m1_162"):
        value = arms[label]
        for field in ("source_revision", "runtime_identity", "instance",
                      "model_path", "workload_id"):
            if value.get(field) != reference.get(field):
                reasons.append(f"{label}: cross-arm {field} differs")
        common = set(_case_map(reference)) & set(_case_map(value))
        if not common:
            reasons.append(f"{label}: no paired capability population")
    return reasons


def critical_smoke_baseline_only(reference: dict[str, Any],
                                 candidate: dict[str, Any]) -> list[str]:
    left, right = _case_map(reference), _case_map(candidate)
    return sorted(case_id for case_id in left.keys() & right.keys()
                  if left[case_id].get("stage") == "smoke"
                  and left[case_id].get("pass") is True
                  and right[case_id].get("pass") is False)


def _bootstrap_lower(differences: list[int], seed: int) -> float:
    generator = random.Random(seed)
    means = [sum(differences[generator.randrange(len(differences))]
                 for _ in differences) / len(differences)
             for _ in range(BOOTSTRAP_SAMPLES)]
    means.sort()
    return means[max(0, math.floor(0.05 * len(means)))]


def _mcnemar(baseline_only: int, candidate_only: int) -> dict[str, Any]:
    discordant = baseline_only + candidate_only
    if discordant == 0:
        pvalue = 1.0
    else:
        tail = sum(math.comb(discordant, index)
                   for index in range(min(baseline_only, candidate_only) + 1))
        pvalue = min(1.0, 2.0 * tail / (2 ** discordant))
    return {"discordant_pairs": discordant, "two_sided_exact_p": pvalue}


def capability_pair(left: dict[str, Any], right: dict[str, Any],
                    name: str) -> dict[str, Any]:
    left_map, right_map = _case_map(left), _case_map(right)
    common = sorted(left_map.keys() & right_map.keys())
    strata = {}
    aggregate = {"both_pass": 0, "baseline_only": 0,
                 "candidate_only": 0, "both_fail": 0}
    differences = []
    for stratum in STRATA:
        counts = {"both_pass": 0, "baseline_only": 0,
                  "candidate_only": 0, "both_fail": 0}
        stratum_differences = []
        for case_id in common:
            if left_map[case_id]["stratum"] != stratum:
                continue
            lhs, rhs = left_map[case_id]["pass"], right_map[case_id]["pass"]
            key = ("both_pass" if lhs and rhs else
                   "baseline_only" if lhs else
                   "candidate_only" if rhs else "both_fail")
            counts[key] += 1
            aggregate[key] += 1
            difference = int(rhs) - int(lhs)
            differences.append(difference)
            stratum_differences.append(difference)
        sample_count = sum(counts.values())
        strata[stratum] = {
            "sample_count": sample_count, **counts,
            "paired_pass_rate_difference": (
                sum(stratum_differences) / sample_count if sample_count else None),
            "paired_bootstrap_one_sided_95_lower": (
                _bootstrap_lower(stratum_differences,
                                 BOOTSTRAP_SEED + STRATA.index(stratum))
                if stratum_differences else None),
            "exact_mcnemar": _mcnemar(
                counts["baseline_only"], counts["candidate_only"]),
            "underpowered_for_stratum_promotion": sample_count < 59,
        }
    sample_count = len(differences)
    lower = (_bootstrap_lower(differences, BOOTSTRAP_SEED)
             if differences else float("nan"))
    critical = aggregate["baseline_only"] > 0
    development_status = (
        "fail" if critical else
        "pass" if sample_count >= 60 and lower > DEVELOPMENT_MARGIN
        else "inconclusive")
    return {
        "comparison": name, "sample_count": sample_count, **aggregate,
        "paired_pass_rate_difference": sum(differences) / sample_count,
        "paired_bootstrap_one_sided_95_lower": lower,
        "development_margin": DEVELOPMENT_MARGIN,
        "development_screen_status": development_status,
        "exact_mcnemar": _mcnemar(
            aggregate["baseline_only"], aggregate["candidate_only"]),
        "strata": strata,
        "all_strata_underpowered_for_promotion": all(
            item["underpowered_for_stratum_promotion"]
            for item in strata.values()),
    }


def _distribution_pair(left: dict[str, Any], right: dict[str, Any],
                       name: str, aa: dict[str, Any]) -> dict[str, Any]:
    result = distribution._pair(left["teacher_forced"], right["teacher_forced"])
    high_threshold = max(0.1, 4 * aa[
        "shared_token_abs_logprob_delta_p99_nats"])
    nll_threshold = max(0.01, 2 * max(
        0.0, aa["position_sampling_one_sided_95_upper_nats"]))
    result["high_margin_flip_count"] = sum(
        item["control_margin_nats"] > high_threshold
        for item in result["flip_details"])
    result["comparison"] = name
    result["thresholds"] = {
        "high_margin_nats": high_threshold,
        "nll_regression_nats": nll_threshold,
    }
    drift = (result["high_margin_flip_count"] > 0
             or result["position_sampling_one_sided_95_upper_nats"]
             > nll_threshold
             or any(item["mean_nll_difference_nats"] > nll_threshold
                    for item in result["nll_by_length"]))
    result["status"] = "inconclusive" if drift else "pass"
    result["classification"] = (
        "distribution_drift_requires_adjudication" if drift
        else "within_reused_aa_envelope")
    return result


def _performance(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    lhs, rhs = left.get("performance"), right.get("performance")
    if not isinstance(lhs, dict) or not isinstance(rhs, dict):
        return {"status": "not_run", "reason": "candidate capability stop"}
    left_cases = lhs.get("cases") or []
    right_cases = rhs.get("cases") or []
    if len(left_cases) != 6 or len(right_cases) != 6:
        return {"status": "invalid", "reason": "timing population incomplete"}
    rows = []
    gains = []
    for left_case, right_case in zip(left_cases, right_cases):
        if (left_case.get("target_prompt_tokens")
                != right_case.get("target_prompt_tokens")
                or left_case.get("repetition") != right_case.get("repetition")):
            return {"status": "invalid", "reason": "timing identity differs"}
        left_ttft = left_case["response"]["ttft_s"]
        right_ttft = right_case["response"]["ttft_s"]
        if not (_finite_number(left_ttft) and _finite_number(right_ttft)
                and left_ttft > 0 and right_ttft > 0
                and left_case["response"]["cached_tokens"] == 0
                and right_case["response"]["cached_tokens"] == 0):
            return {"status": "invalid", "reason": "timing sample invalid"}
        gain = left_ttft / right_ttft - 1
        gains.append(gain)
        rows.append({
            "prompt_tokens": left_case["target_prompt_tokens"],
            "repetition": left_case["repetition"],
            "m1_109_ttft_s": left_ttft,
            "m1_162_ttft_s": right_ttft,
            "gain": gain,
        })
    return {
        "status": "pass" if sum(gains) / len(gains) > 0 else "fail",
        "samples": rows,
        "paired_mean_gain": sum(gains) / len(gains),
        "sample_count": len(gains),
        "diagnostic_only": True,
    }


def compare(fused_off: Any, m1_109: Any, m1_162: Any,
            reused_aa: Any) -> dict[str, Any]:
    arms = {"fused_off": fused_off, "m1_109": m1_109, "m1_162": m1_162}
    reasons = []
    for label, value in arms.items():
        reasons.extend(arm_reasons(value, label))
    if not reasons:
        reasons.extend(cross_arm_reasons(arms))
    if (not isinstance(reused_aa, dict)
            or reused_aa.get("top1_flip_count") != 0
            or reused_aa.get("shared_token_abs_logprob_delta_p99_nats") != 0
            or reused_aa.get("position_sampling_one_sided_95_upper_nats") != 0):
        reasons.append("M1-179 A/A envelope is not exact zero")
    if reasons:
        return {"schema": SCHEMA, "version": VERSION, "status": "invalid",
                "classification": "invalid_evidence", "reasons": reasons,
                "promotion_authorized": False}

    capability = {
        "fused_off_vs_m1_109": capability_pair(
            fused_off, m1_109, "fused_off_vs_m1_109"),
        "m1_109_vs_m1_162": capability_pair(
            m1_109, m1_162, "m1_109_vs_m1_162"),
        "fused_off_vs_m1_162": capability_pair(
            fused_off, m1_162, "fused_off_vs_m1_162"),
    }
    candidate_complete = (m1_162["capability"].get("complete") is True)
    distributions = {}
    if candidate_complete and all(
            isinstance(value.get("teacher_forced"), dict)
            for value in arms.values()):
        try:
            distributions = {
                "fused_off_vs_m1_109": _distribution_pair(
                    fused_off, m1_109, "fused_off_vs_m1_109", reused_aa),
                "m1_109_vs_m1_162": _distribution_pair(
                    m1_109, m1_162, "m1_109_vs_m1_162", reused_aa),
                "fused_off_vs_m1_162": _distribution_pair(
                    fused_off, m1_162, "fused_off_vs_m1_162", reused_aa),
            }
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            return {"schema": SCHEMA, "version": VERSION,
                    "status": "invalid", "classification": "invalid_evidence",
                    "reasons": [str(exc)], "promotion_authorized": False}

    incremental = capability["m1_109_vs_m1_162"]
    critical_fail = incremental["baseline_only"] > 0
    status = "fail" if critical_fail else "inconclusive"
    classification = (
        "m1_162_capability_regression" if critical_fail else
        "development_capability_screen_passed_but_strata_underpowered")
    return {
        "schema": SCHEMA, "version": VERSION,
        "status": status, "classification": classification,
        "source_revision": fused_off["source_revision"],
        "runtime_identity": fused_off["runtime_identity"],
        "instance": fused_off["instance"],
        "model_path": fused_off["model_path"],
        "capability": capability,
        "distribution": distributions,
        "reused_m1_179_aa": reused_aa,
        "incremental_performance": _performance(m1_109, m1_162),
        "decisions": {
            "m1_109": "retained_development_candidate_pending_full_gates",
            "m1_162": ("stop_original_fp16_qk" if critical_fail else
                       "retained_blocked_pending_reviewer"),
            "capability_promotion_pass": False,
            "formal_evaluation_authorized": False,
        },
        "privacy": {
            "contains_prompts": False, "contains_model_outputs": False,
            "contains_token_ids_or_keys": False, "contains_images": False,
            "contains_tool_arguments": False, "contains_credentials": False,
        },
        "promotion_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fused_off", type=Path)
    parser.add_argument("m1_109", type=Path)
    parser.add_argument("m1_162", type=Path)
    parser.add_argument("--m1-179-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    arms = [json.loads(path.read_text(encoding="utf-8"))
            for path in (args.fused_off, args.m1_109, args.m1_162)]
    historical = json.loads(args.m1_179_summary.read_text(encoding="utf-8"))
    result = compare(*arms, historical["aa_distribution"])
    args.out.write_text(json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    return {"inconclusive": 3, "fail": 1, "invalid": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
