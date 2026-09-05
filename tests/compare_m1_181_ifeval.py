#!/usr/bin/env python3
"""Qualify M1-181 IFEval-64 and correctly bound fused-off distribution A/A."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import compare_m1_179_teacher_forced as distribution
import paired_noninferiority as paired


SCHEMA = "bi100-m1-181-adjudication-v1"
ARMS = {"fused_off": "fused_off", "m1_109": "m1_109_fp32_qk",
        "fused_off_b": "fused_off"}
TARGETS = (4096, 16384, 32768, 65536)


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def arm_reasons(value: Any, label: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{label}: arm is not an object"]
    reasons = []
    if (value.get("schema") != "bi100-m1-181-arm-observation-v1"
            or value.get("version") != 1 or value.get("arm") != label
            or value.get("algorithm_variant") != ARMS[label]
            or not all(isinstance(value.get(name), str) and value[name]
                       for name in ("source_revision", "runtime_identity",
                                    "instance", "model_path", "workload_id"))
            or not _finite(value)):
        reasons.append(f"{label}: identity or finite contract differs")
        return reasons
    population = value.get("request_population") or {}
    if (population.get("attempted") != population.get("completed")
            or population.get("failed") != 0
            or not isinstance(population.get("completed"), int)):
        reasons.append(f"{label}: request population is incomplete")
    ifeval = value.get("ifeval")
    if label == "fused_off_b":
        if ifeval is not None:
            reasons.append("fused_off_b: IFEval must not be repeated")
    elif not isinstance(ifeval, dict):
        reasons.append(f"{label}: IFEval evidence is missing")
    else:
        cases = ifeval.get("cases")
        expected = 16 if ifeval.get("stopped_after_smoke") else 64
        if (ifeval.get("selected") != 64
                or ifeval.get("completed") != expected
                or ifeval.get("complete") is not (expected == 64)
                or not isinstance(cases, list) or len(cases) != expected
                or len({case.get("key") for case in cases}) != expected):
            reasons.append(f"{label}: IFEval population differs")
        for case in cases or []:
            if (case.get("http_status") != 200
                    or case.get("finish_reason") not in ("stop", "length")
                    or case.get("all_values_finite") is not True
                    or not isinstance(case.get("strict"), list)
                    or not isinstance(case.get("loose"), list)
                    or not case.get("instruction_id_list")
                    or len(case["strict"]) != len(case["instruction_id_list"])
                    or len(case["loose"]) != len(case["instruction_id_list"])
                    or any(type(item) is not bool
                           for item in case["strict"] + case["loose"])):
                reasons.append(f"{label}: invalid IFEval case")
                break
    teacher = value.get("teacher_forced")
    stopped = isinstance(ifeval, dict) and ifeval.get("stopped_after_smoke")
    if stopped:
        if teacher is not None:
            reasons.append(f"{label}: teacher-forced ran after smoke stop")
    elif not isinstance(teacher, dict):
        reasons.append(f"{label}: teacher-forced evidence is missing")
    else:
        cases = teacher.get("cases")
        if (not isinstance(cases, list) or len(cases) != 4
                or tuple(sorted(case.get("prompt_tokens") for case in cases))
                != TARGETS
                or any(case.get("cached_tokens") != 0
                       or len(case.get("positions") or []) != 64
                       for case in cases)):
            reasons.append(f"{label}: teacher-forced population differs")
    return reasons


def cross_arm_reasons(arms: dict[str, dict[str, Any]]) -> list[str]:
    reasons = []
    reference = arms["fused_off"]
    for label, value in arms.items():
        if label == "fused_off":
            continue
        for name in ("source_revision", "runtime_identity", "instance",
                     "model_path", "workload_id"):
            if value.get(name) != reference.get(name):
                reasons.append(f"{label}: cross-arm {name} differs")
    left = reference.get("ifeval", {}).get("cases", [])
    right = arms["m1_109"].get("ifeval", {}).get("cases", [])
    if ({case.get("key") for case in left}
            != {case.get("key") for case in right}
            and not arms["m1_109"].get("ifeval", {}).get(
                "stopped_after_smoke")):
        reasons.append("IFEval paired key population differs")
    return reasons


def _outcomes(value: dict[str, Any], metric: str) -> dict[int, bool]:
    return {case["key"]: all(case[metric])
            for case in value["ifeval"]["cases"]}


def _paired_metric(baseline: list[bool], candidate: list[bool],
                   seed: int) -> dict[str, Any]:
    result = paired.paired_noninferiority(
        baseline, candidate, margin=0.05, confidence=0.95,
        bootstrap_samples=20000, seed=seed)
    lower = result["statistics"]["one_sided_lower_bound"]
    if len(baseline) >= 59:
        result["qualified"] = lower > -0.05
        result["status"] = "pass" if result["qualified"] else "fail"
        result["reasons"] = ([] if result["qualified"] else
                             ["paired lower bound does not exceed -5pp"])
    return result


def ifeval_pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    if right["ifeval"]["stopped_after_smoke"]:
        return {
            "status": "fail", "classification": "smoke_regression_stop",
            "sample_count": 16,
            "smoke_baseline_only": right["ifeval"]["smoke_baseline_only"],
        }
    metrics = {}
    for index, metric in enumerate(("strict", "loose")):
        lhs, rhs = _outcomes(left, metric), _outcomes(right, metric)
        keys = sorted(lhs)
        if keys != sorted(rhs):
            raise ValueError("IFEval metric keys differ")
        metrics[metric] = _paired_metric(
            [lhs[key] for key in keys], [rhs[key] for key in keys],
            20260905 + index)
    statuses = {value["status"] for value in metrics.values()}
    status = "fail" if "fail" in statuses else (
        "inconclusive" if "inconclusive" in statuses else "pass")
    return {
        "status": status,
        "classification": ("ifeval_64_development_noninferiority_pass"
                           if status == "pass" else
                           "ifeval_64_underpowered" if status == "inconclusive"
                           else "ifeval_64_regression"),
        "sample_count": 64, "metrics": metrics,
        "production_two_point_noninferiority_claimed": False,
    }


def _calibrated_distribution(left: dict[str, Any], right: dict[str, Any],
                             aa: dict[str, Any] | None) -> dict[str, Any]:
    result = distribution._pair(left["teacher_forced"], right["teacher_forced"])
    if aa is None:
        result.update({"status": "inconclusive", "calibrated": False,
                       "classification": "uncalibrated_distribution_diagnostic",
                       "thresholds": None})
        return result
    high = max(0.1, 4 * aa["shared_token_abs_logprob_delta_p99_nats"])
    nll = max(0.01, 2 * max(
        0.0, aa["position_sampling_one_sided_95_upper_nats"]))
    high_count = sum(item["control_margin_nats"] > high
                     for item in result["flip_details"])
    drift = (high_count > 0
             or result["position_sampling_one_sided_95_upper_nats"] > nll
             or any(item["mean_nll_difference_nats"] > nll
                    for item in result["nll_by_length"]))
    result.update({
        "status": "inconclusive" if drift else "pass",
        "calibrated": True, "aa_control_variant": "fused_off",
        "classification": ("distribution_drift_requires_adjudication"
                           if drift else "within_fused_off_aa_envelope"),
        "high_margin_flip_count": high_count,
        "thresholds": {"high_margin_nats": high,
                       "nll_regression_nats": nll},
    })
    return result


def compare(fused_off: Any, m1_109: Any,
            fused_off_b: Any | None = None) -> dict[str, Any]:
    arms = {"fused_off": fused_off, "m1_109": m1_109}
    if fused_off_b is not None:
        arms["fused_off_b"] = fused_off_b
    reasons = []
    for label, value in arms.items():
        reasons.extend(arm_reasons(value, label))
    if not reasons:
        reasons.extend(cross_arm_reasons(arms))
    if reasons:
        return {"schema": SCHEMA, "version": 1, "status": "invalid",
                "classification": "invalid_evidence", "reasons": reasons}
    try:
        capability = ifeval_pair(fused_off, m1_109)
        if capability["classification"] == "smoke_regression_stop":
            aa = None
            candidate_distribution = {
                "status": "not_run", "classification": "capability_smoke_stop"}
        else:
            aa = (distribution._pair(fused_off["teacher_forced"],
                                     fused_off_b["teacher_forced"])
                  if fused_off_b is not None else None)
            candidate_distribution = _calibrated_distribution(
                fused_off, m1_109, aa)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return {"schema": SCHEMA, "version": 1, "status": "invalid",
                "classification": "invalid_evidence", "reasons": [str(exc)]}
    status = "fail" if capability["status"] == "fail" else "inconclusive"
    return {
        "schema": SCHEMA, "version": 1, "status": status,
        "classification": ("m1_109_ifeval_regression"
                           if status == "fail" else
                           "m1_109_retained_pending_distribution_adjudication"),
        "source_revision": fused_off["source_revision"],
        "runtime_identity": fused_off["runtime_identity"],
        "instance": fused_off["instance"], "model_path": fused_off["model_path"],
        "ifeval_statistical_capability": capability,
        "fused_off_aa_distribution": aa,
        "fused_off_vs_m1_109_distribution": candidate_distribution,
        "promotion_authorized": False,
        "privacy": {"contains_prompts": False,
                    "contains_model_outputs": False,
                    "contains_token_ids_or_identity_key": False,
                    "contains_credentials": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fused_off", type=Path)
    parser.add_argument("m1_109", type=Path)
    parser.add_argument("--fused-off-b", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    values = [json.loads(args.fused_off.read_text()),
              json.loads(args.m1_109.read_text())]
    control_b = (json.loads(args.fused_off_b.read_text())
                 if args.fused_off_b else None)
    result = compare(*values, control_b)
    args.out.write_text(json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return {"inconclusive": 3, "fail": 1, "invalid": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
