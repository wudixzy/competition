#!/usr/bin/env python3
"""Compare private teacher-forced top-k observations without retaining IDs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


OBSERVATION_SCHEMA = "bi100-teacher-forced-topk-observation-v1"
REPORT_SCHEMA = "bi100-teacher-forced-topk-comparison-v1"
VERSION = 1
Json = dict[str, Any]


def exit_code(status: str) -> int:
    return {"pass": 0, "fail": 1, "invalid": 2}[status]


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


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty metric")
    ordered = sorted(values)
    rank = math.ceil(percentile * len(ordered)) - 1
    return ordered[max(0, min(rank, len(ordered) - 1))]


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_arm(value: Any, mode: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{mode}: report root must be an object"]
    reasons = []
    expected = {
        "schema",
        "version",
        "mode",
        "source_revision",
        "runtime_identity",
        "instance",
        "model_path",
        "gpu_count",
        "tensor_parallel_size",
        "max_model_len",
        "top_k",
        "optimization",
        "cases",
        "privacy",
    }
    if set(value) != expected:
        reasons.append(f"{mode}: report fields are invalid")
        return reasons
    if (
        value.get("schema") != OBSERVATION_SCHEMA
        or value.get("version") != VERSION
        or value.get("mode") != mode
        or value.get("gpu_count") != 4
        or value.get("tensor_parallel_size") != 4
        or value.get("max_model_len") != 262144
        or not isinstance(value.get("top_k"), int)
        or value["top_k"] < 2
    ):
        reasons.append(f"{mode}: report identity is invalid")
    privacy = value.get("privacy")
    if (
        not isinstance(privacy, dict)
        or privacy.get("contains_private_hmac_token_keys") is not True
        or privacy.get("must_remain_outside_repository") is not True
        or any(privacy.get(name) is not False for name in (
            "contains_raw_prompts",
            "contains_raw_model_outputs",
            "contains_raw_token_ids",
            "contains_credentials",
        ))
    ):
        reasons.append(f"{mode}: privacy contract is invalid")
    cases = value.get("cases")
    if not isinstance(cases, list) or not cases:
        reasons.append(f"{mode}: cases are missing")
        return reasons
    seen_ids = set()
    for case in cases:
        if (
            not isinstance(case, dict)
            or set(case) != {"id", "prompt_tokens", "positions"}
            or not isinstance(case.get("id"), str)
            or not case["id"]
            or case["id"] in seen_ids
            or not isinstance(case.get("prompt_tokens"), int)
            or isinstance(case["prompt_tokens"], bool)
            or case["prompt_tokens"] <= 0
            or not isinstance(case.get("positions"), list)
            or not case["positions"]
        ):
            reasons.append(f"{mode}: case structure is invalid")
            continue
        seen_ids.add(case["id"])
        seen_positions = set()
        for position in case["positions"]:
            if (
                not isinstance(position, dict)
                or set(position)
                != {"position", "actual_token_key", "top_logprobs"}
                or not isinstance(position.get("position"), int)
                or isinstance(position["position"], bool)
                or position["position"] < 0
                or position["position"] in seen_positions
                or not _is_digest(position.get("actual_token_key"))
                or not isinstance(position.get("top_logprobs"), list)
                or len(position["top_logprobs"]) < 2
                or len(position["top_logprobs"]) > value["top_k"] + 1
            ):
                reasons.append(f"{mode}: position structure is invalid")
                continue
            seen_positions.add(position["position"])
            keys = []
            previous = math.inf
            for item in position["top_logprobs"]:
                if (
                    not isinstance(item, dict)
                    or set(item) != {"token_key", "logprob"}
                    or not _is_digest(item.get("token_key"))
                    or not isinstance(item.get("logprob"), (int, float))
                    or isinstance(item["logprob"], bool)
                    or not math.isfinite(item["logprob"])
                    or item["logprob"] > previous
                ):
                    reasons.append(
                        f"{mode}: top-logprob structure is invalid")
                    break
                keys.append(item["token_key"])
                previous = item["logprob"]
            if len(keys) != len(set(keys)):
                reasons.append(f"{mode}: top-logprob token keys repeat")
            if position["actual_token_key"] not in keys:
                reasons.append(
                    f"{mode}: actual token is absent from top-logprobs")
    return reasons


def _case_map(value: Json) -> dict[str, Json]:
    return {case["id"]: case for case in value["cases"]}


def _position_map(case: Json) -> dict[int, Json]:
    return {position["position"]: position for position in case["positions"]}


def _logprob_map(position: Json) -> dict[str, float]:
    return {
        item["token_key"]: float(item["logprob"])
        for item in position["top_logprobs"]
    }


def _identity_reasons(control: Json, candidate: Json) -> list[str]:
    reasons = []
    for field in (
        "source_revision",
        "runtime_identity",
        "instance",
        "model_path",
        "gpu_count",
        "tensor_parallel_size",
        "max_model_len",
        "top_k",
    ):
        if control.get(field) != candidate.get(field):
            reasons.append(f"A/B identity differs in {field}")
    control_optimization = control.get("optimization")
    candidate_optimization = candidate.get("optimization")
    if (
        not isinstance(control_optimization, dict)
        or not isinstance(candidate_optimization, dict)
        or control_optimization.get("fused_prefill") != "0"
        or candidate_optimization.get("fused_prefill") != "1"
    ):
        reasons.append("A/B fused-prefill selectors are invalid")
    elif {
        key: value
        for key, value in control_optimization.items()
        if key != "fused_prefill"
    } != {
        key: value
        for key, value in candidate_optimization.items()
        if key != "fused_prefill"
    }:
        reasons.append("A/B optimization identity differs beyond selector")
    return reasons


def _invalid_result(reasons: list[str]) -> Json:
    return {
        "schema": REPORT_SCHEMA,
        "version": VERSION,
        "status": "invalid",
        "qualified": False,
        "validation_reasons": reasons,
        "reasons": [],
        "authorization": {
            "teacher_forced_numerical_screen_authorized": False,
            "overall_promotion_authorized": False,
        },
        "privacy": {
            "contains_private_token_identity": False,
            "contains_token_ids": False,
            "contains_prompts": False,
            "contains_model_outputs": False,
            "contains_credentials": False,
        },
    }


def compare(control: Any, candidate: Any, contract: Any) -> Json:
    validation_reasons = _validate_arm(control, "control")
    validation_reasons.extend(_validate_arm(candidate, "candidate"))
    if not isinstance(contract, dict):
        validation_reasons.append("contract root must be an object")
    else:
        if (
            contract.get("schema")
            != "bi100-layered-quality-gate-contract-v1"
            or contract.get("version") != 1
            or not isinstance(contract.get("teacher_forced_topk"), dict)
        ):
            validation_reasons.append("contract identity is invalid")
    if not validation_reasons:
        validation_reasons.extend(_identity_reasons(control, candidate))
    if validation_reasons:
        return _invalid_result(validation_reasons)

    thresholds = contract["teacher_forced_topk"]
    required_lengths = thresholds["required_prompt_tokens"]
    control_cases = _case_map(control)
    candidate_cases = _case_map(candidate)
    invalidation_reasons = []
    reasons = []
    if set(control_cases) != set(candidate_cases):
        invalidation_reasons.append("A/B case identities differ")
    observed_lengths = sorted(
        case["prompt_tokens"] for case in control["cases"])
    if observed_lengths != sorted(required_lengths):
        invalidation_reasons.append(
            "required prompt-token matrix is incomplete")

    actual_deltas = []
    shared_deltas = []
    nll_deltas = []
    top1_matches = 0
    top1_mismatches = []
    mutually_uncovered = 0
    case_summaries = []
    total_positions = 0

    if not invalidation_reasons:
        for case_id in sorted(control_cases):
            left_case = control_cases[case_id]
            right_case = candidate_cases[case_id]
            if left_case["prompt_tokens"] != right_case["prompt_tokens"]:
                invalidation_reasons.append(
                    f"{case_id}: prompt token count differs")
                continue
            left_positions = _position_map(left_case)
            right_positions = _position_map(right_case)
            if set(left_positions) != set(right_positions):
                invalidation_reasons.append(
                    f"{case_id}: sampled positions differ")
                continue
            if (
                len(left_positions)
                < thresholds["minimum_positions_per_case"]
            ):
                invalidation_reasons.append(
                    f"{case_id}: too few sampled positions")
            case_top1_matches = 0
            case_uncovered = 0
            for position_id in sorted(left_positions):
                left = left_positions[position_id]
                right = right_positions[position_id]
                if left["actual_token_key"] != right["actual_token_key"]:
                    invalidation_reasons.append(
                        f"{case_id}: teacher token differs at sampled position")
                    continue
                left_values = _logprob_map(left)
                right_values = _logprob_map(right)
                actual = left["actual_token_key"]
                if actual not in right_values:
                    invalidation_reasons.append(
                        f"{case_id}: candidate omitted teacher token")
                    continue
                actual_delta = abs(
                    right_values[actual] - left_values[actual])
                actual_deltas.append(actual_delta)
                nll_deltas.append(
                    left_values[actual] - right_values[actual])
                for token_key in left_values.keys() & right_values.keys():
                    shared_deltas.append(abs(
                        right_values[token_key] - left_values[token_key]))
                left_top = left["top_logprobs"][0]["token_key"]
                right_top = right["top_logprobs"][0]["token_key"]
                baseline_margin = (
                    left["top_logprobs"][0]["logprob"]
                    - left["top_logprobs"][1]["logprob"]
                )
                if left_top == right_top:
                    top1_matches += 1
                    case_top1_matches += 1
                else:
                    covered = (
                        left_top in right_values
                        and right_top in left_values
                    )
                    if not covered:
                        mutually_uncovered += 1
                        case_uncovered += 1
                    top1_mismatches.append({
                        "baseline_margin": baseline_margin,
                        "mutually_covered": covered,
                    })
                total_positions += 1
            case_summaries.append({
                "id": case_id,
                "prompt_tokens": left_case["prompt_tokens"],
                "sampled_positions": len(left_positions),
                "top1_agreement": (
                    case_top1_matches / len(left_positions)
                    if left_positions else 0.0
                ),
                "mutually_uncovered_top1_mismatches": case_uncovered,
            })

    if not actual_deltas or not shared_deltas or total_positions == 0:
        invalidation_reasons.append(
            "no comparable teacher-forced positions")
    if invalidation_reasons:
        return _invalid_result(invalidation_reasons)

    actual_max = max(actual_deltas)
    actual_p99 = _percentile(actual_deltas, 0.99)
    shared_p99 = _percentile(shared_deltas, 0.99)
    mean_nll_delta = sum(nll_deltas) / len(nll_deltas)
    top1_agreement = top1_matches / total_positions
    margin_guard = max(
        thresholds["high_margin_floor_nats"],
        thresholds["high_margin_error_multiplier"] * shared_p99,
    )
    high_margin_mismatches = sum(
        row["baseline_margin"] > margin_guard
        for row in top1_mismatches
    )
    if top1_agreement < thresholds["minimum_top1_agreement"]:
        reasons.append("top-1 agreement is below the fixed floor")
    if actual_max > thresholds["maximum_actual_logprob_absolute_delta"]:
        reasons.append("maximum teacher-token logprob delta is too large")
    if (
        actual_p99
        > thresholds["maximum_actual_logprob_absolute_delta_p99"]
    ):
        reasons.append("p99 teacher-token logprob delta is too large")
    if (
        shared_p99
        > thresholds["maximum_shared_logprob_absolute_delta_p99"]
    ):
        reasons.append("p99 shared top-k logprob delta is too large")
    if (
        mean_nll_delta
        > thresholds["maximum_mean_actual_nll_regression"]
    ):
        reasons.append("mean teacher-token NLL regressed")
    if (
        high_margin_mismatches
        > thresholds["maximum_high_margin_top1_mismatches"]
    ):
        reasons.append("one or more high-margin top-1 choices changed")
    if (
        mutually_uncovered
        > thresholds["maximum_mutually_uncovered_top1_mismatches"]
    ):
        reasons.append(
            "one or more top-1 mismatches lack mutual top-k support")

    qualified = not reasons
    return {
        "schema": REPORT_SCHEMA,
        "version": VERSION,
        "status": "pass" if qualified else "fail",
        "qualified": qualified,
        "validation_reasons": [],
        "reasons": reasons,
        "source_revision": control["source_revision"],
        "instance": control["instance"],
        "gpu_count": control["gpu_count"],
        "tensor_parallel_size": control["tensor_parallel_size"],
        "max_model_len": control["max_model_len"],
        "top_k": control["top_k"],
        "case_count": len(control_cases),
        "sampled_positions": total_positions,
        "metrics": {
            "top1_agreement": top1_agreement,
            "top1_mismatch_count": len(top1_mismatches),
            "mutually_uncovered_top1_mismatches": mutually_uncovered,
            "high_margin_guard_nats": margin_guard,
            "high_margin_top1_mismatches": high_margin_mismatches,
            "teacher_token_logprob_absolute_delta_max": actual_max,
            "teacher_token_logprob_absolute_delta_p99": actual_p99,
            "shared_topk_logprob_absolute_delta_p99": shared_p99,
            "mean_teacher_token_nll_regression": mean_nll_delta,
        },
        "cases": case_summaries,
        "thresholds": thresholds,
        "authorization": {
            "teacher_forced_numerical_screen_authorized": qualified,
            "overall_promotion_authorized": False,
        },
        "privacy": {
            "contains_private_token_identity": False,
            "contains_token_ids": False,
            "contains_prompts": False,
            "contains_model_outputs": False,
            "contains_credentials": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("quality/layered_quality_gate.v1.json"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    control = json.loads(args.control.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    report = compare(control, candidate, contract)
    _atomic_write(args.out, report)
    return exit_code(report["status"])


if __name__ == "__main__":
    raise SystemExit(main())
