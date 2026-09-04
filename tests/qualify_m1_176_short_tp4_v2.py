#!/usr/bin/env python3
"""Qualify the fixed control/control/candidate M1-176 short-TP4 screen."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import statistics
import tempfile
from typing import Any

import short_tp4_v2_service as service
import validate_bi100_metrics_contract as metrics_contract


SCHEMA = "bi100-m1-176-short-tp4-qualification-v2"
BOOTSTRAP_SAMPLES = 20000
BOOTSTRAP_SEED = 20260904
SELECTORS = ("control_a", "control_b", "candidate")
STATES = ("cold", "partial-prefix", "full-warm")


def _load(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if not path.is_relative_to(Path("/tmp")) or path.stat().st_mode & 0o077:
        raise ValueError("private L3 input must be mode 0600 under /tmp")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("L3 input is not an object")
    return value


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("metric sample is empty")
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def _bootstrap_bound(values: list[float], fraction: float, seed: int) -> float:
    if not values:
        raise ValueError("paired sample is empty")
    generator = random.Random(seed)
    count = len(values)
    means = sorted(sum(values[generator.randrange(count)]
                       for _ in range(count)) / count
                   for _ in range(BOOTSTRAP_SAMPLES))
    return _percentile(means, fraction)


def _primary_map(report: dict[str, Any]) -> dict[tuple[str, int, int], dict]:
    result = {}
    for case in report["cold_cases"]:
        identity = (case["target_prompt_tokens"], case["repetition"])
        result[("cold", *identity)] = case["cold"]
        result[("full-warm", *identity)] = case["warm"]
    for case in report["partial_cases"]:
        identity = (case["target_prompt_tokens"], case["repetition"])
        result[("partial-prefix", *identity)] = case["partial"]
    return result


def _output_identity(response: dict[str, Any]) -> tuple[Any, ...]:
    return (
        response["first_output_identity"], response["output_identity"],
        response["completion_tokens"], response["finish_reason"],
    )


def _arm_summary(report: dict[str, Any]) -> dict[str, Any]:
    primary = _primary_map(report)
    values = list(primary.values())
    def distribution(name: str) -> dict[str, float]:
        samples = [float(value[name]) for value in values]
        return {
            "p50": _percentile(samples, 0.50),
            "p90": _percentile(samples, 0.90),
            "p99": _percentile(samples, 0.99),
        }
    return {
        "request_population": {
            "expected": report["expected_requests"],
            "completed": report["completed_requests"],
            "success_rate": report["metrics"]["success_rate"],
            "error_rate": report["metrics"]["error_rate"],
        },
        "ttft_s": distribution("ttft_s"),
        "tpot_s": distribution("tpot_s"),
        "itl_s": distribution("itl_s"),
        "e2e_latency_s": distribution("elapsed_s"),
        "input_tps": distribution("input_tps"),
        "output_tps": distribution("output_tps"),
        "cache_tps": distribution("cache_tps"),
        "request_throughput_rps": distribution("request_throughput_rps"),
        "slo_goodput": {
            "requests": report["metrics"]["slo_goodput_requests"],
            "total": report["metrics"]["slo_total_requests"],
            "rate": (report["metrics"]["slo_goodput_requests"]
                     / report["metrics"]["slo_total_requests"]),
        },
        "service_population_elapsed_s": report["elapsed_s"],
        "service_population_throughput_rps": (
            report["completed_requests"] / report["elapsed_s"]),
    }


def _status_reasons(statuses: dict[str, dict[str, Any]]) -> list[str]:
    reasons = []
    base = statuses["control_a"]
    for selector, value in statuses.items():
        if (value.get("schema") != "bi100-m1-176-short-tp4-arm-runner-v2"
                or value.get("version") != 2
                or value.get("selector") != selector
                or value.get("source_revision") != base.get("source_revision")
                or value.get("runtime_identity") != base.get("runtime_identity")
                or value.get("instance") != base.get("instance")
                or value.get("model_path") != base.get("model_path")
                or value.get("pair_id") != base.get("pair_id")
                or value.get("targets") != list(service.TARGETS)
                or value.get("cache_states") != list(STATES)
                or value.get("repetitions") != service.REPETITIONS
                or value.get("service_startups") != 1
                or value.get("gpu_count") != 4
                or value.get("tensor_parallel_size") != 4
                or value.get("request_population") != {
                    "service_expected": 72,
                    "teacher_forced_expected": 4,
                    "total_expected": 76,
                    "total_completed": 76,
                }):
            reasons.append(f"{selector}: runner identity/population differs")
        artifact = value.get("candidate_artifact") or {}
        if (artifact.get("sha256")
                != (base.get("candidate_artifact") or {}).get("sha256")
                or artifact.get("active") is not (selector == "candidate")):
            reasons.append(f"{selector}: candidate artifact selector differs")
        if value.get("qualified") is True:
            if (value.get("returncode") != 0
                    or value.get("result_status") != "pass"
                    or value.get("terminal_stage") != "complete"
                    or any(gate != 0 for gate in (value.get("gates") or {}).values())
                    or not all((value.get("artifacts_present") or {}).values())):
                reasons.append(f"{selector}: successful lifecycle is malformed")
        dispatch = value.get("dispatch_count")
        if ((selector == "candidate" and not isinstance(dispatch, int))
                or (selector == "candidate" and dispatch < 2)
                or (selector != "candidate" and dispatch != 0)):
            reasons.append(f"{selector}: dispatch marker differs")
    return reasons


def _distribution_reasons(
    distribution: Any,
    base_status: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[list[str], str]:
    if not isinstance(distribution, dict):
        return ["distribution evidence is not an object"], "invalid"
    status = distribution.get("status")
    allowed = {"pass", "inconclusive", "invalid"}
    reasons = []
    if status not in allowed:
        reasons.append("distribution status is unsupported")
        status = "invalid"
    if (distribution.get("schema")
            != "bi100-teacher-forced-distribution-v2"
            or distribution.get("version") != 2):
        reasons.append("distribution schema/version differs")
    for field in ("source_revision", "runtime_identity", "instance",
                  "model_path"):
        if (not isinstance(distribution.get(field), str)
                or not distribution[field]
                or distribution[field] != base_status.get(field)):
            reasons.append(f"distribution {field} differs")
    if distribution.get("targets") != list(service.TARGETS):
        reasons.append("distribution targets differ")
    expected_workload = {
        "case_ids": [f"length_{target}" for target in service.TARGETS],
        "prompt_tokens": list(service.TARGETS),
        "sampled_positions_per_case": [64] * len(service.TARGETS),
    }
    if distribution.get("workload_identity") != expected_workload:
        reasons.append("distribution workload identity differs")
    if distribution.get("arm_binding") != {
            "control_a": "control", "control_b": "control",
            "candidate": "candidate"}:
        reasons.append("distribution arms are not bound to candidate")
    sampled = distribution.get("sampled_positions")
    aa = distribution.get("aa")
    candidate = distribution.get("candidate")
    decision = distribution.get("decision")
    if (not isinstance(sampled, int) or isinstance(sampled, bool)
            or sampled <= 0 or not isinstance(aa, dict) or not aa
            or not isinstance(candidate, dict) or not candidate
            or not isinstance(decision, dict) or not decision
            or aa.get("sampled_positions") != sampled
            or candidate.get("sampled_positions") != sampled):
        reasons.append("distribution sample population is incomplete")
    else:
        try:
            expected_decision = metrics_contract.classify_distribution(
                candidate, aa, contract)
        except (metrics_contract.ContractError, KeyError, TypeError) as exc:
            reasons.append(f"distribution decision is invalid: {exc}")
        else:
            if decision != expected_decision:
                reasons.append("distribution decision does not match evidence")
            if (distribution.get("classification")
                    != expected_decision.get("classification")
                    or distribution.get("status")
                    != expected_decision.get("status")):
                reasons.append("distribution status/classification differs")
    return reasons, "invalid" if reasons else status


def qualify(
    statuses: dict[str, dict[str, Any]],
    measurements: dict[str, dict[str, Any]],
    manifests: dict[str, dict[str, Any]],
    distribution: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    reasons = _status_reasons(statuses)
    try:
        metrics_contract.validate_contract(contract, "layered")
    except metrics_contract.ContractError as exc:
        reasons.append(str(exc))
    distribution_reasons, distribution_status = _distribution_reasons(
        distribution, statuses["control_a"], contract)
    reasons.extend(distribution_reasons)
    base_manifest = manifests["control_a"]
    for selector, manifest in manifests.items():
        if (manifest.get("schema") != "bi100-quality-runtime-manifest-v2"
                or manifest.get("version") != 2):
            reasons.append(f"{selector}: runtime manifest is invalid")
            continue
        for field in (
            "source_revision", "runtime_identity", "instance", "model_path",
            "tokenizer_path", "gpu_count", "tensor_parallel_size",
            "max_model_len", "served_model_name", "command",
        ):
            if manifest.get(field) != base_manifest.get(field):
                reasons.append(f"{selector}: runtime {field} differs")
        environment = dict(manifest.get("environment") or {})
        expected_selector = "1" if selector == "candidate" else "0"
        if environment.pop("BI100_ATTN_COREX_FUSED_PREFILL", None) \
                != expected_selector:
            reasons.append(f"{selector}: runtime selector differs")
        base_environment = dict(base_manifest.get("environment") or {})
        base_environment.pop("BI100_ATTN_COREX_FUSED_PREFILL", None)
        if environment != base_environment:
            reasons.append(f"{selector}: non-candidate environment differs")
    for selector, report in measurements.items():
        evaluation = service.evaluate(report)
        if not evaluation["qualified"]:
            reasons.append(f"{selector}: " + "; ".join(evaluation["reasons"]))
        if (report.get("selector") != selector
                or report.get("prompt_set_id")
                != measurements["control_a"].get("prompt_set_id")
                or report.get("workload_order")
                != measurements["control_a"].get("workload_order")
                or report.get("expected_requests") != 72
                or report.get("completed_requests") != 72):
            reasons.append(f"{selector}: workload identity/population differs")
    if reasons:
        candidate_attributable = (
            statuses["control_a"].get("qualified") is True
            and statuses["control_b"].get("qualified") is True
            and statuses["candidate"].get("qualified") is False
            and (measurements.get("candidate") or {}).get("reasons")
        )
        return {
            "schema": SCHEMA, "version": 2,
            "status": "fail" if candidate_attributable else "invalid",
            "classification": ("candidate_hard_gate_failure"
                               if candidate_attributable else "invalid"),
            "reasons": reasons,
            "promotion_authorized": False,
        }

    primary = {name: _primary_map(value)
               for name, value in measurements.items()}
    identities = sorted(primary["control_a"])
    if any(sorted(primary[name]) != identities for name in SELECTORS):
        return {
            "schema": SCHEMA, "version": 2, "status": "invalid",
            "classification": "invalid",
            "reasons": ["paired timing identities differ"],
            "promotion_authorized": False,
        }
    candidate_gains = [
        primary["control_a"][identity]["ttft_s"]
        / primary["candidate"][identity]["ttft_s"] - 1.0
        for identity in identities
    ]
    aa_gains = [
        primary["control_a"][identity]["ttft_s"]
        / primary["control_b"][identity]["ttft_s"] - 1.0
        for identity in identities
    ]
    point = statistics.mean(candidate_gains)
    lower = _bootstrap_bound(candidate_gains, 0.05, BOOTSTRAP_SEED)
    buckets = []
    supported_regressions = []
    for state in STATES:
        for target in service.TARGETS:
            bucket_identities = [
                identity for identity in identities
                if identity[0] == state and identity[1] == target
            ]
            gains = [
                primary["control_a"][identity]["ttft_s"]
                / primary["candidate"][identity]["ttft_s"] - 1.0
                for identity in bucket_identities
            ]
            bucket_point = statistics.mean(gains)
            bucket_lower = _bootstrap_bound(
                gains, 0.05, BOOTSTRAP_SEED + target + len(state))
            bucket_upper = _bootstrap_bound(
                gains, 0.95, BOOTSTRAP_SEED + target + len(state))
            supported = bucket_upper < -0.05
            if supported:
                supported_regressions.append(-bucket_point)
            buckets.append({
                "state": state,
                "target_prompt_tokens": target,
                "paired_gains": gains,
                "control_a_ttft_s": [
                    primary["control_a"][identity]["ttft_s"]
                    for identity in bucket_identities
                ],
                "control_b_ttft_s": [
                    primary["control_b"][identity]["ttft_s"]
                    for identity in bucket_identities
                ],
                "candidate_ttft_s": [
                    primary["candidate"][identity]["ttft_s"]
                    for identity in bucket_identities
                ],
                "mean_gain": bucket_point,
                "one_sided_95_lower_ci": bucket_lower,
                "one_sided_95_upper_ci": bucket_upper,
                "supported_regression_over_5_percent": supported,
            })
    performance = metrics_contract.classify_performance(
        point, lower, contract,
        common_bucket_regression=(max(supported_regressions)
                                  if supported_regressions else 0.0),
        common_bucket_regression_supported=bool(supported_regressions),
    )
    output_matches = {
        name: sum(
            _output_identity(primary["control_a"][identity])
            == _output_identity(primary[name][identity])
            for identity in identities)
        for name in ("control_b", "candidate")
    }
    if distribution_status == "invalid":
        status = "invalid"
    elif performance["status"] == "fail":
        status = "fail"
    elif (performance["status"] == "inconclusive"
          or distribution_status == "inconclusive"):
        status = "inconclusive"
    else:
        status = "pass"
    return {
        "schema": SCHEMA,
        "version": 2,
        "status": status,
        "classification": "short_tp4_" + status,
        "source_revision": statuses["control_a"]["source_revision"],
        "instance": statuses["control_a"]["instance"],
        "pair_id": statuses["control_a"]["pair_id"],
        "targets": list(service.TARGETS),
        "cache_states": list(STATES),
        "request_population_per_arm": {
            "service": 72,
            "teacher_forced": 4,
            "total": 76,
        },
        "service_startups_per_arm": 1,
        "arm_summaries": {
            name: _arm_summary(measurements[name]) for name in SELECTORS
        },
        "performance": {
            "status": performance["status"],
            "decision": performance,
            "paired_gain_point_estimate": point,
            "paired_gain_one_sided_95_lower_ci": lower,
            "paired_gains": candidate_gains,
            "aa_paired_gain_mean": statistics.mean(aa_gains),
            "aa_paired_gain_p99_abs": _percentile(
                [abs(value) for value in aa_gains], 0.99),
            "buckets": buckets,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "protocol_and_cache": {
            "status": "pass",
            "all_http_sse_usage_finish_reason_valid": True,
            "all_cache_accounting_valid": True,
            "within_arm_deterministic_output_exact": True,
            "control_control_output_matches": output_matches["control_b"],
            "candidate_output_matches": output_matches["candidate"],
            "paired_output_count": len(identities),
        },
        "distribution": {
            "status": distribution_status,
            "classification": distribution.get("classification"),
            "aa": distribution.get("aa"),
            "candidate": distribution.get("candidate"),
            "decision": distribution.get("decision"),
        },
        "capability": {
            "status": "inconclusive",
            "classification": "short_functional_only_full_capability_not_run",
            "reason": "full capability suite is outside the M1-176 L3 scope",
        },
        "experiment_validity": {"status": "pass"},
        "privacy": {
            "contains_private_output_identities": False,
            "contains_teacher_token_keys": False,
            "contains_prompts": False,
            "contains_model_outputs": False,
            "contains_credentials": False,
        },
        "reasons": performance.get("reasons", []),
        "promotion_authorized": False,
    }


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    for selector in SELECTORS:
        parser.add_argument(f"--{selector.replace('_', '-')}-root",
                            type=Path, required=True)
    parser.add_argument("--distribution", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    roots = {
        "control_a": args.control_a_root,
        "control_b": args.control_b_root,
        "candidate": args.candidate_root,
    }
    try:
        statuses = {name: _load(root / "runner_status.json")
                    for name, root in roots.items()}
        measurements = {name: _load(root / "measurement_private.json")
                        for name, root in roots.items()}
        manifests = {name: _load(root / "runtime_manifest_v2.json")
                     for name, root in roots.items()}
        result = qualify(
            statuses, measurements, manifests, _load(args.distribution),
            json.loads(args.contract.read_text(encoding="utf-8")))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {
            "schema": SCHEMA, "version": 2, "status": "invalid",
            "classification": "invalid", "reasons": [str(exc)],
            "promotion_authorized": False,
        }
    _atomic_write(args.out, result)
    print(json.dumps({
        "status": result["status"],
        "classification": result["classification"],
        "reasons": result.get("reasons", []),
    }, sort_keys=True))
    return {"pass": 0, "fail": 1, "invalid": 2, "inconclusive": 3}[
        result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
