#!/usr/bin/env python3
"""Qualify one control/candidate focused attention-operator TP4 pair."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any

import attention_operator_tp4_service as service


SCHEMA = "bi100-attention-operator-tp4-pair-v1"
BOOTSTRAP_SAMPLES = 20000
BOOTSTRAP_SEED = 20260904


def _finite(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("sample is empty")
    ordered = sorted(values)
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def _bootstrap_lower(values: list[float]) -> float:
    if not values:
        raise ValueError("paired sample is empty")
    generator = random.Random(BOOTSTRAP_SEED)
    count = len(values)
    means = sorted(sum(values[generator.randrange(count)]
                       for _ in range(count)) / count
                   for _ in range(BOOTSTRAP_SAMPLES))
    return _percentile(means, 0.05)


def _case_map(report: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    return {(case["target_prompt_tokens"], case["repetition"]):
            case["response"] for case in report["cases"]}


def _arm_reasons(
    selector: str,
    status: Any,
    manifest: Any,
    measurement: Any,
    targets: tuple[int, ...],
    repetitions: int,
) -> list[str]:
    reasons = []
    if not isinstance(status, dict) or not isinstance(manifest, dict):
        return [f"{selector}: runner evidence is malformed"]
    gates = status.get("gates")
    if (status.get("schema") != "bi100-attention-operator-tp4-arm-v1"
            or status.get("version") != 1
            or status.get("change_scope") != "attention_operator"
            or status.get("selector") != selector
            or status.get("qualified") is not True
            or status.get("result_status") != "pass"
            or status.get("returncode") != 0
            or status.get("terminal_stage") != "complete"
            or status.get("targets") != list(targets)
            or status.get("repetitions") != repetitions
            or status.get("request_population") != {
                "expected": len(targets) * repetitions,
                "attempted": len(targets) * repetitions,
                "completed": len(targets) * repetitions,
                "failed": 0}
            or status.get("service_startups") != 1
            or status.get("gpu_count") != 4
            or status.get("tensor_parallel_size") != 4
            or not isinstance(gates, dict)
            or any(value != 0 for value in gates.values())
            or status.get("dispatch_count") is None
            or (selector == "candidate" and status["dispatch_count"] <= 0)
            or (selector == "control" and status["dispatch_count"] != 0)):
        reasons.append(f"{selector}: runner status or lifecycle is invalid")
    if (manifest.get("schema") != "bi100-attention-operator-runtime-v1"
            or manifest.get("version") != 1
            or manifest.get("change_scope") != "attention_operator"
            or manifest.get("max_model_len") != 262144
            or manifest.get("block_size") != 16
            or manifest.get("tensor_parallel_size") != 4
            or not isinstance(manifest.get("command"), list)
            or not manifest["command"]
            or not isinstance(manifest.get("environment"), dict)
            or manifest["environment"].get(
                "BI100_ATTN_COREX_FUSED_PREFILL")
            != ("1" if selector == "candidate" else "0")):
        reasons.append(f"{selector}: runtime manifest is invalid")
    evaluated = service.evaluate(
        measurement, targets=targets, repetitions=repetitions)
    if not evaluated["qualified"]:
        reasons.append(f"{selector}: " + "; ".join(evaluated["reasons"]))
    if isinstance(measurement, dict) and measurement.get("selector") != selector:
        reasons.append(f"{selector}: measurement selector differs")
    return reasons


def qualify(
    statuses: dict[str, dict[str, Any]],
    manifests: dict[str, dict[str, Any]],
    measurements: dict[str, dict[str, Any]],
    *,
    targets: tuple[int, ...] = service.TARGETS,
    repetitions: int = service.REPETITIONS,
) -> dict[str, Any]:
    reasons = []
    for selector in ("control", "candidate"):
        reasons.extend(_arm_reasons(
            selector, statuses.get(selector), manifests.get(selector),
            measurements.get(selector), targets, repetitions))
    control_status = statuses.get("control") or {}
    candidate_status = statuses.get("candidate") or {}
    control_manifest = manifests.get("control") or {}
    candidate_manifest = manifests.get("candidate") or {}
    for field in ("source_revision", "source_dirty_summary",
                  "runtime_identity", "instance", "model_path",
                  "workload_id", "session_preflight_id"):
        if (not control_status.get(field)
                or control_status.get(field) != candidate_status.get(field)):
            reasons.append(f"cross-arm {field} differs or is empty")
    for field in ("source_revision", "runtime_identity", "instance",
                  "model_path", "tokenizer_path", "command"):
        if (not control_manifest.get(field)
                or control_manifest.get(field) != candidate_manifest.get(field)):
            reasons.append(f"cross-arm runtime {field} differs or is empty")
    control_environment = dict(control_manifest.get("environment") or {})
    candidate_environment = dict(candidate_manifest.get("environment") or {})
    control_environment.pop("BI100_ATTN_COREX_FUSED_PREFILL", None)
    candidate_environment.pop("BI100_ATTN_COREX_FUSED_PREFILL", None)
    if control_environment != candidate_environment:
        reasons.append("cross-arm non-candidate environment differs")
    for field in ("workload_id", "targets", "repetitions", "max_tokens",
                  "seed", "workload_order", "expected_requests",
                  "attempted_requests", "completed_requests",
                  "failed_requests"):
        left = (measurements.get("control") or {}).get(field)
        right = (measurements.get("candidate") or {}).get(field)
        if left is None or left != right:
            reasons.append(f"cross-arm workload {field} differs")
    if reasons:
        candidate_failure = (
            (statuses.get("control") or {}).get("qualified") is True
            and ((statuses.get("candidate") or {}).get("result_status") == "fail"
                 or any(reason.startswith("candidate:") for reason in reasons)))
        return {
            "schema": SCHEMA, "version": 1,
            "status": "fail" if candidate_failure else "invalid",
            "classification": ("candidate_hard_failure"
                               if candidate_failure else "invalid_evidence"),
            "reasons": reasons,
            "long_context_authorized": False,
        }
    control = _case_map(measurements["control"])
    candidate = _case_map(measurements["candidate"])
    if not control or set(control) != set(candidate):
        return {
            "schema": SCHEMA, "version": 1, "status": "invalid",
            "classification": "invalid_evidence",
            "reasons": ["paired request identities differ"],
            "long_context_authorized": False,
        }
    identities = sorted(control)
    if any(not _finite(control[item].get("ttft_s"))
           or not _finite(candidate[item].get("ttft_s"))
           or control[item]["ttft_s"] <= 0
           or candidate[item]["ttft_s"] <= 0 for item in identities):
        return {
            "schema": SCHEMA, "version": 1, "status": "invalid",
            "classification": "invalid_evidence",
            "reasons": ["paired TTFT sample is non-finite or non-positive"],
            "long_context_authorized": False,
        }
    gains = [control[item]["ttft_s"] / candidate[item]["ttft_s"] - 1.0
             for item in identities]
    first_token_matches = sum(
        control[item]["first_token_sha256"]
        == candidate[item]["first_token_sha256"]
        for item in identities)
    output_matches = sum(
        control[item]["output_sha256"] == candidate[item]["output_sha256"]
        for item in identities)
    point = statistics.mean(gains)
    lower = _bootstrap_lower(gains)
    buckets = []
    bucket_stable = True
    for target in targets:
        target_ids = [item for item in identities if item[0] == target]
        target_gains = [control[item]["ttft_s"]
                        / candidate[item]["ttft_s"] - 1.0
                        for item in target_ids]
        mean_gain = statistics.mean(target_gains)
        stable = mean_gain > 0.0 and min(target_gains) > -0.05
        bucket_stable &= stable
        buckets.append({
            "target_prompt_tokens": target,
            "control_ttft_s": [control[item]["ttft_s"] for item in target_ids],
            "candidate_ttft_s": [candidate[item]["ttft_s"] for item in target_ids],
            "paired_gains": target_gains,
            "mean_gain": mean_gain,
            "stable": stable,
        })
    if point < 0.02:
        status, classification = "fail", "gain_below_two_percent"
        decision_reasons = ["aggregate TTFT gain is below 2%"]
    elif point < 0.05:
        status, classification = "inconclusive", "gain_two_to_five_percent"
        decision_reasons = ["aggregate TTFT gain is in the 2%-5% gray zone"]
    elif lower <= 0.0 or not bucket_stable:
        status, classification = "inconclusive", "noisy_or_unstable_gain"
        decision_reasons = ["gain is not positive-CI and stable in every length"]
    else:
        status, classification = "pass", "focused_tp4_gain_pass"
        decision_reasons = []
    return {
        "schema": SCHEMA,
        "version": 1,
        "status": status,
        "classification": classification,
        "change_scope": "attention_operator",
        "source_revision": control_status["source_revision"],
        "source_dirty_summary": control_status["source_dirty_summary"],
        "runtime_identity": control_status["runtime_identity"],
        "instance": control_status["instance"],
        "model_path": control_status["model_path"],
        "workload_id": control_status["workload_id"],
        "session_preflight_id": control_status["session_preflight_id"],
        "targets": list(targets),
        "repetitions": repetitions,
        "request_population_per_arm": len(targets) * repetitions,
        "service_startups_per_arm": 1,
        "dispatch": {
            "control": control_status["dispatch_count"],
            "candidate": candidate_status["dispatch_count"],
        },
        "performance": {
            "estimator": "mean_of_paired_control_over_candidate_ttft_gains",
            "paired_gains": gains,
            "aggregate_gain": point,
            "one_sided_95_lower_ci": lower,
            "buckets": buckets,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "distribution": {"status": "not_run",
                         "reason": "run only after focused performance survives",
                         "first_token_match_count": first_token_matches,
                         "full_output_match_count": output_matches,
                         "paired_request_count": len(identities)},
        "capability": {"status": "not_run",
                       "reason": "outside attention-operator development scope"},
        "reasons": decision_reasons,
        "long_context_authorized": status == "pass",
    }


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    for selector in ("control", "candidate"):
        parser.add_argument(f"--{selector}-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", choices=("short", "long"),
                        default="short")
    args = parser.parse_args()
    try:
        roots = {name: getattr(args, f"{name}_root")
                 for name in ("control", "candidate")}
        targets, repetitions = (
            (service.TARGETS, service.REPETITIONS)
            if args.profile == "short" else ((131072, 235000), 2)
        )
        result = qualify(
            {name: _load(root / "runner_status.json")
             for name, root in roots.items()},
            {name: _load(root / "runtime_manifest.json")
             for name, root in roots.items()},
            {name: _load(root / "measurement.json")
             for name, root in roots.items()},
            targets=targets,
            repetitions=repetitions,
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        result = {
            "schema": SCHEMA, "version": 1, "status": "invalid",
            "classification": "invalid_evidence", "reasons": [str(exc)],
            "long_context_authorized": False,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({"status": result["status"],
                      "classification": result["classification"],
                      "reasons": result.get("reasons", [])}, sort_keys=True))
    return {"pass": 0, "fail": 1, "invalid": 2, "inconclusive": 3}[
        result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
