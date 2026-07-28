#!/usr/bin/env python3
"""Qualify a same-policy fused-prefill TP4 functional and Agent A/B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import compare_admission64_quality_service_ab as base


Json = dict[str, Any]
SCHEMA = "bi100-fused-prefill-quality-service-ab-v1"
VERSION = 1
CONTROL_LABEL = "m1-112-control-fused-off"
CANDIDATE_LABEL = "m1-112-candidate-fused-on"
COMMON_OPTIMIZATION = {
    "gdn_cache_policy": "admission64",
    "gdn_restore_mode": "hybrid64",
    "kv_eviction_policy": "lru",
    "kernel_profile": "submission",
}


def _optimization(fused_prefill: str) -> dict[str, str]:
    return {
        **COMMON_OPTIMIZATION,
        "fused_prefill": fused_prefill,
    }


def compare(
    *,
    control_status: Any,
    candidate_status: Any,
    control_contract: Any,
    candidate_contract: Any,
    control_4xx: Any,
    candidate_4xx: Any,
    control_process_identity: Any,
    candidate_process_identity: Any,
    quality_comparison: Any,
    agent_comparison: Any,
    file_sha256s: dict[str, dict[str, str]],
) -> Json:
    reasons = base._status_reasons(
        control_status,
        label="control",
        expected_policy="admission64",
        expected_label=CONTROL_LABEL,
        file_sha256s=file_sha256s["control"],
        expected_optimization=_optimization("0"),
    )
    reasons.extend(base._status_reasons(
        candidate_status,
        label="candidate",
        expected_policy="admission64",
        expected_label=CANDIDATE_LABEL,
        file_sha256s=file_sha256s["candidate"],
        expected_optimization=_optimization("1"),
    ))
    if isinstance(control_status, dict):
        reasons.extend(base._runtime_reasons(
            control_contract,
            control_status,
            label="control",
            expected_policy="admission64",
            expected_restore_mode="hybrid64",
            expected_fused_prefill="0",
        ))
    if isinstance(candidate_status, dict):
        reasons.extend(base._runtime_reasons(
            candidate_contract,
            candidate_status,
            label="candidate",
            expected_policy="admission64",
            expected_restore_mode="hybrid64",
            expected_fused_prefill="1",
        ))

    reasons.extend(base._comparison_reasons(
        quality_comparison,
        label="quality",
        expected_schema=base.quality_compare.COMPARISON_SCHEMA,
        authorization_field="quality_non_regression_authorized",
        baseline_sha256=file_sha256s["control"]["quality_report"],
        candidate_sha256=file_sha256s["candidate"]["quality_report"],
        expected_ids=base.EXPECTED_QUALITY_IDS,
        has_failed_count=True,
    ))
    reasons.extend(base._comparison_reasons(
        agent_comparison,
        label="Agent",
        expected_schema=base.agent_compare.SCHEMA,
        authorization_field="agent_quality_non_regression_authorized",
        baseline_sha256=file_sha256s["control"]["agent_workload"],
        candidate_sha256=file_sha256s["candidate"]["agent_workload"],
        expected_ids=base.EXPECTED_AGENT_IDS,
        has_failed_count=False,
    ))
    reasons.extend(base._api_4xx_reasons(control_4xx, "control"))
    reasons.extend(base._api_4xx_reasons(candidate_4xx, "candidate"))
    reasons.extend(base._process_identity_reasons(
        control_process_identity, "control"))
    reasons.extend(base._process_identity_reasons(
        candidate_process_identity, "candidate"))

    if (
        isinstance(control_process_identity, dict)
        and isinstance(candidate_process_identity, dict)
        and control_process_identity.get("session_token")
        == candidate_process_identity.get("session_token")
    ):
        reasons.append("A/B process session tokens must differ")

    fused_only_delta = False
    if (
        isinstance(control_status, dict)
        and isinstance(candidate_status, dict)
        and isinstance(control_contract, dict)
        and isinstance(candidate_contract, dict)
    ):
        for field in ("source_revision", "source_branch", "instance"):
            if control_status.get(field) != candidate_status.get(field):
                reasons.append(f"A/B status differs in {field}")

        normalized_control = json.loads(json.dumps(control_contract))
        normalized_candidate = json.loads(json.dumps(candidate_contract))
        control_env = normalized_control.get("environment")
        candidate_env = normalized_candidate.get("environment")
        changed_environment: set[str] = set()
        if isinstance(control_env, dict) and isinstance(candidate_env, dict):
            changed_environment = {
                name for name in set(control_env) | set(candidate_env)
                if control_env.get(name) != candidate_env.get(name)
            }
            control_env["BI100_ATTN_COREX_FUSED_PREFILL"] = "1"
        else:
            reasons.append("A/B runtime environments are invalid")
        normalized_control["optimization_label"] = normalized_candidate.get(
            "optimization_label")
        if changed_environment != {"BI100_ATTN_COREX_FUSED_PREFILL"}:
            reasons.append(
                "A/B environment delta must contain only "
                "BI100_ATTN_COREX_FUSED_PREFILL")
        if normalized_control != normalized_candidate:
            reasons.append(
                "A/B runtime contracts differ beyond fused-prefill and label")
        fused_only_delta = (
            changed_environment == {"BI100_ATTN_COREX_FUSED_PREFILL"}
            and normalized_control == normalized_candidate
        )

    if isinstance(control_4xx, dict) and isinstance(candidate_4xx, dict):
        if control_4xx != candidate_4xx:
            reasons.append("A/B 4xx attribution or request shapes differ")

    qualified = not reasons and fused_only_delta
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "qualified": qualified,
        "fused_prefill_quality_non_regression_authorized": qualified,
        "fused_only_runtime_delta_attested": fused_only_delta,
        "performance_authorized": False,
        "default_change_authorized": False,
        "production_promotion_authorized": False,
        "limits": {
            "gpu_count": 4,
            "tensor_parallel_size": 4,
            "max_model_len": 262144,
            "gdn_cache_policy": "admission64",
            "gdn_restore_mode": "hybrid64",
            "required_runtime_delta": "fused-prefill 0->1",
            "kv_eviction_policy": "lru",
            "kernel_profile": "submission",
        },
        "observed": {
            "source_revision": (
                control_status.get("source_revision")
                if isinstance(control_status, dict) else None
            ),
            "instance": (
                control_status.get("instance")
                if isinstance(control_status, dict) else None
            ),
            "control_4xx_count": (
                control_4xx.get("chat_4xx_access_count")
                if isinstance(control_4xx, dict) else None
            ),
            "candidate_4xx_count": (
                candidate_4xx.get("chat_4xx_access_count")
                if isinstance(candidate_4xx, dict) else None
            ),
        },
        "reasons": reasons,
        "privacy": {
            "contains_raw_requests": False,
            "contains_raw_model_outputs": False,
            "contains_credentials": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--quality-comparison", type=Path, required=True)
    parser.add_argument("--agent-comparison", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    control_paths = base._arm_paths(args.control_root)
    candidate_paths = base._arm_paths(args.candidate_root)
    file_sha256s = {
        "control": {
            name: base._file_sha256(path)
            for name, path in control_paths.items()
            if name != "status"
        },
        "candidate": {
            name: base._file_sha256(path)
            for name, path in candidate_paths.items()
            if name != "status"
        },
    }
    result = compare(
        control_status=base._load(control_paths["status"]),
        candidate_status=base._load(candidate_paths["status"]),
        control_contract=base._load(control_paths["runtime_contract"]),
        candidate_contract=base._load(candidate_paths["runtime_contract"]),
        control_4xx=base._load(control_paths["api_4xx_attribution"]),
        candidate_4xx=base._load(candidate_paths["api_4xx_attribution"]),
        control_process_identity=base._load(
            control_paths["process_group_identity"]),
        candidate_process_identity=base._load(
            candidate_paths["process_group_identity"]),
        quality_comparison=base._load(args.quality_comparison),
        agent_comparison=base._load(args.agent_comparison),
        file_sha256s=file_sha256s,
    )
    result["inputs"] = {
        "control_status_sha256": base._file_sha256(control_paths["status"]),
        "candidate_status_sha256": base._file_sha256(
            candidate_paths["status"]),
        "control_runtime_contract_sha256": file_sha256s["control"][
            "runtime_contract"],
        "candidate_runtime_contract_sha256": file_sha256s["candidate"][
            "runtime_contract"],
        "control_process_group_identity_sha256": file_sha256s["control"][
            "process_group_identity"],
        "candidate_process_group_identity_sha256": file_sha256s["candidate"][
            "process_group_identity"],
        "control_service_recovery_sha256": file_sha256s["control"][
            "service_recovery"],
        "candidate_service_recovery_sha256": file_sha256s["candidate"][
            "service_recovery"],
        "control_service_recovery_clean_sha256": file_sha256s["control"][
            "service_recovery_clean"],
        "candidate_service_recovery_clean_sha256": file_sha256s["candidate"][
            "service_recovery_clean"],
        "quality_comparison_sha256": base._file_sha256(
            args.quality_comparison),
        "agent_comparison_sha256": base._file_sha256(args.agent_comparison),
    }
    base._atomic_write(args.out, result)
    print(json.dumps({
        "qualified": result["qualified"],
        "fused_only_runtime_delta_attested": result[
            "fused_only_runtime_delta_attested"],
        "reasons": result["reasons"],
        "out": str(args.out),
    }, indent=2, sort_keys=True))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
