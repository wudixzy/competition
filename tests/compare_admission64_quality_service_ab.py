#!/usr/bin/env python3
"""Qualify the fixed-policy M1-85 TP4 functional and Agent A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import compare_agent_workload_reports as agent_compare
import compare_quality_gate_reports as quality_compare
import quality_runtime_contract as runtime_contract
import summarize_api_4xx_log as api_4xx


Json = dict[str, Any]
SCHEMA = "bi100-admission64-quality-service-ab-v2"
VERSION = 2
STATUS_SCHEMA = "bi100-quality-service-gate-status-v2"
EXPECTED_STATUS_GATES = {
    "runtime_identity",
    "runtime_contract",
    "prefix_allocator",
    "gdn_action_broadcast",
    "preflight_before",
    "process_group",
    "startup",
    "startup_contract",
    "quality",
    "agent_workload",
    "api_4xx_attribution",
    "cleanup",
    "service_recovery",
    "service_recovery_clean",
    "service_postflight",
    "fatal_scan",
    "timeout_scan",
    "preflight_after",
    "preflight_comparison",
}
EXPECTED_COMMON_OPTIMIZATION = {
    "gdn_restore_mode": "direct",
    "fused_prefill": "0",
    "kv_eviction_policy": "lru",
    "kernel_profile": "submission",
}
EXPECTED_4XX_FIELDS = {
    "schema",
    "version",
    "complete",
    "classified",
    "qualified",
    "chat_4xx_access_count",
    "attributed_count",
    "attribution_delta",
    "malformed_marker_count",
    "by_access_code",
    "by_attributed_code",
    "by_endpoint",
    "by_reason",
    "request_shapes",
    "privacy",
}
EXPECTED_4XX_PRIVACY = {
    "contains_raw_log_lines": False,
    "contains_request_content": False,
    "contains_response_content": False,
    "contains_tool_schema": False,
    "contains_multimodal_url_or_bytes": False,
}
EXPECTED_QUALITY_COMPARISON_PRIVACY = {
    "contains_raw_requests": False,
    "contains_raw_model_outputs": False,
    "contains_credentials": False,
}
EXPECTED_PROCESS_IDENTITY_FIELDS = {
    "schema",
    "version",
    "pid",
    "pgid",
    "sid",
    "starttime_ticks",
    "session_token",
}
EXPECTED_AGENT_CASES = len(
    agent_compare.workload.load_manifest(
        agent_compare.workload.DEFAULT_MANIFEST)[0]["cases"]
)
EXPECTED_QUALITY_IDS = [
    case["id"]
    for case in quality_compare._load_manifest(
        quality_compare.DEFAULT_MANIFEST)[0]["cases"]
]
EXPECTED_AGENT_IDS = [
    case["id"]
    for case in agent_compare.workload.load_manifest(
        agent_compare.workload.DEFAULT_MANIFEST)[0]["cases"]
]


def _atomic_write(path: Path, value: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=True, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_zero_rc(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == 0


def _load(path: Path) -> Json:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


def _status_reasons(
    status: Any,
    *,
    label: str,
    expected_policy: str,
    expected_label: str,
    file_sha256s: dict[str, str],
) -> list[str]:
    reasons: list[str] = []
    if not isinstance(status, dict):
        return [f"{label}: status root must be an object"]
    if status.get("schema") != STATUS_SCHEMA or status.get("version") != 2:
        reasons.append(f"{label}: status schema or version is invalid")
    if status.get("suite") != "functional":
        reasons.append(f"{label}: suite must be functional")
    if not _is_zero_rc(status.get("overall_rc")):
        reasons.append(f"{label}: quality service runner did not pass")
    if not runtime_contract.is_git_revision(status.get("source_revision")):
        reasons.append(f"{label}: source revision is invalid")
    for field in ("source_branch", "label", "instance"):
        if not isinstance(status.get(field), str) or not status[field]:
            reasons.append(f"{label}: {field} is missing")
    if status.get("label") != expected_label:
        reasons.append(f"{label}: fixed experiment label differs")

    optimization = status.get("optimization")
    expected_optimization = {
        **EXPECTED_COMMON_OPTIMIZATION,
        "gdn_cache_policy": expected_policy,
    }
    if optimization != expected_optimization:
        reasons.append(f"{label}: optimization contract differs")

    gates = status.get("gates")
    if not isinstance(gates, dict) or set(gates) != EXPECTED_STATUS_GATES:
        reasons.append(f"{label}: status gate set is invalid")
    elif any(not _is_zero_rc(value) for value in gates.values()):
        failed = sorted(
            name for name, value in gates.items() if not _is_zero_rc(value))
        reasons.append(f"{label}: gates did not pass: {', '.join(failed)}")

    artifacts = status.get("artifacts")
    if (
        set(file_sha256s) != {
            "runtime_contract",
            "quality_report",
            "agent_workload",
            "api_4xx_attribution",
            "process_group_identity",
            "service_recovery",
            "service_recovery_clean",
        }
        or any(not _is_sha256(value) for value in file_sha256s.values())
    ):
        reasons.append(f"{label}: computed artifact identities are invalid")
    expected_artifacts = {
        "runtime_contract_sha256": file_sha256s["runtime_contract"],
        "quality_report_sha256": file_sha256s["quality_report"],
        "agent_workload_sha256": file_sha256s["agent_workload"],
        "api_4xx_attribution_sha256": file_sha256s["api_4xx_attribution"],
        "process_group_identity_sha256": file_sha256s[
            "process_group_identity"],
        "service_recovery_sha256": file_sha256s["service_recovery"],
        "service_recovery_clean_sha256": file_sha256s[
            "service_recovery_clean"],
    }
    if artifacts != expected_artifacts:
        reasons.append(f"{label}: status artifact bindings differ")
    if status.get("privacy") != {
        "raw_service_log_outside_repository": True,
        "contains_credentials": False,
    }:
        reasons.append(f"{label}: status privacy declaration is invalid")
    return reasons


def _runtime_reasons(
    contract: Any,
    status: Json,
    *,
    label: str,
    expected_policy: str,
) -> list[str]:
    if not isinstance(contract, dict):
        return [f"{label}: runtime contract root must be an object"]
    if (
        not isinstance(contract.get("runtime_identity"), str)
        or not contract["runtime_identity"]
    ):
        return [f"{label}: runtime identity is missing"]
    expected = {
        "source_revision": status.get("source_revision"),
        "instance": status.get("instance"),
        "gpu_count": 4,
        "tensor_parallel_size": 4,
        "max_model_len": 262144,
        "model_path": contract.get("model_path"),
        "tokenizer_path": contract.get("tokenizer_path"),
        "served_model_name": "llm",
    }
    reasons: list[str] = []
    try:
        runtime_contract.validate_runtime_contract(
            contract, expected, require_cache_trace=True)
    except runtime_contract.RuntimeContractError as error:
        reasons.append(f"{label}: {error}")
        return reasons

    environment = contract["environment"]
    expected_environment = runtime_contract.service_environment(
        environment["BI100_RUNTIME_SITE_PACKAGES"],
        gdn_cache_policy=expected_policy,
        gdn_restore_mode="direct",
        fused_prefill="0",
        kv_eviction_policy="lru",
        kernel_profile="submission",
    )
    if environment != expected_environment:
        reasons.append(f"{label}: fixed service environment differs")
    if contract.get("optimization_label") != status.get("label"):
        reasons.append(f"{label}: optimization label differs from status")
    return reasons


def _comparison_reasons(
    report: Any,
    *,
    label: str,
    expected_schema: str,
    authorization_field: str,
    baseline_sha256: str,
    candidate_sha256: str,
    expected_ids: list[str],
    has_failed_count: bool,
) -> list[str]:
    reasons: list[str] = []
    if not isinstance(report, dict):
        return [f"{label}: comparison root must be an object"]
    if report.get("schema") != expected_schema or report.get("version") != 1:
        reasons.append(f"{label}: comparison schema or version is invalid")
    if report.get("qualified") is not True:
        reasons.append(f"{label}: comparison did not qualify")
    if report.get(authorization_field) is not True:
        reasons.append(f"{label}: non-regression was not authorized")
    if report.get("overall_promotion_authorized") is not False:
        reasons.append(f"{label}: comparison authorized overall promotion")
    if report.get("reasons") != []:
        reasons.append(f"{label}: comparison contains rejection reasons")
    summary = report.get("summary")
    expected_summary = {
        "compared_cases": len(expected_ids),
        "qualified_cases": len(expected_ids),
    }
    if has_failed_count:
        expected_summary["failed_cases"] = 0
    if summary != expected_summary:
        reasons.append(f"{label}: comparison summary is incomplete")
    cases = report.get("cases")
    if (
        not isinstance(cases, list)
        or len(cases) != len(expected_ids)
        or [
            case.get("id") if isinstance(case, dict) else None
            for case in cases
        ] != expected_ids
        or any(
            not isinstance(case, dict)
            or case.get("qualified") is not True
            or case.get("reasons") != []
            for case in cases
        )
    ):
        reasons.append(f"{label}: qualified case evidence is incomplete")
    if (
        expected_schema == quality_compare.COMPARISON_SCHEMA
        and report.get("privacy") != EXPECTED_QUALITY_COMPARISON_PRIVACY
    ):
        reasons.append(f"{label}: comparison privacy declaration is invalid")
    inputs = report.get("inputs")
    if not isinstance(inputs, dict):
        reasons.append(f"{label}: input bindings are missing")
    elif (
        not _is_sha256(inputs.get("baseline_file_sha256"))
        or not _is_sha256(inputs.get("candidate_file_sha256"))
        or inputs.get("baseline_file_sha256") != baseline_sha256
        or inputs.get("candidate_file_sha256") != candidate_sha256
    ):
        reasons.append(f"{label}: input bindings differ")
    return reasons


def _api_4xx_reasons(report: Any, label: str) -> list[str]:
    reasons: list[str] = []
    if not isinstance(report, dict):
        return [f"{label}: 4xx report root must be an object"]
    if set(report) != EXPECTED_4XX_FIELDS:
        reasons.append(f"{label}: 4xx report fields are invalid")
    if (
        report.get("schema") != api_4xx.REPORT_SCHEMA
        or report.get("version") != api_4xx.REPORT_VERSION
    ):
        reasons.append(f"{label}: 4xx report schema or version is invalid")
    if (
        report.get("qualified") is not True
        or report.get("complete") is not True
        or report.get("classified") is not True
        or report.get("attribution_delta") != 0
        or report.get("malformed_marker_count") != 0
    ):
        reasons.append(f"{label}: 4xx attribution is incomplete")
    if report.get("privacy") != EXPECTED_4XX_PRIVACY:
        reasons.append(f"{label}: 4xx privacy declaration is invalid")
    access = report.get("chat_4xx_access_count")
    attributed = report.get("attributed_count")
    if (
        not isinstance(access, int)
        or isinstance(access, bool)
        or access <= 0
        or attributed != access
    ):
        reasons.append(f"{label}: 4xx counts are invalid")
    access_codes = report.get("by_access_code")
    attributed_codes = report.get("by_attributed_code")
    endpoints = report.get("by_endpoint")
    by_reason = report.get("by_reason")
    shapes = report.get("request_shapes")
    mappings = (access_codes, attributed_codes, endpoints, by_reason)
    if (
        any(
            not isinstance(mapping, dict)
            or any(
                not isinstance(key, str)
                or not key
                or not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                for key, value in mapping.items()
            )
            for mapping in mappings
        )
        or not isinstance(shapes, list)
        or any(not isinstance(shape, dict) for shape in shapes)
    ):
        reasons.append(f"{label}: 4xx attribution details are invalid")
    elif (
        sum(access_codes.values()) != access
        or sum(attributed_codes.values()) != attributed
        or sum(endpoints.values()) != attributed
        or sum(by_reason.values()) != attributed
        or by_reason.get("unclassified_chat_error", 0) != 0
    ):
        reasons.append(f"{label}: 4xx attribution totals differ")
    return reasons


def _process_identity_reasons(identity: Any, label: str) -> list[str]:
    if not isinstance(identity, dict):
        return [f"{label}: process identity root must be an object"]
    reasons: list[str] = []
    if set(identity) != EXPECTED_PROCESS_IDENTITY_FIELDS:
        reasons.append(f"{label}: process identity fields are invalid")
    pid = identity.get("pid")
    starttime = identity.get("starttime_ticks")
    token = identity.get("session_token")
    if (
        identity.get("schema") != "bi100-process-session-v1"
        or identity.get("version") != 1
        or not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 1
        or identity.get("pgid") != pid
        or identity.get("sid") != pid
        or not isinstance(starttime, int)
        or isinstance(starttime, bool)
        or starttime <= 0
        or not isinstance(token, str)
        or len(token) != 32
        or any(character not in "0123456789abcdef" for character in token)
    ):
        reasons.append(f"{label}: process identity is not attested")
    return reasons


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
    reasons = _status_reasons(
        control_status,
        label="control",
        expected_policy="fine32",
        expected_label="m1-85-control-fine32",
        file_sha256s=file_sha256s["control"],
    )
    reasons.extend(_status_reasons(
        candidate_status,
        label="candidate",
        expected_policy="admission64",
        expected_label="m1-85-candidate-admission64",
        file_sha256s=file_sha256s["candidate"],
    ))
    if isinstance(control_status, dict):
        reasons.extend(_runtime_reasons(
            control_contract,
            control_status,
            label="control",
            expected_policy="fine32",
        ))
    if isinstance(candidate_status, dict):
        reasons.extend(_runtime_reasons(
            candidate_contract,
            candidate_status,
            label="candidate",
            expected_policy="admission64",
        ))

    reasons.extend(_comparison_reasons(
        quality_comparison,
        label="quality",
        expected_schema=quality_compare.COMPARISON_SCHEMA,
        authorization_field="quality_non_regression_authorized",
        baseline_sha256=file_sha256s["control"]["quality_report"],
        candidate_sha256=file_sha256s["candidate"]["quality_report"],
        expected_ids=EXPECTED_QUALITY_IDS,
        has_failed_count=True,
    ))
    reasons.extend(_comparison_reasons(
        agent_comparison,
        label="Agent",
        expected_schema=agent_compare.SCHEMA,
        authorization_field="agent_quality_non_regression_authorized",
        baseline_sha256=file_sha256s["control"]["agent_workload"],
        candidate_sha256=file_sha256s["candidate"]["agent_workload"],
        expected_ids=EXPECTED_AGENT_IDS,
        has_failed_count=False,
    ))
    reasons.extend(_api_4xx_reasons(control_4xx, "control"))
    reasons.extend(_api_4xx_reasons(candidate_4xx, "candidate"))
    reasons.extend(_process_identity_reasons(
        control_process_identity, "control"))
    reasons.extend(_process_identity_reasons(
        candidate_process_identity, "candidate"))
    if (
        isinstance(control_process_identity, dict)
        and isinstance(candidate_process_identity, dict)
        and control_process_identity.get("session_token")
        == candidate_process_identity.get("session_token")
    ):
        reasons.append("A/B process session tokens must differ")

    policy_only_delta = False
    if (
        isinstance(control_status, dict)
        and isinstance(candidate_status, dict)
        and isinstance(control_contract, dict)
        and isinstance(candidate_contract, dict)
    ):
        for field in (
            "source_revision",
            "source_branch",
            "instance",
        ):
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
            normalized_control["environment"][
                "BI100_GDN_CACHE_POLICY"] = "admission64"
        else:
            reasons.append("A/B runtime environments are invalid")
        normalized_control["optimization_label"] = normalized_candidate.get(
            "optimization_label")
        if changed_environment != {"BI100_GDN_CACHE_POLICY"}:
            reasons.append(
                "A/B environment delta must contain only "
                "BI100_GDN_CACHE_POLICY")
        if normalized_control != normalized_candidate:
            reasons.append(
                "A/B runtime contracts differ beyond policy and label")
        policy_only_delta = (
            changed_environment == {"BI100_GDN_CACHE_POLICY"}
            and normalized_control == normalized_candidate
        )

    if isinstance(control_4xx, dict) and isinstance(candidate_4xx, dict):
        if control_4xx != candidate_4xx:
            reasons.append("A/B 4xx attribution or request shapes differ")

    qualified = not reasons and policy_only_delta
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "qualified": qualified,
        "admission64_quality_non_regression_authorized": qualified,
        "policy_only_runtime_delta_attested": policy_only_delta,
        "performance_authorized": False,
        "default_policy_change_authorized": False,
        "production_promotion_authorized": False,
        "limits": {
            "gpu_count": 4,
            "tensor_parallel_size": 4,
            "max_model_len": 262144,
            "required_policy_delta": "fine32->admission64",
            "restore_mode": "direct",
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


def _arm_paths(root: Path) -> dict[str, Path]:
    return {
        "status": root / "status.json",
        "runtime_contract": root / "runtime_contract.json",
        "quality_report": root / "quality_report.json",
        "agent_workload": root / "agent_workload.json",
        "api_4xx_attribution": root / "api_4xx_attribution.json",
        "process_group_identity": root / "process_group_identity.json",
        "service_recovery": root / "service_recovery.json",
        "service_recovery_clean": root / "service_recovery_clean.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--quality-comparison", type=Path, required=True)
    parser.add_argument("--agent-comparison", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    control_paths = _arm_paths(args.control_root)
    candidate_paths = _arm_paths(args.candidate_root)
    file_sha256s = {
        "control": {
            name: _file_sha256(path)
            for name, path in control_paths.items()
            if name != "status"
        },
        "candidate": {
            name: _file_sha256(path)
            for name, path in candidate_paths.items()
            if name != "status"
        },
    }
    result = compare(
        control_status=_load(control_paths["status"]),
        candidate_status=_load(candidate_paths["status"]),
        control_contract=_load(control_paths["runtime_contract"]),
        candidate_contract=_load(candidate_paths["runtime_contract"]),
        control_4xx=_load(control_paths["api_4xx_attribution"]),
        candidate_4xx=_load(candidate_paths["api_4xx_attribution"]),
        control_process_identity=_load(
            control_paths["process_group_identity"]),
        candidate_process_identity=_load(
            candidate_paths["process_group_identity"]),
        quality_comparison=_load(args.quality_comparison),
        agent_comparison=_load(args.agent_comparison),
        file_sha256s=file_sha256s,
    )
    result["inputs"] = {
        "control_status_sha256": _file_sha256(control_paths["status"]),
        "candidate_status_sha256": _file_sha256(candidate_paths["status"]),
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
        "quality_comparison_sha256": _file_sha256(args.quality_comparison),
        "agent_comparison_sha256": _file_sha256(args.agent_comparison),
    }
    _atomic_write(args.out, result)
    print(json.dumps({
        "qualified": result["qualified"],
        "policy_only_runtime_delta_attested": result[
            "policy_only_runtime_delta_attested"],
        "reasons": result["reasons"],
        "out": str(args.out),
    }, indent=2, sort_keys=True))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
