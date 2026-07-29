#!/usr/bin/env python3
"""Qualify the M1-122 same-runtime fused-prefill IFEval TP4 A/B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import compare_admission64_quality_service_ab as service
import compare_ifeval_reports as ifeval
import compare_ifeval_paired_noninferiority as paired_ifeval
import summarize_api_4xx_log as api_4xx


Json = dict[str, Any]
SCHEMA = "bi100-m1-122-ifeval-fused-prefill-ab-v2"
VERSION = 2
EXPECTED_GATES = {
    "runtime_identity",
    "runtime_contract",
    "prefix_allocator",
    "gdn_action_broadcast",
    "preflight_before",
    "process_group",
    "startup",
    "startup_contract",
    "ifeval_environment",
    "ifeval",
    "api_4xx_attribution",
    "cleanup",
    "service_recovery",
    "service_recovery_clean",
    "service_postflight",
    "fatal_scan",
    "timeout_scan",
    "checkpoint_cleanup",
    "preflight_after",
    "preflight_comparison",
}
EXPECTED_ARTIFACTS = {
    "runtime_contract",
    "ifeval_install",
    "ifeval_report",
    "ifeval_progress",
    "api_4xx_attribution",
    "process_group_identity",
    "service_recovery",
    "service_recovery_clean",
}
EXPECTED_OPTIMIZATION = {
    "gdn_cache_policy": "admission64",
    "gdn_restore_mode": "hybrid64",
    "kv_eviction_policy": "lru",
    "kernel_profile": "submission",
}
EXPECTED_PRIVACY = {
    "raw_service_log_outside_repository": True,
    "contains_credentials": False,
    "raw_checkpoint_absent_after_lifecycle": True,
}
LAYERED_CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "quality/layered_quality_gate.v1.json"
)
EXPECTED_LAYERED_CONTRACT_SHA256 = service._file_sha256(LAYERED_CONTRACT)


def _arm_paths(root: Path) -> dict[str, Path]:
    return {
        "status": root / "status.json",
        "runtime_contract": root / "runtime_contract.json",
        "ifeval_report": root / "ifeval_report.json",
        "ifeval_progress": root / "ifeval_progress.json",
        "api_4xx_attribution": root / "api_4xx_attribution.json",
        "process_group_identity": root / "process_group_identity.json",
        "service_recovery": root / "service_recovery.json",
        "service_recovery_clean": root / "service_recovery_clean.json",
    }


def _optimization(fused_prefill: str) -> Json:
    return {
        **EXPECTED_OPTIMIZATION,
        "fused_prefill": fused_prefill,
    }


def _status_reasons(
    status: Any,
    *,
    label: str,
    expected_label: str,
    fused_prefill: str,
    file_sha256s: dict[str, str],
) -> list[str]:
    if not isinstance(status, dict):
        return [f"{label}: status root must be an object"]
    reasons = []
    if (
        status.get("schema") != service.STATUS_SCHEMA
        or status.get("version") != 2
        or status.get("suite") != "ifeval"
        or status.get("overall_rc") != 0
    ):
        reasons.append(f"{label}: service status did not qualify")
    if status.get("label") != expected_label:
        reasons.append(f"{label}: fixed label differs")
    for field in ("source_revision", "source_branch", "instance"):
        if not isinstance(status.get(field), str) or not status[field]:
            reasons.append(f"{label}: {field} is missing")
    if status.get("optimization") != _optimization(fused_prefill):
        reasons.append(f"{label}: optimization contract differs")
    gates = status.get("gates")
    if not isinstance(gates, dict) or set(gates) != EXPECTED_GATES:
        reasons.append(f"{label}: service gate identities differ")
    elif any(value != 0 or isinstance(value, bool) for value in gates.values()):
        reasons.append(f"{label}: one or more service gates failed")
    if set(file_sha256s) != EXPECTED_ARTIFACTS:
        reasons.append(f"{label}: computed artifact identities differ")
    expected_artifacts = {
        f"{name}_sha256": file_sha256s.get(name)
        for name in sorted(EXPECTED_ARTIFACTS)
    }
    if status.get("artifacts") != expected_artifacts:
        reasons.append(f"{label}: service artifact bindings differ")
    if status.get("privacy") != EXPECTED_PRIVACY:
        reasons.append(f"{label}: service privacy contract differs")
    return reasons


def _zero_4xx_reasons(report: Any, label: str) -> list[str]:
    if not isinstance(report, dict):
        return [f"{label}: 4xx report root must be an object"]
    reasons = []
    if set(report) != service.EXPECTED_4XX_FIELDS:
        reasons.append(f"{label}: 4xx report fields are invalid")
    if (
        report.get("schema") != api_4xx.REPORT_SCHEMA
        or report.get("version") != api_4xx.REPORT_VERSION
        or report.get("qualified") is not True
        or report.get("complete") is not True
        or report.get("classified") is not True
        or report.get("chat_4xx_access_count") != 0
        or report.get("attributed_count") != 0
        or report.get("attribution_delta") != 0
        or report.get("malformed_marker_count") != 0
    ):
        reasons.append(f"{label}: IFEval must have zero fully classified 4xx")
    for name in (
        "by_access_code",
        "by_attributed_code",
        "by_endpoint",
        "by_reason",
        "by_validation_field",
        "by_validation_type",
    ):
        if report.get(name) != {}:
            reasons.append(f"{label}: unexpected 4xx detail in {name}")
    if report.get("request_shapes") != []:
        reasons.append(f"{label}: unexpected 4xx request shape")
    if report.get("privacy") != service.EXPECTED_4XX_PRIVACY:
        reasons.append(f"{label}: 4xx privacy contract differs")
    return reasons


def _progress_reasons(
    progress: Any,
    report: Json,
    report_sha256: str,
    label: str,
) -> list[str]:
    if not isinstance(progress, dict):
        return [f"{label}: IFEval progress root must be an object"]
    reasons = []
    if (
        progress.get("schema") != "bi100-ifeval-progress-v1"
        or progress.get("version") != 1
        or progress.get("run_id_sha256") != report.get("run_id_sha256")
        or progress.get("selected") != 64
        or progress.get("attempted") != 64
        or progress.get("successful") != 64
        or progress.get("errors") != 0
        or progress.get("last_ordinal") != 64
        or progress.get("complete") is not True
        or progress.get("report_sha256") != report_sha256
        or progress.get("failures") != []
    ):
        reasons.append(f"{label}: IFEval progress is incomplete")
    privacy = progress.get("privacy") or {}
    if any(
        privacy.get(name) is not False
        for name in (
            "contains_credentials",
            "contains_raw_prompts",
            "contains_raw_model_outputs",
            "contains_reasoning_text",
        )
    ):
        reasons.append(f"{label}: IFEval progress privacy differs")
    return reasons


def _comparison_reasons(
    report: Any,
    *,
    baseline_sha256: str,
    candidate_sha256: str,
    require_exact_output: bool,
    expected_reasons: list[str],
    label: str,
) -> list[str]:
    if not isinstance(report, dict):
        return [f"{label}: comparison root must be an object"]
    reasons = []
    expected_qualified = not expected_reasons
    if (
        report.get("schema") != ifeval.COMPARISON_SCHEMA
        or report.get("version") != 1
        or report.get("qualified") is not expected_qualified
        or report.get("promotion_authorized") is not False
        or report.get("baseline_sha256") != baseline_sha256
        or report.get("candidate_sha256") != candidate_sha256
        or report.get("allowed_switches") != ["fused_prefill"]
        or report.get("require_exact_output") is not require_exact_output
        or report.get("reasons") != expected_reasons
    ):
        reasons.append(f"{label}: comparison evidence differs")
    score_delta = report.get("score_delta")
    if (
        not isinstance(score_delta, dict)
        or set(score_delta)
        != {
            "strict_prompt_passed",
            "loose_prompt_passed",
            "strict_instruction_passed",
            "loose_instruction_passed",
        }
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in score_delta.values()
        )
    ):
        reasons.append(f"{label}: score delta is invalid")
    return reasons


def _paired_noninferiority_reasons(
    report: Any,
    *,
    baseline_sha256: str,
    candidate_sha256: str,
) -> list[str]:
    if not isinstance(report, dict):
        return ["paired non-inferiority report root must be an object"]
    reasons = []
    if (
        report.get("schema") != paired_ifeval.SCHEMA
        or report.get("version") != paired_ifeval.VERSION
        or report.get("status") != "pass"
        or report.get("qualified") is not True
        or report.get("baseline_sha256") != baseline_sha256
        or report.get("candidate_sha256") != candidate_sha256
        or report.get("contract_sha256")
        != EXPECTED_LAYERED_CONTRACT_SHA256
        or report.get("allowed_switches") != ["fused_prefill"]
        or report.get("sample_count") != 64
        or report.get("reasons") != []
    ):
        reasons.append("paired non-inferiority screen did not qualify")
    screen = report.get("screen") or {}
    checks = screen.get("checks") or {}
    if (
        screen.get("confidence") != 0.95
        or screen.get("noninferiority_margin") != 0.05
        or screen.get("bootstrap_samples") != 20000
        or set(checks) != {"strict_prompt", "loose_prompt"}
        or any(
            value.get("status") != "pass"
            or value.get("qualified") is not True
            or value.get("sample_count") != 64
            for value in checks.values()
            if isinstance(value, dict)
        )
        or any(not isinstance(value, dict) for value in checks.values())
    ):
        reasons.append("paired strict/loose prompt checks differ")
    power = report.get("promotion_power") or {}
    if (
        power.get("noninferiority_margin") != 0.02
        or power.get("minimum_zero_regression_samples") != 149
        or power.get("sufficient") is not False
    ):
        reasons.append("paired promotion power declaration differs")
    if report.get("authorization") != {
        "five_point_screen_authorized": True,
        "two_point_promotion_authorized": False,
        "overall_promotion_authorized": False,
    }:
        reasons.append("paired authorization boundary differs")
    expected_privacy = {
        "contains_sample_outcomes": False,
        "contains_prompts": False,
        "contains_model_outputs": False,
        "contains_token_ids": False,
        "contains_credentials": False,
    }
    if report.get("privacy") != expected_privacy:
        reasons.append("paired non-inferiority privacy contract differs")
    return reasons


def compare(
    *,
    control_status: Any,
    candidate_status: Any,
    control_contract: Any,
    candidate_contract: Any,
    control_report: Any,
    candidate_report: Any,
    control_progress: Any,
    candidate_progress: Any,
    control_4xx: Any,
    candidate_4xx: Any,
    control_identity: Any,
    candidate_identity: Any,
    score_comparison: Any,
    exact_comparison: Any,
    paired_noninferiority: Any,
    file_sha256s: dict[str, dict[str, str]],
) -> Json:
    reasons = _status_reasons(
        control_status,
        label="control",
        expected_label="m1-122-control-fused-off",
        fused_prefill="0",
        file_sha256s=file_sha256s["control"],
    )
    reasons.extend(_status_reasons(
        candidate_status,
        label="candidate",
        expected_label="m1-122-candidate-fused-on",
        fused_prefill="1",
        file_sha256s=file_sha256s["candidate"],
    ))
    if isinstance(control_status, dict):
        reasons.extend(service._runtime_reasons(
            control_contract,
            control_status,
            label="control",
            expected_policy="admission64",
            expected_restore_mode="hybrid64",
            expected_fused_prefill="0",
        ))
    if isinstance(candidate_status, dict):
        reasons.extend(service._runtime_reasons(
            candidate_contract,
            candidate_status,
            label="candidate",
            expected_policy="admission64",
            expected_restore_mode="hybrid64",
            expected_fused_prefill="1",
        ))

    score_reasons = ifeval.comparison_reasons(
        control_report,
        candidate_report,
        {"fused_prefill"},
        False,
    )
    exact_reasons = ifeval.comparison_reasons(
        control_report,
        candidate_report,
        {"fused_prefill"},
        True,
    )
    exact_output_reasons = [
        reason
        for reason in exact_reasons
        if reason.startswith("candidate output differs for key ")
    ]
    reasons.extend(_comparison_reasons(
        score_comparison,
        baseline_sha256=file_sha256s["control"]["ifeval_report"],
        candidate_sha256=file_sha256s["candidate"]["ifeval_report"],
        require_exact_output=False,
        expected_reasons=score_reasons,
        label="score",
    ))
    reasons.extend(_comparison_reasons(
        exact_comparison,
        baseline_sha256=file_sha256s["control"]["ifeval_report"],
        candidate_sha256=file_sha256s["candidate"]["ifeval_report"],
        require_exact_output=True,
        expected_reasons=exact_reasons,
        label="exact-output diagnostic",
    ))
    reasons.extend(_paired_noninferiority_reasons(
        paired_noninferiority,
        baseline_sha256=file_sha256s["control"]["ifeval_report"],
        candidate_sha256=file_sha256s["candidate"]["ifeval_report"],
    ))
    if isinstance(control_report, dict):
        reasons.extend(_progress_reasons(
            control_progress,
            control_report,
            file_sha256s["control"]["ifeval_report"],
            "control",
        ))
    if isinstance(candidate_report, dict):
        reasons.extend(_progress_reasons(
            candidate_progress,
            candidate_report,
            file_sha256s["candidate"]["ifeval_report"],
            "candidate",
        ))
    reasons.extend(_zero_4xx_reasons(control_4xx, "control"))
    reasons.extend(_zero_4xx_reasons(candidate_4xx, "candidate"))
    reasons.extend(service._process_identity_reasons(
        control_identity, "control"))
    reasons.extend(service._process_identity_reasons(
        candidate_identity, "candidate"))

    if (
        isinstance(control_status, dict)
        and isinstance(candidate_status, dict)
    ):
        for field in ("source_revision", "source_branch", "instance"):
            if control_status.get(field) != candidate_status.get(field):
                reasons.append(f"A/B status differs in {field}")
    if control_4xx != candidate_4xx:
        reasons.append("A/B zero-4xx reports differ")
    if (
        isinstance(control_identity, dict)
        and isinstance(candidate_identity, dict)
        and control_identity.get("session_token")
        == candidate_identity.get("session_token")
    ):
        reasons.append("A/B process session tokens must differ")

    qualified = not reasons
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "qualified": qualified,
        "ifeval_five_point_screen_authorized": qualified,
        "ifeval_two_point_promotion_authorized": False,
        "strict_zero_stratum_qualified": not score_reasons,
        "strict_zero_stratum_reason_count": len(score_reasons),
        "strict_exact_output_qualified": not exact_output_reasons,
        "strict_exact_output_mismatch_count": len(exact_output_reasons),
        "score_delta": (
            score_comparison.get("score_delta")
            if isinstance(score_comparison, dict) else None
        ),
        "performance_authorized": False,
        "default_change_authorized": False,
        "production_promotion_authorized": False,
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
    parser.add_argument("--score-comparison", type=Path, required=True)
    parser.add_argument("--exact-comparison", type=Path, required=True)
    parser.add_argument("--paired-noninferiority", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "control": _arm_paths(args.control_root),
        "candidate": _arm_paths(args.candidate_root),
    }
    values = {
        arm: {name: service._load(path) for name, path in arm_paths.items()}
        for arm, arm_paths in paths.items()
    }
    file_sha256s = {
        arm: {
            name: service._file_sha256(path)
            for name, path in arm_paths.items()
            if name != "status"
        }
        for arm, arm_paths in paths.items()
    }
    result = compare(
        control_status=values["control"]["status"],
        candidate_status=values["candidate"]["status"],
        control_contract=values["control"]["runtime_contract"],
        candidate_contract=values["candidate"]["runtime_contract"],
        control_report=values["control"]["ifeval_report"],
        candidate_report=values["candidate"]["ifeval_report"],
        control_progress=values["control"]["ifeval_progress"],
        candidate_progress=values["candidate"]["ifeval_progress"],
        control_4xx=values["control"]["api_4xx_attribution"],
        candidate_4xx=values["candidate"]["api_4xx_attribution"],
        control_identity=values["control"]["process_group_identity"],
        candidate_identity=values["candidate"]["process_group_identity"],
        score_comparison=service._load(args.score_comparison),
        exact_comparison=service._load(args.exact_comparison),
        paired_noninferiority=service._load(args.paired_noninferiority),
        file_sha256s=file_sha256s,
    )
    result["inputs"] = {
        f"{arm}_{name}_sha256": digest
        for arm, arm_hashes in file_sha256s.items()
        for name, digest in arm_hashes.items()
    }
    result["inputs"].update({
        "control_status_sha256": service._file_sha256(
            paths["control"]["status"]),
        "candidate_status_sha256": service._file_sha256(
            paths["candidate"]["status"]),
        "score_comparison_sha256": service._file_sha256(
            args.score_comparison),
        "exact_comparison_sha256": service._file_sha256(
            args.exact_comparison),
        "paired_noninferiority_sha256": service._file_sha256(
            args.paired_noninferiority),
    })
    service._atomic_write(args.out, result)
    print(json.dumps({
        "qualified": result["qualified"],
        "strict_exact_output_qualified": result[
            "strict_exact_output_qualified"],
        "strict_exact_output_mismatch_count": result[
            "strict_exact_output_mismatch_count"],
        "reasons": result["reasons"],
        "out": str(args.out),
    }, sort_keys=True))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
