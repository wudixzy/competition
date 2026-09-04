#!/usr/bin/env python3
"""Qualify the M1-137 fused-prefill IFEval power149 TP4 A/B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import compare_admission64_quality_service_ab as service
import compare_ifeval_reports as ifeval
import compare_ifeval_paired_noninferiority as paired_ifeval
import compare_m1_122_ifeval_service_ab as legacy


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "bi100-m1-137-ifeval-power149-fused-prefill-ab-v1"
VERSION = 1
PAIR_COUNT = 149
MANIFEST_SHA256 = ifeval.EXPECTED_POWER_MANIFEST_SHA256
MANIFEST_PATH = (
    ROOT / "quality/external/google_ifeval/manifest.power149.v2.json")
LAYERED_CONTRACT = ROOT / "quality/layered_quality_gate.v2.json"
# M1-137 is a historical July-v2 evidence adapter. Its report identity remains
# pinned even though the live v2 contract was superseded on 2026-09-04.
EXPECTED_LAYERED_CONTRACT_SHA256 = (
    "5a7a9dc6fb430118abd821a96506f01685cfd1eec421c24a262dc4b6c9cdd5dd"
)
LAYERED_CONTRACT_SHA256 = EXPECTED_LAYERED_CONTRACT_SHA256
CURRENT_LAYERED_CONTRACT_SHA256 = service._file_sha256(LAYERED_CONTRACT)
Json = dict[str, Any]

if service._file_sha256(MANIFEST_PATH) != MANIFEST_SHA256:
    raise RuntimeError("M1-137 power149 manifest identity differs")
_MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
_OFFLINE_ENVIRONMENT = _MANIFEST["offline_environment"]
EXPECTED_DISTRIBUTION_SHA256 = {
    Path(item["path"]).name: item["sha256"]
    for item in _OFFLINE_ENVIRONMENT["distribution_artifacts"]
}
EXPECTED_PUNKT_TAB_SHA256 = (
    _OFFLINE_ENVIRONMENT["nltk_punkt_tab"]["archive_sha256"])


def _arm_paths(root: Path) -> dict[str, Path]:
    return {
        **legacy._arm_paths(root),
        "ifeval_install": root / "ifeval_install.json",
    }


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
        or progress.get("selected") != PAIR_COUNT
        or progress.get("attempted") != PAIR_COUNT
        or progress.get("successful") != PAIR_COUNT
        or progress.get("errors") != 0
        or progress.get("last_ordinal") != PAIR_COUNT
        or progress.get("complete") is not True
        or progress.get("report_sha256") != report_sha256
        or progress.get("failures") != []
    ):
        reasons.append(f"{label}: IFEval power149 progress is incomplete")
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


def _install_reasons(install: Any, label: str) -> list[str]:
    if not isinstance(install, dict):
        return [f"{label}: IFEval install root must be an object"]
    if (
        install.get("schema") != "bi100-ifeval-offline-environment-v1"
        or install.get("version") != 1
        or install.get("qualified") is not True
        or install.get("manifest_sha256") != MANIFEST_SHA256
        or not str(install.get("python", "")).startswith("3.10.")
        or install.get("system_site_packages_modified") is not False
        or install.get("distribution_sha256")
        != EXPECTED_DISTRIBUTION_SHA256
        or install.get("punkt_tab_archive_sha256")
        != EXPECTED_PUNKT_TAB_SHA256
    ):
        return [f"{label}: IFEval power149 environment identity differs"]
    return []


def _paired_reasons(
    report: Any,
    *,
    baseline_sha256: str,
    candidate_sha256: str,
) -> list[str]:
    if not isinstance(report, dict):
        return ["paired power149 report root must be an object"]
    reasons = []
    if (
        report.get("schema") != paired_ifeval.SCHEMA_V2
        or report.get("version") != 2
        or report.get("contract_version") != 2
        or report.get("status") != "pass"
        or report.get("qualified") is not True
        or report.get("baseline_sha256") != baseline_sha256
        or report.get("candidate_sha256") != candidate_sha256
        or report.get("contract_sha256") != LAYERED_CONTRACT_SHA256
        or report.get("allowed_switches") != ["fused_prefill"]
        or report.get("sample_count") != PAIR_COUNT
        or report.get("reasons") != []
    ):
        reasons.append("paired power149 noninferiority did not qualify")
    screen = report.get("screen") or {}
    checks = screen.get("checks") or {}
    if (
        screen.get("name") != "default-two-point"
        or screen.get("confidence") != 0.95
        or screen.get("noninferiority_margin") != 0.02
        or screen.get("bootstrap_samples") != 20000
        or screen.get("bootstrap_seed") != 20260729
        or set(checks) != {"strict_prompt", "loose_prompt"}
    ):
        reasons.append("paired power149 screen contract differs")
    for name, expected_seed in (
        ("strict_prompt", 20260729),
        ("loose_prompt", 20260730),
    ):
        value = checks.get(name)
        statistics = value.get("statistics") if isinstance(value, dict) else {}
        if (
            not isinstance(value, dict)
            or value.get("status") != "pass"
            or value.get("qualified") is not True
            or value.get("sample_count") != PAIR_COUNT
            or statistics.get("confidence") != 0.95
            or statistics.get("noninferiority_margin") != 0.02
            or statistics.get("bootstrap_samples") != 20000
            or statistics.get("bootstrap_seed") != expected_seed
            or statistics.get("minimum_zero_regression_samples")
            != PAIR_COUNT
        ):
            reasons.append(f"paired {name} power149 result differs")
    power = report.get("promotion_power") or {}
    if (
        power.get("noninferiority_margin") != 0.02
        or power.get("minimum_zero_regression_samples") != PAIR_COUNT
        or power.get("sufficient") is not True
    ):
        reasons.append("power149 sample sufficiency differs")
    if report.get("authorization") != {
        "five_point_screen_authorized": False,
        "two_point_capability_surface_authorized": True,
        "two_point_promotion_authorized": False,
        "overall_promotion_authorized": False,
    }:
        reasons.append("power149 authorization boundary differs")
    expected_privacy = {
        "contains_sample_outcomes": False,
        "contains_prompts": False,
        "contains_model_outputs": False,
        "contains_token_ids": False,
        "contains_credentials": False,
    }
    if report.get("privacy") != expected_privacy:
        reasons.append("paired power149 privacy contract differs")
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
    control_install: Any,
    candidate_install: Any,
    control_4xx: Any,
    candidate_4xx: Any,
    control_identity: Any,
    candidate_identity: Any,
    score_comparison: Any,
    exact_comparison: Any,
    paired_noninferiority: Any,
    file_sha256s: dict[str, dict[str, str]],
) -> Json:
    reasons = legacy._status_reasons(
        control_status,
        label="control",
        expected_label="m1-137-control-fused-off",
        fused_prefill="0",
        file_sha256s=file_sha256s["control"],
    )
    reasons.extend(legacy._status_reasons(
        candidate_status,
        label="candidate",
        expected_label="m1-137-candidate-fused-on",
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

    pair_reasons = ifeval.pair_identity_reasons(
        control_report, candidate_report, {"fused_prefill"})
    reasons.extend(pair_reasons)
    score_reasons = ifeval.comparison_reasons(
        control_report, candidate_report, {"fused_prefill"}, False)
    exact_reasons = ifeval.comparison_reasons(
        control_report, candidate_report, {"fused_prefill"}, True)
    exact_output_reasons = [
        reason for reason in exact_reasons
        if reason.startswith("candidate output differs for key ")
    ]
    reasons.extend(legacy._comparison_reasons(
        score_comparison,
        baseline_sha256=file_sha256s["control"]["ifeval_report"],
        candidate_sha256=file_sha256s["candidate"]["ifeval_report"],
        require_exact_output=False,
        expected_reasons=score_reasons,
        label="score diagnostic",
    ))
    reasons.extend(legacy._comparison_reasons(
        exact_comparison,
        baseline_sha256=file_sha256s["control"]["ifeval_report"],
        candidate_sha256=file_sha256s["candidate"]["ifeval_report"],
        require_exact_output=True,
        expected_reasons=exact_reasons,
        label="exact-output diagnostic",
    ))
    reasons.extend(_paired_reasons(
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
    reasons.extend(_install_reasons(control_install, "control"))
    reasons.extend(_install_reasons(candidate_install, "candidate"))
    reasons.extend(legacy._zero_4xx_reasons(control_4xx, "control"))
    reasons.extend(legacy._zero_4xx_reasons(candidate_4xx, "candidate"))
    reasons.extend(service._process_identity_reasons(
        control_identity, "control"))
    reasons.extend(service._process_identity_reasons(
        candidate_identity, "candidate"))

    if isinstance(control_status, dict) and isinstance(candidate_status, dict):
        for field in ("source_revision", "source_branch", "instance"):
            if control_status.get(field) != candidate_status.get(field):
                reasons.append(f"A/B status differs in {field}")
    if control_install != candidate_install:
        reasons.append("A/B IFEval environment attestations differ")
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
        "ifeval_two_point_capability_surface_statistically_qualified": (
            qualified),
        "ifeval_two_point_capability_surface_authorized": False,
        "outer_lifecycle_pending": True,
        "strict_zero_stratum_qualified": not score_reasons,
        "strict_zero_stratum_reason_count": len(score_reasons),
        "cross_arm_exact_output_qualified": not exact_output_reasons,
        "cross_arm_exact_output_mismatch_count": len(exact_output_reasons),
        "cross_arm_exact_output_role": "diagnostic",
        "performance_authorized": False,
        "default_change_authorized": False,
        "yaml_change_authorized": False,
        "main_merge_authorized": False,
        "production_promotion_authorized": False,
        "reasons": reasons,
        "privacy": {
            "contains_raw_requests": False,
            "contains_raw_model_outputs": False,
            "contains_sample_outcomes": False,
            "contains_credentials": False,
        },
    }


def compare_from_paths(
    *,
    control_root: Path,
    candidate_root: Path,
    score_comparison: Path,
    exact_comparison: Path,
    paired_noninferiority: Path,
) -> Json:
    paths = {
        "control": _arm_paths(control_root),
        "candidate": _arm_paths(candidate_root),
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
    return compare(
        control_status=values["control"]["status"],
        candidate_status=values["candidate"]["status"],
        control_contract=values["control"]["runtime_contract"],
        candidate_contract=values["candidate"]["runtime_contract"],
        control_report=values["control"]["ifeval_report"],
        candidate_report=values["candidate"]["ifeval_report"],
        control_progress=values["control"]["ifeval_progress"],
        candidate_progress=values["candidate"]["ifeval_progress"],
        control_install=values["control"]["ifeval_install"],
        candidate_install=values["candidate"]["ifeval_install"],
        control_4xx=values["control"]["api_4xx_attribution"],
        candidate_4xx=values["candidate"]["api_4xx_attribution"],
        control_identity=values["control"]["process_group_identity"],
        candidate_identity=values["candidate"]["process_group_identity"],
        score_comparison=service._load(score_comparison),
        exact_comparison=service._load(exact_comparison),
        paired_noninferiority=service._load(paired_noninferiority),
        file_sha256s=file_sha256s,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--score-comparison", type=Path, required=True)
    parser.add_argument("--exact-comparison", type=Path, required=True)
    parser.add_argument("--paired-noninferiority", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = compare_from_paths(
        control_root=args.control_root,
        candidate_root=args.candidate_root,
        score_comparison=args.score_comparison,
        exact_comparison=args.exact_comparison,
        paired_noninferiority=args.paired_noninferiority,
    )
    service._atomic_write(args.out, result)
    print(json.dumps({
        "out": str(args.out),
        "qualified": result["qualified"],
        "reasons": result["reasons"],
    }, sort_keys=True))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
