#!/usr/bin/env python3
"""Version-dispatched BI100 validation-contract helpers.

The v1 branch intentionally validates only its frozen identity and historical
shape.  New classifications are available only with the September 2026 v2
contract; an unknown or cross-bound schema/version pair fails closed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


STATES = ("pass", "fail", "inconclusive", "invalid")
LAYERED_SCHEMAS = {
    1: "bi100-layered-quality-gate-contract-v1",
    2: "bi100-layered-quality-gate-contract-v2",
}
FUNNEL_SCHEMAS = {
    1: "bi100-experiment-funnel-v1",
    2: "bi100-experiment-funnel-v2",
}


class ContractError(ValueError):
    """Raised when a contract or report identity is unsupported."""


def _finite(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be an object")
    return value


def _number(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if not _finite(value) or float(value) < minimum:
        raise ContractError(f"{name} must be a finite number >= {minimum}")
    return float(value)


def contract_identity(contract: Any, family: str) -> tuple[str, int]:
    value = _mapping(contract, "contract")
    schemas = {"layered": LAYERED_SCHEMAS, "funnel": FUNNEL_SCHEMAS}.get(
        family)
    if schemas is None:
        raise ContractError("unknown contract family")
    version = value.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ContractError("contract version must be an integer")
    expected = schemas.get(version)
    if expected is None or value.get("schema") != expected:
        raise ContractError("unknown or mismatched contract schema/version")
    return expected, version


def validate_contract(contract: Any, family: str) -> dict[str, Any]:
    schema, version = contract_identity(contract, family)
    value = _mapping(contract, "contract")
    if family == "layered":
        required = (
            ("operator_numerics", "teacher_forced_distribution",
             "paired_task_capability", "performance_and_lifecycle")
            if version == 2 else
            ("operator_numerics", "teacher_forced_topk",
             "paired_capability")
        )
    else:
        required = ("scope", "stages", "transition_rules")
        if version == 2:
            required += ("global_validity", "metrics_contract")
    missing = [name for name in required if name not in value]
    if missing:
        raise ContractError(
            f"{family} v{version} contract is missing: {', '.join(missing)}")
    if family == "layered" and version == 2:
        if value.get("result_states") != list(STATES):
            raise ContractError("v2 result states differ")
        validity = _mapping(value.get("experiment_validity"),
                            "experiment_validity")
        provenance = _mapping(validity.get("provenance_policy"),
                              "provenance_policy")
        required_names = (validity.get("required_identity", [])
                          + validity.get("required_population", []))
        if (any("sha256" in str(name).lower() for name in required_names)
                or "runtime_overlay_file_tree" not in
                provenance.get("sha256_not_required_for", [])
                or "temporary_activation" not in
                provenance.get("sha256_not_required_for", [])
                or "prefix_cache_content_identity" not in
                provenance.get("sha256_required_only_for", [])):
            raise ContractError("v2 lightweight provenance policy differs")
        numeric = _mapping(value["operator_numerics"], "operator_numerics")
        if (numeric.get("relative_l2_error_ratio_limit") != 2.0
                or numeric.get("maximum_absolute_error_ratio_limit") != 2.0
                or numeric.get("ratio_denominator_floor") != 1e-12
                or numeric.get(
                    "candidate_vs_rounded_is_universal_hard_gate") is not False):
            raise ContractError("v2 calibrated numeric contract differs")
        distribution = _mapping(
            value["teacher_forced_distribution"],
            "teacher_forced_distribution")
        if (distribution.get("minimum_top1_agreement") is not None
                or distribution.get("control_control_calibration_required")
                is not True):
            raise ContractError("v2 distribution calibration differs")
    return {"family": family, "schema": schema, "version": version}


def validate_report_binding(
    report: Any,
    contract: Any,
    family: str,
) -> dict[str, Any]:
    """Require a report to bind the exact contract without relabelling it."""
    identity = validate_contract(contract, family)
    value = _mapping(report, "report")
    report_version = value.get("version")
    expected_report_schema = f"bi100-validation-layer-report-v{report_version}"
    bound = value.get("contract")
    if (report_version != identity["version"]
            or value.get("schema") != expected_report_schema
            or not isinstance(bound, dict)
            or bound.get("schema") != identity["schema"]
            or bound.get("version") != identity["version"]):
        raise ContractError("report and contract schema/version do not match")
    status = value.get("status")
    allowed = set(STATES) if report_version == 2 else {"pass", "fail", "invalid"}
    if status not in allowed:
        raise ContractError("report status is unsupported for its version")
    return {**identity, "status": status}


def classify_validity(
    evidence: Any,
    contract: dict[str, Any],
) -> dict[str, Any]:
    validate_contract(contract, "layered")
    if contract["version"] != 2:
        raise ContractError("new validity classification requires v2")
    value = _mapping(evidence, "validity evidence")
    rules = contract["experiment_validity"]
    required = (rules["required_identity"] + rules["required_population"]
                + rules["required_lifecycle"] + ["timing_samples"])
    missing = [name for name in required if name not in value]
    if missing:
        return {"status": "invalid", "reasons": [
            f"missing validity evidence: {', '.join(missing)}"]}
    counts = [value.get(name) for name in (
        "expected_request_count", "attempted_request_count",
        "completed_request_count", "failed_request_count")]
    if (any(not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in counts)
            or counts[0] != counts[1]
            or counts[1] != counts[2] + counts[3]
            or not isinstance(value.get("timing_samples"), list)
            or not value["timing_samples"]
            or any(value.get(name) is not True for name in
                   rules["required_lifecycle"])):
        return {"status": "invalid", "reasons": [
            "request population, timing samples, or lifecycle is invalid"]}
    return {"status": "pass", "reasons": []}


def error_ratio(candidate_error: Any, baseline_error: Any, floor: Any) -> float:
    numerator = _number(candidate_error, "candidate error")
    denominator = max(
        _number(baseline_error, "baseline error"),
        _number(floor, "ratio floor", minimum=1e-300),
    )
    return numerator / denominator


def classify_fp16_numerics(
    evidence: Any,
    contract: dict[str, Any],
    *,
    operator: str = "attention",
) -> dict[str, Any]:
    validate_contract(contract, "layered")
    if contract["version"] != 2:
        raise ContractError("new numeric classification requires v2")
    value = _mapping(evidence, "numeric evidence")
    required = {
        "reference_finite", "baseline_finite", "rounded_reference_finite",
        "candidate_finite", "candidate_repeat_finite", "metadata_exact",
        "candidate_relative_l2", "baseline_relative_l2",
        "candidate_max_abs", "baseline_max_abs",
        "candidate_lse_relative_l2", "baseline_lse_relative_l2",
    }
    if not required.issubset(value):
        return {"status": "invalid", "reasons": ["numeric evidence is incomplete"]}
    if any(type(value[name]) is not bool for name in (
            "reference_finite", "baseline_finite", "rounded_reference_finite",
            "candidate_finite", "candidate_repeat_finite", "metadata_exact")):
        return {"status": "invalid", "reasons": ["numeric flags are malformed"]}
    if not value["reference_finite"] or not value["baseline_finite"] \
            or not value["rounded_reference_finite"]:
        return {"status": "invalid", "reasons": ["reference or baseline is nonfinite"]}
    if not value["candidate_finite"] or not value["candidate_repeat_finite"]:
        return {"status": "fail", "reasons": ["candidate is nonfinite"]}
    if not value["metadata_exact"]:
        return {"status": "fail", "reasons": ["discrete metadata differs"]}
    rules = contract["operator_numerics"]
    try:
        rel_ratio = error_ratio(
            value["candidate_relative_l2"], value["baseline_relative_l2"],
            rules["ratio_denominator_floor"])
        max_ratio = error_ratio(
            value["candidate_max_abs"], value["baseline_max_abs"],
            rules["ratio_denominator_floor"])
        candidate_lse = _number(
            value["candidate_lse_relative_l2"], "candidate LSE error")
        baseline_lse = _number(
            value["baseline_lse_relative_l2"], "baseline LSE error")
    except ContractError as exc:
        return {"status": "invalid", "reasons": [str(exc)]}
    lse_limit = max(rules["attention_lse"]["absolute_floor"],
                    rules["attention_lse"]["baseline_error_multiplier"]
                    * baseline_lse)
    failures = []
    if rel_ratio > rules["relative_l2_error_ratio_limit"]:
        failures.append("relative-L2 error ratio exceeds limit")
    if max_ratio > rules["maximum_absolute_error_ratio_limit"]:
        failures.append("maximum-absolute error ratio exceeds limit")
    if operator == "attention" and candidate_lse > lse_limit:
        failures.append("attention LSE error exceeds calibrated limit")
    if operator == "gdn" and not value.get("gdn_checkpoints_exact", False):
        failures.append("GDN checkpoint or final state check failed")
    return {
        "status": "fail" if failures else "pass",
        "relative_l2_error_ratio": rel_ratio,
        "maximum_absolute_error_ratio": max_ratio,
        "attention_lse_limit": lse_limit,
        "reasons": failures,
    }


def classify_distribution(
    candidate: Any,
    control_control: Any,
    contract: dict[str, Any],
) -> dict[str, Any]:
    validate_contract(contract, "layered")
    if contract["version"] != 2:
        raise ContractError("new distribution classification requires v2")
    cand = _mapping(candidate, "candidate distribution")
    aa = _mapping(control_control, "control/control distribution")
    required_candidate = {
        "top1_agreement", "mutual_topk_coverage",
        "teacher_token_logprob_delta", "shared_token_logprob_delta",
        "paired_nll_difference", "paired_nll_one_sided_95_upper_ci",
        "first_divergent_token", "baseline_top1_margin",
        "high_margin_flips",
    }
    required_aa = {"shared_logprob_delta_p99", "paired_nll_upper_ci"}
    if not required_candidate.issubset(cand) or not required_aa.issubset(aa):
        return {"status": "invalid", "classification": "invalid",
                "reasons": ["A/A or candidate distribution evidence is incomplete"]}
    numeric_names = required_candidate - {"first_divergent_token"}
    if (any(not _finite(cand[name]) for name in numeric_names)
            or any(not _finite(aa[name]) for name in required_aa)
            or not isinstance(cand["high_margin_flips"], int)
            or isinstance(cand["high_margin_flips"], bool)):
        return {"status": "invalid", "classification": "invalid",
                "reasons": ["distribution metrics are malformed"]}
    rules = contract["teacher_forced_distribution"]
    margin_threshold = max(
        rules["high_margin_threshold"]["absolute_floor_nats"],
        rules["high_margin_threshold"][
            "aa_p99_shared_logprob_delta_multiplier"]
        * float(aa["shared_logprob_delta_p99"]),
    )
    nll_limit = max(
        rules["nll_regression_upper_ci"]["absolute_floor_nats"],
        rules["nll_regression_upper_ci"]["aa_upper_ci_multiplier"]
        * float(aa["paired_nll_upper_ci"]),
    )
    reasons = []
    if cand["high_margin_flips"] > rules["maximum_unexplained_high_margin_flips"]:
        reasons.append("unexplained high-margin flips observed")
    if cand["paired_nll_one_sided_95_upper_ci"] > nll_limit:
        reasons.append("paired NLL regression upper CI exceeds calibrated limit")
    drift = bool(reasons)
    return {
        "status": "inconclusive" if drift else "pass",
        "classification": (rules["threshold_exceedance_classification"]
                           if drift else "distribution_within_aa_envelope"),
        "high_margin_threshold_nats": margin_threshold,
        "nll_regression_upper_ci_limit": nll_limit,
        "top1_agreement": cand["top1_agreement"],
        "reasons": reasons,
    }


def classify_capability(
    evidence: Any,
    contract: dict[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    validate_contract(contract, "layered")
    if contract["version"] != 2:
        raise ContractError("new capability classification requires v2")
    value = _mapping(evidence, "capability evidence")
    if phase not in {"development", "promotion"}:
        return {"status": "invalid", "reasons": ["unknown capability phase"]}
    required = {
        "deterministic_baseline_only_failures", "paired_lower_ci",
        "paired_bootstrap_reported", "exact_mcnemar_reported",
        "underpowered", "strata",
    }
    if not required.issubset(value):
        return {"status": "invalid", "reasons": [
            "capability evidence is incomplete"]}
    rules = contract["paired_task_capability"]
    if (not isinstance(value["deterministic_baseline_only_failures"], int)
            or isinstance(value["deterministic_baseline_only_failures"], bool)
            or not _finite(value["paired_lower_ci"])
            or any(type(value[name]) is not bool for name in (
                "paired_bootstrap_reported", "exact_mcnemar_reported",
                "underpowered"))
            or not isinstance(value["strata"], dict)
            or set(value["strata"]) != set(rules["required_strata"])):
        return {"status": "invalid", "reasons": [
            "capability evidence is malformed or strata are incomplete"]}
    if (not value["paired_bootstrap_reported"]
            or not value["exact_mcnemar_reported"]):
        return {"status": "invalid", "reasons": [
            "paired bootstrap and exact McNemar diagnostics are required"]}
    if value["deterministic_baseline_only_failures"] > 0:
        return {"status": "fail", "reasons": [
            "new deterministic baseline-only failure observed"]}
    if value["underpowered"]:
        return {"status": "inconclusive", "reasons": [
            "capability sample is underpowered"]}
    margin = rules[f"{phase}_noninferiority_margin"]
    if float(value["paired_lower_ci"]) <= -margin:
        return {"status": "fail", "reasons": [
            "paired lower CI does not clear the non-inferiority margin"]}
    return {"status": "pass", "reasons": []}


def amdahl_projected_gain(hotspot_fraction: Any, speedup: Any) -> float:
    fraction = _number(hotspot_fraction, "hotspot fraction")
    factor = _number(speedup, "speedup", minimum=1e-300)
    if fraction > 1.0:
        raise ContractError("hotspot fraction must not exceed one")
    return 1.0 / (1.0 - fraction + fraction / factor) - 1.0


def classify_performance(
    point_estimate: Any,
    one_sided_lower_ci: Any,
    contract: dict[str, Any],
    *,
    common_bucket_regression: float = 0.0,
    common_bucket_regression_supported: bool = False,
) -> dict[str, Any]:
    validate_contract(contract, "layered")
    if contract["version"] != 2:
        raise ContractError("new performance classification requires v2")
    point = _number(point_estimate, "paired gain")
    lower = float(one_sided_lower_ci) if _finite(one_sided_lower_ci) else None
    if lower is None:
        return {"status": "invalid", "reasons": ["lower CI is invalid"]}
    rules = contract["performance_and_lifecycle"]
    short = rules["short_tp4"]
    if (common_bucket_regression_supported
            and common_bucket_regression >
            rules["maximum_supported_common_bucket_regression"]):
        return {"status": "fail", "reasons": [
            "statistically supported common-bucket regression exceeds limit"]}
    if point < short["stop_below_point_estimate"]:
        return {"status": "fail", "reasons": ["point estimate is below 2%"]}
    if point < short["normal_minimum_paired_gain"]:
        return {"status": "inconclusive", "extra_pairs_allowed":
                short["maximum_extra_paired_runs"], "reasons": [
                    "point estimate is in the 2%-3% gray zone"]}
    if lower <= short["one_sided_95_lower_ci_must_exceed"]:
        return {"status": "inconclusive", "extra_pairs_allowed":
                short["maximum_extra_paired_runs"], "reasons": [
                    "one-sided 95% lower CI does not exceed zero"]}
    return {"status": "pass", "reasons": []}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--family", choices=("layered", "funnel"), required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
        result = validate_contract(contract, args.family)
        if args.report:
            report = json.loads(args.report.read_text(encoding="utf-8"))
            result["report"] = validate_report_binding(
                report, contract, args.family)
    except (ContractError, json.JSONDecodeError, OSError) as exc:
        print(json.dumps({"status": "invalid", "reason": str(exc)},
                         sort_keys=True))
        return 2
    print(json.dumps({"status": "pass", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
