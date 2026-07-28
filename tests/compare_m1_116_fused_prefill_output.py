#!/usr/bin/env python3
"""Compare M1-116 fused-prefill divergence diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import diagnose_m1_116_fused_prefill_output as diagnostic
import quality_runtime_contract as runtime_contract


SCHEMA = "bi100-m1-116-fused-prefill-output-comparison-v1"
VERSION = 1
Json = dict[str, Any]
REPORT_FIELDS = {
    "schema",
    "version",
    "mode",
    "qualified_diagnostic",
    "strict_quality_non_regression_authorized",
    "production_promotion_authorized",
    "reasons",
    "source_revision",
    "runtime_identity",
    "instance",
    "model_path",
    "target_prompt_tokens",
    "reproduction_max_tokens",
    "max_tokens_ladder",
    "seed",
    "run_id_sha256",
    "runtime_contract",
    "reproduction",
    "ladder",
    "privacy",
}


def _load(path: Path) -> Json:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} root must be an object")
    return value


def _validate_report(report: Any, mode: str) -> list[str]:
    reasons = []
    if not isinstance(report, dict):
        return [f"{mode} report root is invalid"]
    if set(report) != REPORT_FIELDS:
        reasons.append(f"{mode} report fields are invalid")
    if (report.get("schema") != diagnostic.SCHEMA
            or report.get("version") != diagnostic.VERSION):
        reasons.append(f"{mode} report schema differs")
    if report.get("mode") != mode:
        reasons.append(f"{mode} report mode differs")
    if (report.get("qualified_diagnostic") is not True
            or report.get("reasons") != []):
        reasons.append(f"{mode} diagnostic did not qualify")
    if (report.get("strict_quality_non_regression_authorized") is not False
            or report.get("production_promotion_authorized") is not False):
        reasons.append(f"{mode} authorization state is invalid")
    if report.get("target_prompt_tokens") != diagnostic.TARGET_PROMPT_TOKENS:
        reasons.append(f"{mode} target prompt differs")
    if (report.get("reproduction_max_tokens")
            != diagnostic.REPRODUCTION_MAX_TOKENS):
        reasons.append(f"{mode} reproduction budget differs")
    if report.get("max_tokens_ladder") != list(
            diagnostic.MAX_TOKENS_LADDER):
        reasons.append(f"{mode} ladder differs")
    if report.get("seed") != diagnostic.SEED:
        reasons.append(f"{mode} seed differs")
    for field in ("source_revision", "runtime_identity", "instance",
                  "model_path"):
        if not isinstance(report.get(field), str) or not report[field]:
            reasons.append(f"{mode} {field} is invalid")
    if not runtime_contract.is_git_revision(report.get("source_revision")):
        reasons.append(f"{mode} source revision is invalid")
    if not runtime_contract.is_sha256(report.get("run_id_sha256")):
        reasons.append(f"{mode} run identity is invalid")
    privacy = report.get("privacy")
    if privacy != {
            "contains_raw_requests": False,
            "contains_raw_model_outputs": False,
            "contains_token_ids": False,
            "contains_credentials": False,
    }:
        reasons.append(f"{mode} privacy contract is invalid")

    wrapper = report.get("runtime_contract")
    if (not isinstance(wrapper, dict)
            or set(wrapper) != {"sha256", "contract"}):
        reasons.append(f"{mode} runtime contract wrapper is invalid")
    else:
        contract = wrapper.get("contract")
        expected = {
            "source_revision": report.get("source_revision"),
            "runtime_identity": report.get("runtime_identity"),
            "instance": report.get("instance"),
            "gpu_count": 4,
            "tensor_parallel_size": 4,
            "max_model_len": 262144,
            "model_path": report.get("model_path"),
            "tokenizer_path": report.get("model_path"),
            "served_model_name": "llm",
        }
        try:
            digest = runtime_contract.validate_runtime_contract(
                contract, expected, require_cache_trace=True)
        except (runtime_contract.RuntimeContractError, TypeError) as error:
            reasons.append(f"{mode} runtime contract is invalid: {error}")
        else:
            if wrapper.get("sha256") != digest:
                reasons.append(f"{mode} runtime contract digest differs")
            selector = "0" if mode == "control" else "1"
            if (contract["environment"].get(
                    "BI100_ATTN_COREX_FUSED_PREFILL") != selector):
                reasons.append(f"{mode} fused-prefill selector differs")

    reproduction = report.get("reproduction")
    if (not isinstance(reproduction, dict)
            or set(reproduction) != {"cold", "warm", "cold_warm_exact"}):
        reasons.append(f"{mode} reproduction fields are invalid")
    else:
        observation_reasons = diagnostic._validate_observations(
            reproduction["cold"],
            reproduction["warm"],
            report.get("ladder") if isinstance(
                report.get("ladder"), list) else [],
        )
        reasons.extend(f"{mode} {reason}" for reason in observation_reasons)
        if reproduction["cold_warm_exact"] is not True:
            reasons.append(f"{mode} cold/warm reproduction differs")
    return reasons


def _runtime_ab_reasons(control: Json, candidate: Json) -> list[str]:
    reasons = []
    for field in (
            "source_revision", "runtime_identity", "instance", "model_path",
            "target_prompt_tokens", "reproduction_max_tokens",
            "max_tokens_ladder", "seed", "run_id_sha256"):
        if control.get(field) != candidate.get(field):
            reasons.append(f"A/B report contract differs in {field}")
    control_contract = control["runtime_contract"]["contract"]
    candidate_contract = candidate["runtime_contract"]["contract"]
    for field in runtime_contract.FIELDS - {
            "environment", "optimization_label"}:
        if control_contract.get(field) != candidate_contract.get(field):
            reasons.append(f"A/B runtime contract differs in {field}")
    control_env = control_contract["environment"]
    candidate_env = candidate_contract["environment"]
    changed = {
        name for name in set(control_env) | set(candidate_env)
        if control_env.get(name) != candidate_env.get(name)
    }
    if changed != {"BI100_ATTN_COREX_FUSED_PREFILL"}:
        reasons.append(
            "A/B runtime environment did not change only fused prefill")
    return reasons


def _request_ab_reasons(
    control: Json,
    candidate: Json,
    label: str,
) -> tuple[list[str], bool, bool]:
    reasons = []
    for field in (
            "status", "prompt_tokens", "completion_tokens",
            "finish_reason", "request_contract_sha256"):
        if control.get(field) != candidate.get(field):
            reasons.append(f"{label} {field} differs")
    first_token_exact = (
        control.get("first_token_hmac_sha256")
        == candidate.get("first_token_hmac_sha256"))
    output_exact = (
        control.get("output_hmac_sha256")
        == candidate.get("output_hmac_sha256"))
    if not first_token_exact:
        reasons.append(f"{label} first token differs")
    return reasons, first_token_exact, output_exact


def compare(control: Any, candidate: Any) -> Json:
    validation_reasons = _validate_report(control, "control")
    validation_reasons.extend(_validate_report(candidate, "candidate"))
    comparison_contract_reasons = []
    quality_reasons = []
    rows = []
    all_first_tokens_exact = True
    all_outputs_exact = True
    first_divergent_budget = None

    if not validation_reasons:
        comparison_contract_reasons.extend(
            _runtime_ab_reasons(control, candidate))
    if not validation_reasons and not comparison_contract_reasons:
        control_repro = control["reproduction"]
        candidate_repro = candidate["reproduction"]
        for phase in ("cold", "warm"):
            row_reasons, first_exact, output_exact = _request_ab_reasons(
                control_repro[phase],
                candidate_repro[phase],
                f"reproduction {phase}",
            )
            comparison_contract_reasons.extend(
                reason for reason in row_reasons
                if not reason.endswith("first token differs"))
            quality_reasons.extend(
                reason for reason in row_reasons
                if reason.endswith("first token differs"))
            if not output_exact:
                quality_reasons.append(
                    f"reproduction {phase} output differs")
            all_first_tokens_exact &= first_exact
            all_outputs_exact &= output_exact
            rows.append({
                "phase": f"reproduction_{phase}",
                "max_tokens": diagnostic.REPRODUCTION_MAX_TOKENS,
                "first_token_exact": first_exact,
                "output_exact": output_exact,
            })
        for control_row, candidate_row in zip(
                control["ladder"], candidate["ladder"]):
            budget = control_row["max_tokens"]
            budget_first_exact = True
            budget_output_exact = True
            for repeat in ("warm_1", "warm_2"):
                row_reasons, first_exact, output_exact = (
                    _request_ab_reasons(
                        control_row[repeat],
                        candidate_row[repeat],
                        f"max_tokens={budget} {repeat}",
                    )
                )
                comparison_contract_reasons.extend(
                    reason for reason in row_reasons
                    if not reason.endswith("first token differs"))
                quality_reasons.extend(
                    reason for reason in row_reasons
                    if reason.endswith("first token differs"))
                budget_first_exact &= first_exact
                budget_output_exact &= output_exact
            all_first_tokens_exact &= budget_first_exact
            all_outputs_exact &= budget_output_exact
            if not budget_output_exact and first_divergent_budget is None:
                first_divergent_budget = budget
            if not budget_output_exact:
                quality_reasons.append(
                    f"max_tokens={budget} output differs")
            rows.append({
                "phase": "ladder",
                "max_tokens": budget,
                "first_token_exact": budget_first_exact,
                "output_exact": budget_output_exact,
            })

    diagnostic_valid = (
        not validation_reasons and not comparison_contract_reasons)
    strict_output_exact = diagnostic_valid and all_outputs_exact
    next_token_exact = diagnostic_valid and all_first_tokens_exact
    strict_quality_non_regression = (
        strict_output_exact and next_token_exact)
    reasons = (
        validation_reasons + comparison_contract_reasons + quality_reasons)
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "diagnostic_valid": diagnostic_valid,
        "strict_output_exact": strict_output_exact,
        "next_token_exact": next_token_exact,
        "quality_adjudication_required": (
            diagnostic_valid and next_token_exact and not strict_output_exact),
        "strict_quality_non_regression_authorized":
            strict_quality_non_regression,
        "production_promotion_authorized": False,
        "first_divergent_max_tokens": first_divergent_budget,
        "validation_reasons": (
            validation_reasons + comparison_contract_reasons),
        "quality_reasons": quality_reasons,
        "reasons": reasons,
        "comparisons": rows,
        "privacy": {
            "contains_raw_requests": False,
            "contains_raw_model_outputs": False,
            "contains_token_ids": False,
            "contains_credentials": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare(_load(args.control), _load(args.candidate))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        result = {
            "schema": SCHEMA,
            "version": VERSION,
            "diagnostic_valid": False,
            "strict_output_exact": False,
            "next_token_exact": False,
            "quality_adjudication_required": False,
            "strict_quality_non_regression_authorized": False,
            "production_promotion_authorized": False,
            "first_divergent_max_tokens": None,
            "validation_reasons": [str(error)],
            "quality_reasons": [],
            "reasons": [str(error)],
            "comparisons": [],
            "privacy": {
                "contains_raw_requests": False,
                "contains_raw_model_outputs": False,
                "contains_token_ids": False,
                "contains_credentials": False,
            },
        }
    diagnostic._atomic_write(args.out, result)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if result["strict_quality_non_regression_authorized"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
