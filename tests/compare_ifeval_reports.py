#!/usr/bin/env python3
"""Compare a frozen IFEval candidate with its same-generation baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


REPORT_SCHEMA = "bi100-ifeval-result-v1"
COMPARISON_SCHEMA = "bi100-ifeval-comparison-v1"
EXPECTED_MANIFEST_SHA256 = (
    "07ec4efb5fe7afaacb55723c1d53be4c2f58c840bbd6a54bf944e15cfbca1855"
)
ALLOWED_SWITCHES = {
    "gdn_cache_policy", "gdn_restore_mode", "fused_prefill",
    "kv_eviction_policy",
}
SWITCH_ENVIRONMENT = {
    "gdn_cache_policy": "BI100_GDN_CACHE_POLICY",
    "gdn_restore_mode": "BI100_GDN_RESTORE_MODE",
    "fused_prefill": "BI100_ATTN_COREX_FUSED_PREFILL",
    "kv_eviction_policy": "BI100_KV_EVICTION_POLICY",
}
Json = dict[str, Any]


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _summarize_cases(cases: list[Json]) -> Json:
    by_id: dict[str, Json] = {}
    by_family: dict[str, Json] = {}
    strict_prompts = 0
    loose_prompts = 0
    strict_instructions = 0
    loose_instructions = 0
    instruction_total = 0
    for case in cases:
        instruction_ids = case.get("instruction_id_list")
        strict = case.get("strict")
        loose = case.get("loose")
        if (
            case.get("status") != "pass"
            or not isinstance(instruction_ids, list)
            or not instruction_ids
            or any(not isinstance(value, str) or not value
                   for value in instruction_ids)
            or not isinstance(strict, list)
            or not isinstance(loose, list)
            or len(strict) != len(instruction_ids)
            or len(loose) != len(instruction_ids)
            or any(type(value) is not bool for value in strict + loose)
            or not _is_sha256(case.get("semantic_output_sha256"))
        ):
            raise ValueError("case outcome structure is incomplete")
        strict_prompts += int(all(strict))
        loose_prompts += int(all(loose))
        strict_instructions += sum(strict)
        loose_instructions += sum(loose)
        instruction_total += len(instruction_ids)
        for instruction_id, strict_value, loose_value in zip(
            instruction_ids, strict, loose
        ):
            family = instruction_id.split(":", 1)[0]
            for group, name in (
                (by_id, instruction_id),
                (by_family, family),
            ):
                counts = group.setdefault(name, {
                    "total": 0,
                    "strict_passed": 0,
                    "loose_passed": 0,
                })
                counts["total"] += 1
                counts["strict_passed"] += int(strict_value)
                counts["loose_passed"] += int(loose_value)
    return {
        "prompt_total": len(cases),
        "instruction_total": instruction_total,
        "strict_prompt_passed": strict_prompts,
        "loose_prompt_passed": loose_prompts,
        "strict_instruction_passed": strict_instructions,
        "loose_instruction_passed": loose_instructions,
        "by_instruction_id": dict(sorted(by_id.items())),
        "by_family": dict(sorted(by_family.items())),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path: Path, value: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
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


def validate_report(value: Any, label: str) -> list[str]:
    reasons = []
    if not isinstance(value, dict):
        return [f"{label}: report root must be an object"]
    if value.get("schema") != REPORT_SCHEMA or value.get("version") != 1:
        reasons.append(f"{label}: report schema or version differs")
    if (value.get("qualified") is not True
            or value.get("quality_run_eligible_for_baseline") is not True
            or value.get("promotion_authorized") is not False):
        reasons.append(f"{label}: report is not a qualified baseline-eligible run")
    manifest = value.get("manifest") or {}
    if (manifest.get("sha256") != EXPECTED_MANIFEST_SHA256
            or manifest.get("full_selection") is not True
            or len(manifest.get("selected_keys") or []) != 64
            or any(
                not isinstance(key, (str, int)) or isinstance(key, bool)
                for key in (manifest.get("selected_keys") or [])
            )):
        reasons.append(f"{label}: manifest identity or selection differs")
    runtime = value.get("runtime") or {}
    if (runtime.get("gpu_count") != 4
            or runtime.get("tensor_parallel_size") != 4
            or runtime.get("max_model_len") != 262144):
        reasons.append(f"{label}: runtime topology or capacity differs")
    contract_wrapper = value.get("runtime_contract") or {}
    contract = contract_wrapper.get("contract") or {}
    if (not isinstance(contract_wrapper.get("sha256"), str)
            or len(contract_wrapper["sha256"]) != 64
            or not isinstance(contract_wrapper.get("file_sha256"), str)
            or len(contract_wrapper["file_sha256"]) != 64
            or contract.get("schema") != "bi100-quality-runtime-contract-v1"):
        reasons.append(f"{label}: runtime contract is incomplete")
    summary = value.get("summary") or {}
    if (summary.get("prompt_total") != 64
            or not isinstance(summary.get("instruction_total"), int)
            or summary.get("instruction_total", 0) <= 0):
        reasons.append(f"{label}: score summary is incomplete")
    count_fields = (
        "strict_prompt_passed", "loose_prompt_passed",
        "strict_instruction_passed", "loose_instruction_passed",
    )
    if any(
        not isinstance(summary.get(name), int)
        or isinstance(summary.get(name), bool)
        for name in count_fields
    ):
        reasons.append(f"{label}: score totals are incomplete")
    for group_name in ("by_instruction_id", "by_family"):
        group = summary.get(group_name)
        if (
            not isinstance(group, dict)
            or not group
            or any(
                not isinstance(counts, dict)
                or any(
                    not isinstance(counts.get(name), int)
                    or isinstance(counts.get(name), bool)
                    for name in ("total", "strict_passed", "loose_passed")
                )
                for counts in group.values()
            )
        ):
            reasons.append(f"{label}: {group_name} summary is incomplete")
    cases = value.get("cases")
    if (
        not isinstance(cases, list)
        or len(cases) != 64
        or any(
            not isinstance(case, dict)
            or not isinstance(case.get("key"), (str, int))
            or isinstance(case.get("key"), bool)
            for case in cases
        )
    ):
        reasons.append(f"{label}: cases are incomplete")
    else:
        selected_keys = manifest["selected_keys"]
        if [case["key"] for case in cases] != selected_keys:
            reasons.append(f"{label}: selected case order differs")
        try:
            derived = _summarize_cases(cases)
        except ValueError:
            reasons.append(f"{label}: case outcomes are incomplete")
        else:
            for name in (
                "prompt_total", "instruction_total",
                "strict_prompt_passed", "loose_prompt_passed",
                "strict_instruction_passed", "loose_instruction_passed",
                "by_instruction_id", "by_family",
            ):
                if summary.get(name) != derived[name]:
                    reasons.append(
                        f"{label}: summary differs from cases in {name}")
    privacy = value.get("privacy") or {}
    if any(privacy.get(key) is not False for key in (
            "contains_credentials", "contains_raw_prompts",
            "contains_raw_model_outputs", "contains_reasoning_text")):
        reasons.append(f"{label}: report privacy contract differs")
    return reasons


def no_regression_reasons(baseline: Json, candidate: Json) -> list[str]:
    reasons = []
    for name in (
            "strict_prompt_passed", "loose_prompt_passed",
            "strict_instruction_passed", "loose_instruction_passed"):
        if candidate["summary"][name] < baseline["summary"][name]:
            reasons.append(f"candidate regressed aggregate {name}")
    for group in ("by_instruction_id", "by_family"):
        baseline_group = baseline["summary"][group]
        candidate_group = candidate["summary"][group]
        if set(candidate_group) != set(baseline_group):
            reasons.append(f"candidate {group} identities differ")
            continue
        for name in sorted(baseline_group):
            if candidate_group[name].get("total") != baseline_group[name].get("total"):
                reasons.append(f"candidate {group} total differs for {name}")
            for metric in ("strict_passed", "loose_passed"):
                if (candidate_group[name].get(metric, -1)
                        < baseline_group[name].get(metric, -1)):
                    reasons.append(
                        f"candidate regressed {group} {name} {metric}")
    return reasons


def pair_identity_reasons(
    baseline: Json,
    candidate: Json,
    allowed_switches: set[str],
) -> list[str]:
    """Validate that two reports form one controlled paired experiment."""
    reasons = validate_report(baseline, "baseline")
    reasons.extend(validate_report(candidate, "candidate"))
    if reasons:
        return reasons
    for field in ("manifest", "request_conversion", "evaluator"):
        if candidate[field] != baseline[field]:
            reasons.append(f"candidate field differs: {field}")
    baseline_runtime = dict(baseline["runtime"])
    candidate_runtime = dict(candidate["runtime"])
    baseline_optimization = baseline_runtime.pop("optimization")
    candidate_optimization = candidate_runtime.pop("optimization")
    baseline_runtime.pop("runtime_contract_sha256")
    candidate_runtime.pop("runtime_contract_sha256")
    if candidate_runtime != baseline_runtime:
        reasons.append("candidate runtime identity differs")
    if not allowed_switches <= ALLOWED_SWITCHES:
        reasons.append("comparison declares an unsupported switch")
    for name in ALLOWED_SWITCHES - allowed_switches:
        if candidate_optimization.get(name) != baseline_optimization.get(name):
            reasons.append(f"undeclared optimization differs: {name}")
    if not any(candidate_optimization.get(name)
               != baseline_optimization.get(name)
               for name in allowed_switches):
        reasons.append("declared optimization switches contain no difference")

    baseline_contract = dict(baseline["runtime_contract"]["contract"])
    candidate_contract = dict(candidate["runtime_contract"]["contract"])
    baseline_environment = dict(baseline_contract.pop("environment"))
    candidate_environment = dict(candidate_contract.pop("environment"))
    baseline_contract.pop("optimization_label")
    candidate_contract.pop("optimization_label")
    if candidate_contract != baseline_contract:
        reasons.append("candidate runtime contract differs outside environment")
    allowed_environment = {
        SWITCH_ENVIRONMENT[name] for name in allowed_switches
        if name in SWITCH_ENVIRONMENT
    }
    for name in sorted(set(baseline_environment) | set(candidate_environment)):
        if (baseline_environment.get(name) != candidate_environment.get(name)
                and name not in allowed_environment):
            reasons.append(f"candidate runtime environment differs: {name}")

    baseline_cases = {case["key"]: case for case in baseline["cases"]}
    candidate_cases = {case["key"]: case for case in candidate["cases"]}
    if len(baseline_cases) != len(baseline["cases"]):
        reasons.append("baseline case identities are not unique")
    if len(candidate_cases) != len(candidate["cases"]):
        reasons.append("candidate case identities are not unique")
    if set(candidate_cases) != set(baseline_cases):
        reasons.append("candidate case identities differ")
    else:
        for key in sorted(baseline_cases):
            if (candidate_cases[key].get("instruction_id_list")
                    != baseline_cases[key].get("instruction_id_list")):
                reasons.append(
                    f"candidate instruction identities differ for key {key}")
    return reasons


def comparison_reasons(
    baseline: Json,
    candidate: Json,
    allowed_switches: set[str],
    require_exact_output: bool,
) -> list[str]:
    reasons = pair_identity_reasons(
        baseline, candidate, allowed_switches)
    if reasons:
        return reasons
    reasons.extend(no_regression_reasons(baseline, candidate))

    if require_exact_output:
        baseline_cases = {case["key"]: case for case in baseline["cases"]}
        candidate_cases = {case["key"]: case for case in candidate["cases"]}
        for key in sorted(baseline_cases):
            if (candidate_cases[key].get("semantic_output_sha256")
                    != baseline_cases[key].get("semantic_output_sha256")):
                reasons.append(f"candidate output differs for key {key}")
    return reasons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--allowed-switch", action="append", default=[])
    parser.add_argument("--require-exact-output", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    allowed = set(args.allowed_switch)
    reasons = comparison_reasons(
        baseline, candidate, allowed, args.require_exact_output)
    report = {
        "schema": COMPARISON_SCHEMA,
        "version": 1,
        "qualified": not reasons,
        "promotion_authorized": False,
        "baseline_sha256": sha256(args.baseline),
        "candidate_sha256": sha256(args.candidate),
        "allowed_switches": sorted(allowed),
        "require_exact_output": args.require_exact_output,
        "reasons": reasons,
        "score_delta": {
            name: (candidate.get("summary") or {}).get(name, 0)
            - (baseline.get("summary") or {}).get(name, 0)
            for name in (
                "strict_prompt_passed", "loose_prompt_passed",
                "strict_instruction_passed", "loose_instruction_passed")
        },
    }
    atomic_write(args.out, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
