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
    "8ac44a97a6f569056415deedb8a59cbc815cbad6577cbb2e713016864cc7f0fa"
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
            or len(manifest.get("selected_keys") or []) != 64):
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


def comparison_reasons(
    baseline: Json,
    candidate: Json,
    allowed_switches: set[str],
    require_exact_output: bool,
) -> list[str]:
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
    reasons.extend(no_regression_reasons(baseline, candidate))

    baseline_cases = {case["key"]: case for case in baseline["cases"]}
    candidate_cases = {case["key"]: case for case in candidate["cases"]}
    if set(candidate_cases) != set(baseline_cases):
        reasons.append("candidate case identities differ")
    elif require_exact_output:
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
