#!/usr/bin/env python3
"""Compare privacy-safe Agent workload reports for exact non-regression."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import agent_workload_matrix as workload
import quality_runtime_contract as runtime_contract


Json = dict[str, Any]
SCHEMA = "bi100-agent-workload-comparison-v1"
VERSION = 1
ALLOWED_AB_ENV_DIFFERENCES = {
    "BI100_ATTN_COREX_FUSED_PREFILL",
    "BI100_GDN_COMBINED_QK_NORM",
    "BI100_GDN_CACHE_POLICY",
    "BI100_GDN_RESTORE_MODE",
    "BI100_KV_EVICTION_POLICY",
}
OBSERVATION_FIELDS = {
    "elapsed_s", "finish_reason", "content_chars", "reasoning_chars",
    "tool_call_count", "prompt_tokens", "cached_tokens",
    "completion_tokens", "semantic_output_sha256", "facts",
}


def is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, report: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                     dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def report_reasons(report: Any, label: str, manifest: Json,
                   manifest_sha: str) -> list[str]:
    reasons: list[str] = []
    if not isinstance(report, dict):
        return [f"{label}: report root must be an object"]
    if (report.get("schema") != workload.REPORT_SCHEMA
            or report.get("version") != workload.REPORT_VERSION):
        reasons.append(f"{label}: report schema or version is invalid")
    if report.get("qualified") is not True:
        reasons.append(f"{label}: Agent workload run is not qualified")
    if report.get("promotion_authorized") is not False:
        reasons.append(f"{label}: standalone report authorized promotion")
    if report.get("manifest") != {
            "path_name": workload.DEFAULT_MANIFEST.name,
            "sha256": manifest_sha,
            "revision": manifest["revision"],
            "case_count": len(manifest["cases"]),
    }:
        reasons.append(f"{label}: manifest identity differs")
    if report.get("privacy") != {
            "contains_raw_requests": False,
            "contains_raw_model_outputs": False,
            "contains_tool_arguments": False,
            "contains_credentials": False,
    }:
        reasons.append(f"{label}: privacy declaration is invalid")

    runtime = report.get("runtime") or {}
    if (not runtime_contract.is_git_revision(runtime.get("source_revision"))
            or not isinstance(runtime.get("runtime_identity"), str)
            or not runtime.get("runtime_identity")
            or not is_sha256(runtime.get("runtime_overlay_sha256"))
            or not is_sha256(runtime.get("runtime_contract_sha256"))
            or not isinstance(runtime.get("instance"), str)
            or not runtime.get("instance")
            or runtime.get("gpu_count") != 4
            or runtime.get("tensor_parallel_size") != 4
            or runtime.get("max_model_len") != 262144):
        reasons.append(f"{label}: runtime identity or topology is invalid")
    wrapper = report.get("runtime_contract") or {}
    contract = wrapper.get("contract") if isinstance(wrapper, dict) else None
    if (not isinstance(wrapper, dict)
            or set(wrapper) != {"sha256", "contract"}
            or not isinstance(contract, dict)):
        reasons.append(f"{label}: runtime contract wrapper is invalid")
    else:
        expected = {
            "source_revision": runtime.get("source_revision"),
            "runtime_identity": runtime.get("runtime_identity"),
            "instance": runtime.get("instance"),
            "gpu_count": 4,
            "tensor_parallel_size": 4,
            "max_model_len": 262144,
            "model_path": contract.get("model_path"),
            "tokenizer_path": contract.get("tokenizer_path"),
            "served_model_name": contract.get("served_model_name"),
        }
        try:
            digest = runtime_contract.validate_runtime_contract(
                contract, expected, require_cache_trace=True)
        except runtime_contract.RuntimeContractError as error:
            reasons.append(f"{label}: {error}")
        else:
            if wrapper.get("sha256") != digest:
                reasons.append(f"{label}: runtime contract SHA-256 differs")
            if runtime.get("runtime_contract_sha256") != digest:
                reasons.append(f"{label}: runtime contract binding differs")

    generator = report.get("generator") or {}
    if (not isinstance(generator, dict)
            or set(generator) != {"runner_sha256", "seed"}
            or not is_sha256(generator.get("runner_sha256"))
            or generator.get("seed") != manifest["seed"]):
        reasons.append(f"{label}: generator identity is invalid")
    cases = report.get("cases")
    expected_ids = [case["id"] for case in manifest["cases"]]
    if not isinstance(cases, list) or [
            case.get("id") for case in cases if isinstance(case, dict)
    ] != expected_ids:
        reasons.append(f"{label}: case identity or order differs")
        return reasons
    for case in cases:
        case_id = case["id"]
        if (set(case) != {
                "id", "status", "error_type", "error_sha256", "observation"}
                or case.get("status") != "pass"
                or case.get("error_type") != ""
                or case.get("error_sha256") is not None):
            reasons.append(f"{label}: {case_id} did not pass cleanly")
            continue
        observation = case.get("observation") or {}
        if (not isinstance(observation, dict)
                or set(observation) != OBSERVATION_FIELDS
                or not is_sha256(observation.get("semantic_output_sha256"))
                or not isinstance(observation.get("facts"), dict)
                or not observation["facts"]):
            reasons.append(f"{label}: {case_id} observation is invalid")
    summary = report.get("summary") or {}
    if summary != {
            "complete": True,
            "passed": len(expected_ids),
            "failed": 0,
            "total": len(expected_ids),
    }:
        reasons.append(f"{label}: summary differs from complete pass")
    return reasons


def compare_reports(baseline: Any, candidate: Any) -> Json:
    manifest, manifest_sha = workload.load_manifest(workload.DEFAULT_MANIFEST)
    reasons = report_reasons(baseline, "baseline", manifest, manifest_sha)
    reasons.extend(report_reasons(
        candidate, "candidate", manifest, manifest_sha))
    case_results: list[Json] = []

    if isinstance(baseline, dict) and isinstance(candidate, dict):
        baseline_runtime = baseline.get("runtime") or {}
        candidate_runtime = candidate.get("runtime") or {}
        for field in (
                "source_revision", "runtime_identity", "runtime_overlay_sha256",
                "instance", "gpu_count", "tensor_parallel_size",
                "max_model_len"):
            if baseline_runtime.get(field) != candidate_runtime.get(field):
                reasons.append(f"runtime contract differs in {field}")
        if baseline.get("generator") != candidate.get("generator"):
            reasons.append("baseline and candidate generators differ")

        baseline_contract = (
            (baseline.get("runtime_contract") or {}).get("contract") or {})
        candidate_contract = (
            (candidate.get("runtime_contract") or {}).get("contract") or {})
        for field in (
                "source_revision", "runtime_identity", "runtime_overlay_sha256",
                "instance", "base_image", "command", "gpu_count",
                "tensor_parallel_size", "max_model_len", "model_path",
                "tokenizer_path", "served_model_name", "cache_trace_enabled"):
            if baseline_contract.get(field) != candidate_contract.get(field):
                reasons.append(f"A/B runtime contract differs in {field}")
        baseline_env = baseline_contract.get("environment") or {}
        candidate_env = candidate_contract.get("environment") or {}
        changed_env = {
            key for key in set(baseline_env) | set(candidate_env)
            if baseline_env.get(key) != candidate_env.get(key)
        }
        disallowed = changed_env - ALLOWED_AB_ENV_DIFFERENCES
        if disallowed:
            reasons.append(
                "A/B changed disallowed runtime environment values: "
                + ", ".join(sorted(disallowed)))

        baseline_cases = {
            case.get("id"): case for case in baseline.get("cases", [])
            if isinstance(case, dict)
        }
        candidate_cases = {
            case.get("id"): case for case in candidate.get("cases", [])
            if isinstance(case, dict)
        }
        for expected in manifest["cases"]:
            case_id = expected["id"]
            base_observation = (
                baseline_cases.get(case_id, {}).get("observation") or {})
            candidate_observation = (
                candidate_cases.get(case_id, {}).get("observation") or {})
            failures = []
            for field in (
                    "finish_reason", "content_chars", "reasoning_chars",
                    "tool_call_count", "prompt_tokens", "completion_tokens",
                    "semantic_output_sha256", "facts"):
                if base_observation.get(field) != candidate_observation.get(field):
                    failures.append(f"{field} differs")
            case_results.append({
                "id": case_id,
                "qualified": not failures,
                "reasons": failures,
            })
            reasons.extend(f"{case_id}: {reason}" for reason in failures)

    qualified = not reasons and len(case_results) == len(manifest["cases"])
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "qualified": qualified,
        "agent_quality_non_regression_authorized": qualified,
        "overall_promotion_authorized": False,
        "summary": {
            "compared_cases": len(case_results),
            "qualified_cases": sum(
                case["qualified"] for case in case_results),
        },
        "cases": case_results,
        "reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = compare_reports(
        json.loads(args.baseline.read_text(encoding="utf-8")),
        json.loads(args.candidate.read_text(encoding="utf-8")),
    )
    report["inputs"] = {
        "baseline_file_sha256": file_sha256(args.baseline),
        "candidate_file_sha256": file_sha256(args.candidate),
    }
    atomic_write(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
