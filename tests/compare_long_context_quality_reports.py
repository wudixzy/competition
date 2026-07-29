#!/usr/bin/env python3
"""Compare complete long-context quality reports against a CoreX baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import quality_runtime_contract as runtime_contract
import validate_quality_data_manifests as manifest_validator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "quality/long_context_matrix.v6.json"
EXPECTED_MANIFEST_SHA256 = (
    "787d603818e5238b8fd45332d30c2991a7b0873d0012e2f0caad0a5c50b40115"
)
REPORT_SCHEMA = "bi100-long-context-quality-result-v6"
COMPARISON_SCHEMA = "bi100-long-context-quality-comparison-v5"
EXPECTED_CASES = 12
BASE_IMAGE = runtime_contract.BASE_IMAGE
Json = dict[str, Any]

ALLOWED_AB_ENV_DIFFERENCES = {
    "BI100_ATTN_COREX_FUSED_PREFILL",
    "BI100_GDN_COMBINED_QK_NORM",
    "BI100_GDN_CACHE_POLICY",
    "BI100_GDN_RESTORE_MODE",
    "BI100_KV_EVICTION_POLICY",
}

REQUEST_COUNTS = {
    "short_basic_recall": 1,
    "4k_cold_warm_recall": 2,
    "32k_partial_branch": 4,
    "32k_multimodal_isolation": 3,
    "65k_multiturn_large_tools": 2,
    "65k_long_tool_result": 2,
    "65k_interleaved_sessions": 4,
    "131k_cold_warm_recall": 2,
    "131k_reasoning_recall": 1,
    "235k_agent_large_output_budget": 2,
    "235k_partial_branch": 5,
    "near_262k_capacity": 3,
}
CONSTRUCTION_COUNTS = {
    "short_basic_recall": 1,
    "4k_cold_warm_recall": 1,
    "32k_partial_branch": 3,
    "32k_multimodal_isolation": 2,
    "65k_multiturn_large_tools": 1,
    "65k_long_tool_result": 1,
    "65k_interleaved_sessions": 2,
    "131k_cold_warm_recall": 1,
    "131k_reasoning_recall": 1,
    "235k_agent_large_output_budget": 1,
    "235k_partial_branch": 3,
    "near_262k_capacity": 2,
}
PARTIAL_POLICY_TRUE_FACTS = {
    "fine32": (
        "first_sibling_strict_partial_hit",
        "subsequent_sibling_strict_partial_hit",
        "subsequent_sibling_restored",
    ),
    "admission64": (
        "first_sibling_effective_miss",
        "repeated_branch_admitted",
        "subsequent_sibling_strict_partial_hit",
        "subsequent_sibling_restored",
    ),
}
TRUE_FACTS = {
    "short_basic_recall": ("marker_rule_passed",),
    "4k_cold_warm_recall": ("marker_rule_passed", "cold_warm_exact"),
    "32k_partial_branch": (
        "branch_markers_correct", "cache_trace_session_attested",
        "cold_warm_exact", "strict_partial_hit"),
    "32k_multimodal_isolation": (
        "red_blue_rules_passed", "same_image_cold_warm_exact",
        "different_image_isolated", "image_identity_digests_distinct",
        "cache_trace_identity_passed"),
    "65k_multiturn_large_tools": (
        "tool_call_rule_passed", "cold_warm_exact"),
    "65k_long_tool_result": ("marker_rule_passed", "cold_warm_exact"),
    "65k_interleaved_sessions": (
        "session_markers_correct", "warm_outputs_exact",
        "session_cache_isolated"),
    "131k_cold_warm_recall": ("marker_rule_passed", "cold_warm_exact"),
    "131k_reasoning_recall": (
        "answer_rule_passed", "marker_rule_passed",
        "reasoning_content_split", "natural_finish_before_max_tokens",
        "content_arithmetic_present", "content_contains_expected",
        "content_expected_single_occurrence", "content_expected_suffix",
        "content_markers_in_order", "content_markers_present"),
    "235k_agent_large_output_budget": (
        "large_max_tokens_accepted", "tool_call_rule_passed",
        "reasoning_present", "cold_warm_exact",
        "natural_finish_before_max_tokens"),
    "235k_partial_branch": (
        "branch_markers_correct", "cache_trace_session_attested",
        "cold_warm_exact", "strict_partial_hit"),
    "near_262k_capacity": (
        "marker_rule_passed", "cold_warm_exact",
        "exact_capacity_boundary_passed",
        "minus_one_capacity_boundary_passed"),
}
EXACT_BASELINE_IDS = {
    "short_basic_recall", "4k_cold_warm_recall", "32k_partial_branch",
    "32k_multimodal_isolation", "65k_multiturn_large_tools",
    "65k_long_tool_result", "65k_interleaved_sessions",
}
NEXT_TOKEN_IDS = {"131k_cold_warm_recall", "near_262k_capacity"}
SEMANTIC_IDS = {
    "131k_reasoning_recall", "235k_agent_large_output_budget",
    "235k_partial_branch",
}


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _is_git_revision(value: Any) -> bool:
    return (isinstance(value, str) and 7 <= len(value) <= 64
            and all(character in "0123456789abcdef" for character in value))


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _load_manifest(path: Path) -> tuple[Json, str]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_MANIFEST_SHA256:
        raise ValueError("long-context matrix SHA-256 differs")
    value = json.loads(payload)
    reasons = manifest_validator.validate_matrix(value)
    if reasons:
        raise ValueError("long-context matrix is invalid")
    return value, digest


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


def _validate_tool_call_structure(value: Any, label: str) -> list[str]:
    root_fields = {"container_type", "count", "calls"}
    if not isinstance(value, dict) or set(value) != root_fields:
        return [f"{label}: tool-call structure fields are invalid"]
    reasons = []
    if not isinstance(value["container_type"], str):
        reasons.append(f"{label}: tool-call container type is invalid")
    calls = value["calls"]
    count = value["count"]
    if not isinstance(calls, list):
        return reasons + [f"{label}: tool-call structures are invalid"]
    if count is None:
        if calls:
            reasons.append(f"{label}: absent tool-call count has entries")
        return reasons
    if (not isinstance(count, int) or isinstance(count, bool)
            or count < 0 or count != len(calls)):
        reasons.append(f"{label}: tool-call count is invalid")
    base_fields = {
        "call_type", "function_type", "name_sha256", "arguments_type",
    }
    argument_fields = {
        "arguments_length", "arguments_sha256", "starts_object",
        "ends_object", "contains_tool_call_tag", "contains_function_prefix",
        "contains_code_fence", "json_type",
    }
    json_types = {"dict", "list", "str", "int", "float", "bool", "NoneType"}
    for index, call in enumerate(calls, 1):
        call_label = f"{label}: tool call {index}"
        if not isinstance(call, dict) or not base_fields <= set(call):
            reasons.append(f"{call_label} fields are invalid")
            continue
        if not all(isinstance(call[field], str) for field in (
                "call_type", "function_type", "arguments_type")):
            reasons.append(f"{call_label} type fields are invalid")
        if not _is_sha256(call.get("name_sha256")):
            reasons.append(f"{call_label} name digest is invalid")
        if call.get("arguments_type") == "str":
            if set(call) != base_fields | argument_fields:
                reasons.append(f"{call_label} argument fields are invalid")
                continue
            if (not isinstance(call["arguments_length"], int)
                    or isinstance(call["arguments_length"], bool)
                    or call["arguments_length"] < 0
                    or not _is_sha256(call["arguments_sha256"])):
                reasons.append(f"{call_label} argument identity is invalid")
            for field in (
                    "starts_object", "ends_object", "contains_tool_call_tag",
                    "contains_function_prefix", "contains_code_fence"):
                if not isinstance(call[field], bool):
                    reasons.append(f"{call_label} {field} is invalid")
            if call["json_type"] not in json_types:
                reasons.append(f"{call_label} JSON type is invalid")
        elif set(call) != base_fields:
            reasons.append(f"{call_label} non-string argument fields are invalid")
    return reasons


def _validate_request(
    request: Any,
    case: Json,
    label: str,
    expected_local_prompt_tokens: int,
) -> list[str]:
    required = {
        "status", "model", "local_prompt_tokens", "prompt_tokens",
        "cached_tokens", "completion_tokens", "total_tokens",
        "finish_reason", "semantic_output_sha256", "content_sha256",
        "content_length", "reasoning_sha256", "reasoning_length",
        "tool_calls_sha256", "tool_call_structure",
        "first_generated_token_sha256", "request_contract_sha256",
        "token_accounting", "protocol_validated", "elapsed_s",
    }
    if not isinstance(request, dict) or set(request) != required:
        return [f"{label}: request summary fields are invalid"]
    reasons = []
    integer_fields = (
        "status", "local_prompt_tokens", "prompt_tokens", "cached_tokens",
        "completion_tokens", "total_tokens", "content_length",
        "reasoning_length",
    )
    if any(not isinstance(request[field], int)
           or isinstance(request[field], bool) for field in integer_fields):
        reasons.append(f"{label}: integer counters are invalid")
        return reasons
    if request["content_length"] < 0 or request["reasoning_length"] < 0:
        reasons.append(f"{label}: response lengths are invalid")
    if request["status"] != 200:
        reasons.append(f"{label}: request status is not HTTP 200")
    if request["model"] != "llm":
        reasons.append(f"{label}: response model differs")
    if request["local_prompt_tokens"] != expected_local_prompt_tokens:
        reasons.append(f"{label}: local prompt token target differs")
    expected_accounting = (
        "local_template_plus_vision"
        if case["id"] == "32k_multimodal_isolation"
        else "server_exact"
    )
    if request["token_accounting"] != expected_accounting:
        reasons.append(f"{label}: token accounting mode differs")
    if expected_accounting == "server_exact":
        if request["prompt_tokens"] != expected_local_prompt_tokens:
            reasons.append(f"{label}: server prompt token target differs")
    elif request["prompt_tokens"] < expected_local_prompt_tokens:
        reasons.append(f"{label}: multimodal prompt lost local tokens")
    if (request["cached_tokens"] < 0
            or request["cached_tokens"] > request["prompt_tokens"]):
        reasons.append(f"{label}: cached token count is invalid")
    if request["total_tokens"] != (
            request["prompt_tokens"] + request["completion_tokens"]):
        reasons.append(f"{label}: total token count is inconsistent")
    if not (case["min_completion_tokens"]
            <= request["completion_tokens"] <= case["max_tokens"]):
        reasons.append(f"{label}: completion token count is outside contract")
    if (case["id"] in {
            "131k_reasoning_recall", "235k_agent_large_output_budget"}
            and request["completion_tokens"] >= case["max_tokens"]):
        reasons.append(f"{label}: response did not finish before the cap")
    if not isinstance(request["finish_reason"], str):
        reasons.append(f"{label}: finish_reason is invalid")
    for field in (
            "semantic_output_sha256", "content_sha256", "reasoning_sha256",
            "tool_calls_sha256", "request_contract_sha256"):
        if not _is_sha256(request[field]):
            reasons.append(f"{label}: {field} is invalid")
    if request["protocol_validated"] is not True:
        reasons.append(f"{label}: protocol validation evidence is missing")
    first_token = request["first_generated_token_sha256"]
    if first_token is not None and not _is_sha256(first_token):
        reasons.append(f"{label}: first generated token digest is invalid")
    if case["id"] in NEXT_TOKEN_IDS and not _is_sha256(first_token):
        reasons.append(f"{label}: next-token evidence is missing")
    reasons.extend(_validate_tool_call_structure(
        request["tool_call_structure"], label))
    elapsed = request["elapsed_s"]
    if (not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool)
            or not math.isfinite(elapsed) or elapsed <= 0):
        reasons.append(f"{label}: elapsed_s is invalid")
    return reasons


def _validate_construction(
    construction: Any,
    expected_target: int,
    expected_thinking: bool,
    label: str,
) -> list[str]:
    required = {
        "schema", "target_prompt_tokens", "local_prompt_tokens",
        "fixed_prompt_tokens", "filler_token_ids_requested",
        "filler_text_sha256", "filler_source_sha256",
        "rendered_prompt_token_ids_sha256", "messages_sha256",
        "tools_sha256", "thinking", "template_kwargs_mode", "attempts",
    }
    if not isinstance(construction, dict) or set(construction) != required:
        return [f"{label}: construction evidence fields are invalid"]
    reasons = []
    if construction["schema"] != "bi100-exact-chat-prompt-v1":
        reasons.append(f"{label}: construction schema differs")
    if (construction["target_prompt_tokens"] != expected_target
            or construction["local_prompt_tokens"] != expected_target):
        reasons.append(f"{label}: construction token target differs")
    for field in ("fixed_prompt_tokens", "filler_token_ids_requested"):
        value = construction[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            reasons.append(f"{label}: construction {field} is invalid")
    if (not isinstance(construction["attempts"], int)
            or isinstance(construction["attempts"], bool)
            or construction["attempts"] <= 0):
        reasons.append(f"{label}: construction attempts are invalid")
    for field in (
            "filler_text_sha256", "filler_source_sha256",
            "rendered_prompt_token_ids_sha256", "messages_sha256",
            "tools_sha256"):
        if not _is_sha256(construction[field]):
            reasons.append(f"{label}: construction {field} is invalid")
    if construction["thinking"] is not expected_thinking:
        reasons.append(f"{label}: construction thinking mode differs")
    if construction["template_kwargs_mode"] not in ("direct", "nested"):
        reasons.append(f"{label}: construction template mode is invalid")
    return reasons


def _validate_case(
    case: Any,
    expected: Json,
    label: str,
    gdn_cache_policy: str | None,
) -> list[str]:
    reasons = []
    metadata = (
        "ordinal", "id", "tier", "target_prompt_tokens", "max_tokens",
        "min_completion_tokens", "request_shape", "cache_scenario",
        "capabilities", "equivalence", "validation",
    )
    if not isinstance(case, dict):
        return [f"{label}: case result is not an object"]
    if any(case.get(field) != expected[field] for field in metadata):
        reasons.append(f"{label}: case metadata differs from matrix")
    if (case.get("status") != "pass" or case.get("ok") is not True
            or case.get("error_code") != ""):
        reasons.append(f"{label}: case is not a clean pass")
    elapsed = case.get("elapsed_s")
    if (not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool)
            or not math.isfinite(elapsed) or elapsed <= 0):
        reasons.append(f"{label}: case elapsed_s is invalid")
    observation = case.get("observation") or {}
    if set(observation) != {"requests", "construction", "facts"}:
        reasons.append(f"{label}: observation fields are invalid")
        return reasons
    requests = observation["requests"]
    expected_count = REQUEST_COUNTS[expected["id"]]
    if not isinstance(requests, list) or len(requests) != expected_count:
        reasons.append(f"{label}: request count differs")
        requests = []
    request_targets = [expected["target_prompt_tokens"]] * expected_count
    if expected["id"] == "near_262k_capacity":
        request_targets[-1] = expected["target_prompt_tokens"] - 1
    for index, (request, target) in enumerate(
            zip(requests, request_targets), 1):
        reasons.extend(_validate_request(
            request, expected, f"{label}: request {index}", target))
    request_shape_valid = all(
        isinstance(request, dict)
        and {"cached_tokens", "prompt_tokens", "request_contract_sha256",
             "semantic_output_sha256", "completion_tokens", "finish_reason",
             "content_sha256", "reasoning_sha256", "tool_calls_sha256"}
        <= set(request)
        for request in requests
    )
    constructions = observation["construction"]
    expected_constructions = CONSTRUCTION_COUNTS[expected["id"]]
    if (not isinstance(constructions, list)
            or len(constructions) != expected_constructions):
        reasons.append(f"{label}: construction evidence count differs")
        constructions = []
    construction_targets = [expected["target_prompt_tokens"]] * (
        expected_constructions)
    if expected["id"] == "near_262k_capacity":
        construction_targets[-1] = expected["target_prompt_tokens"] - 1
    expected_thinking = expected["id"] in {
        "131k_reasoning_recall", "235k_agent_large_output_budget",
    }
    for index, (construction, target) in enumerate(
            zip(constructions, construction_targets), 1):
        reasons.extend(_validate_construction(
            construction, target, expected_thinking,
            f"{label}: construction {index}"))
    facts = observation["facts"]
    if not isinstance(facts, dict):
        reasons.append(f"{label}: facts are invalid")
        facts = {}
    for fact in TRUE_FACTS[expected["id"]]:
        if facts.get(fact) is not True:
            reasons.append(f"{label}: required fact {fact} is not true")
    if expected["id"] in {"32k_partial_branch", "235k_partial_branch"}:
        policy_facts = PARTIAL_POLICY_TRUE_FACTS.get(gdn_cache_policy or "")
        if policy_facts is None:
            reasons.append(f"{label}: GDN cache policy is invalid")
        else:
            for fact in policy_facts:
                if facts.get(fact) is not True:
                    reasons.append(
                        f"{label}: required policy fact {fact} is not true")
    if expected["id"] in (
            "65k_multiturn_large_tools", "235k_agent_large_output_budget"):
        if facts.get("tool_count") != 92:
            reasons.append(f"{label}: large tools count differs")
    if expected["id"] == "235k_agent_large_output_budget":
        if facts.get("tool_choice_mode") != "auto":
            reasons.append(f"{label}: Agent tool-choice mode differs")
        if facts.get("tool_content_mode") != "optional":
            reasons.append(f"{label}: Agent tool-content mode differs")
    if expected["id"] == "32k_multimodal_isolation":
        assets = manifest_validator.EXPECTED_GENERATED_ASSETS
        if (facts.get("red_image_sha256")
                != assets["red_png_data_url_sha256"]
                or facts.get("blue_image_sha256")
                != assets["blue_png_data_url_sha256"]):
            reasons.append(f"{label}: multimodal fixture identities differ")
        if (not _is_sha256(facts.get("cache_trace_records_sha256"))
                or facts.get("cache_trace_version") != 4):
            reasons.append(f"{label}: multimodal cache trace proof is invalid")
    if constructions:
        assets = manifest_validator.EXPECTED_GENERATED_ASSETS
        expected_tool_sha = {
            "65k_multiturn_large_tools": "large_tools_65k_sha256",
            "65k_long_tool_result": "fetch_record_tool_sha256",
            "235k_agent_large_output_budget": "large_tools_235k_sha256",
        }.get(expected["id"])
        if (expected_tool_sha is not None
                and isinstance(constructions[0], dict)
                and constructions[0].get("tools_sha256")
                != assets[expected_tool_sha]):
            reasons.append(f"{label}: generated tool schema identity differs")

    if len(requests) == expected_count and request_shape_valid:
        cached = [request["cached_tokens"] for request in requests]
        if expected["id"] in (
                "4k_cold_warm_recall", "65k_multiturn_large_tools",
                "65k_long_tool_result", "131k_cold_warm_recall",
                "235k_agent_large_output_budget", "near_262k_capacity"):
            if cached[0] != 0 or cached[1] <= 0:
                reasons.append(f"{label}: cold/warm cache accounting differs")
        elif expected["id"] in ("32k_partial_branch", "235k_partial_branch"):
            if (cached[0] != 0 or cached[1] <= 0
                    or not 0 < cached[3] < expected["target_prompt_tokens"]):
                reasons.append(f"{label}: partial branch cache accounting differs")
            if gdn_cache_policy == "fine32" and not (
                    0 < cached[2] < expected["target_prompt_tokens"]):
                reasons.append(
                    f"{label}: fine32 first sibling cache accounting differs")
            if gdn_cache_policy == "admission64" and cached[2] != 0:
                reasons.append(
                    f"{label}: admission64 first sibling must effectively miss")
            if expected["id"] == "235k_partial_branch" and cached[4] <= 0:
                reasons.append(f"{label}: branch warm repeat did not hit cache")
        elif expected["id"] == "32k_multimodal_isolation":
            if cached[0] != 0 or cached[1] <= 0 or cached[2] != 0:
                reasons.append(f"{label}: multimodal cache isolation differs")
            if len({request["prompt_tokens"] for request in requests}) != 1:
                reasons.append(f"{label}: multimodal vision token counts differ")
        elif expected["id"] == "65k_interleaved_sessions":
            if (cached[0] != 0 or cached[1] != 0
                    or cached[2] <= 0 or cached[3] <= 0):
                reasons.append(f"{label}: interleaved cache isolation differs")
        repeated_pairs = {
            "4k_cold_warm_recall": ((0, 1),),
            "32k_partial_branch": ((0, 1),),
            "32k_multimodal_isolation": ((0, 1),),
            "65k_multiturn_large_tools": ((0, 1),),
            "65k_long_tool_result": ((0, 1),),
            "65k_interleaved_sessions": ((0, 2), (1, 3)),
            "131k_cold_warm_recall": ((0, 1),),
            "235k_agent_large_output_budget": ((0, 1),),
            "235k_partial_branch": ((0, 1), (2, 4)),
            "near_262k_capacity": ((0, 1),),
        }.get(expected["id"], ())
        for left, right in repeated_pairs:
            for field in (
                    "request_contract_sha256", "semantic_output_sha256",
                    "completion_tokens", "finish_reason", "content_sha256",
                    "reasoning_sha256", "tool_calls_sha256"):
                if requests[left][field] != requests[right][field]:
                    reasons.append(
                        f"{label}: repeated requests differ in {field}")
        if expected["id"] == "32k_multimodal_isolation" and (
                requests[0]["request_contract_sha256"]
                == requests[2]["request_contract_sha256"]):
            reasons.append(f"{label}: distinct images have one request identity")
        if expected["id"] in {
                "65k_multiturn_large_tools",
                "235k_agent_large_output_budget"}:
            if any(request["finish_reason"] != "tool_calls"
                   for request in requests):
                reasons.append(f"{label}: tool call did not finish as tool_calls")
    return reasons


def _validate_report(
    report: Any,
    label: str,
    manifest: Json,
    manifest_sha: str,
    manifest_name: str,
) -> tuple[dict[str, Json], list[str]]:
    reasons = []
    gdn_cache_policy: str | None = None
    if not isinstance(report, dict):
        return {}, [f"{label}: report root is not an object"]
    if report.get("schema") != REPORT_SCHEMA or report.get("version") != 6:
        reasons.append(f"{label}: report schema or version is invalid")
    if (report.get("qualified") is not True
            or report.get("quality_run_eligible_for_baseline") is not True
            or report.get("overall_promotion_authorized") is not False):
        reasons.append(f"{label}: report qualification state is invalid")
    expected_manifest = {
        "path_name": manifest_name,
        "sha256": manifest_sha,
        "total_cases": EXPECTED_CASES,
        "seed": 20260724,
    }
    if report.get("manifest") != expected_manifest:
        reasons.append(f"{label}: matrix identity differs")
    for field in ("label", "created_at_utc"):
        if not isinstance(report.get(field), str) or not report[field]:
            reasons.append(f"{label}: report field {field} is invalid")
    if not _is_sha256(report.get("run_id_sha256")):
        reasons.append(f"{label}: report run identity is invalid")
    runtime = report.get("runtime") or {}
    for field in ("runtime_identity", "instance", "model_path"):
        if not isinstance(runtime.get(field), str) or not runtime[field]:
            reasons.append(f"{label}: runtime field {field} is invalid")
    if not _is_git_revision(runtime.get("source_revision")):
        reasons.append(f"{label}: runtime source_revision is invalid")
    for field in (
            "runtime_overlay_sha256", "service_command_sha256",
            "service_env_sha256", "model_list_contract_sha256"):
        if not _is_sha256(runtime.get(field)):
            reasons.append(f"{label}: runtime field {field} is invalid")
    if (runtime.get("gpu_count") != 4
            or runtime.get("tensor_parallel_size") != 4
            or runtime.get("max_model_len") != 262144
            or runtime.get("served_model_name") != "llm"
            or runtime.get("fresh_service_attested") is not True
            or runtime.get("cache_trace_v4_attested") is not True):
        reasons.append(f"{label}: runtime capacity/topology is invalid")
    runtime_contract_wrapper = report.get("runtime_contract") or {}
    if (not isinstance(runtime_contract_wrapper, dict)
            or set(runtime_contract_wrapper) != {"sha256", "contract"}):
        reasons.append(f"{label}: runtime contract wrapper is invalid")
    else:
        contract = runtime_contract_wrapper["contract"]
        contract_fields = {
            "schema", "version", "source_revision", "runtime_identity",
            "runtime_overlay_sha256", "instance", "gpu_count",
            "tensor_parallel_size", "max_model_len", "model_path",
            "tokenizer_path", "served_model_name", "base_image", "command", "environment",
            "cache_trace_enabled", "optimization_label",
        }
        if not isinstance(contract, dict) or set(contract) != contract_fields:
            reasons.append(f"{label}: runtime contract fields are invalid")
        else:
            expected_runtime_contract = {
                "source_revision": runtime.get("source_revision"),
                "runtime_identity": runtime.get("runtime_identity"),
                "instance": runtime.get("instance"),
                "gpu_count": runtime.get("gpu_count"),
                "tensor_parallel_size": runtime.get("tensor_parallel_size"),
                "max_model_len": runtime.get("max_model_len"),
                "model_path": runtime.get("model_path"),
                "tokenizer_path": runtime.get("model_path"),
                "served_model_name": runtime.get("served_model_name"),
            }
            try:
                validated_contract_sha = (
                    runtime_contract.validate_runtime_contract(
                        contract,
                        expected_runtime_contract,
                        require_cache_trace=True,
                    ))
            except runtime_contract.RuntimeContractError as error:
                reasons.append(f"{label}: {error}")
                validated_contract_sha = None
            if (runtime_contract_wrapper["sha256"] != _sha256_json(contract)
                    or (validated_contract_sha is not None
                        and runtime_contract_wrapper["sha256"]
                        != validated_contract_sha)
                    or contract["schema"]
                    != "bi100-quality-runtime-contract-v1"
                    or contract["version"] != 1
                    or contract["base_image"] != BASE_IMAGE
                    or contract["cache_trace_enabled"] is not True):
                reasons.append(f"{label}: runtime contract identity is invalid")
            runtime_matches = {
                "source_revision": runtime.get("source_revision"),
                "runtime_identity": runtime.get("runtime_identity"),
                "runtime_overlay_sha256": runtime.get(
                    "runtime_overlay_sha256"),
                "instance": runtime.get("instance"),
                "gpu_count": runtime.get("gpu_count"),
                "tensor_parallel_size": runtime.get("tensor_parallel_size"),
                "max_model_len": runtime.get("max_model_len"),
                "model_path": runtime.get("model_path"),
                "tokenizer_path": runtime.get("model_path"),
                "served_model_name": runtime.get("served_model_name"),
            }
            if any(contract[field] != expected
                   for field, expected in runtime_matches.items()):
                reasons.append(f"{label}: runtime contract differs from runtime")
            command = contract["command"]
            environment = contract["environment"]
            if isinstance(environment, dict):
                policy = environment.get("BI100_GDN_CACHE_POLICY")
                if isinstance(policy, str):
                    gdn_cache_policy = policy
            if (not isinstance(command, list) or not command
                    or not all(isinstance(item, str) and item
                               for item in command)
                    or not isinstance(environment, dict)
                    or environment.get("BI100_CACHE_TRACE") != "1"
                    or not all(isinstance(key, str) and key
                               and isinstance(value, str)
                               for key, value in environment.items())
                    or not isinstance(contract["optimization_label"], str)
                    or not contract["optimization_label"]):
                reasons.append(f"{label}: runtime command/environment is invalid")
            elif (runtime.get("service_command_sha256")
                  != _sha256_json(command)
                  or runtime.get("service_env_sha256")
                  != _sha256_json(environment)):
                reasons.append(f"{label}: runtime command/environment hash differs")
            blocked_names = (
                "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")
            if isinstance(environment, dict) and any(
                    fragment in key.upper()
                    for key in environment for fragment in blocked_names):
                reasons.append(f"{label}: runtime contract has a secret field")
            serialized_contract = json.dumps(
                contract, ensure_ascii=True).lower()
            if any(marker in serialized_contract for marker in (
                    "begin openssh private key", "github_pat_", "ghp_",
                    "modelhub_access_token", "proxy-authorization")):
                reasons.append(
                    f"{label}: runtime contract has a credential marker")
    generator = report.get("generator") or {}
    if not isinstance(generator, dict) or set(generator) != {
            "runner_sha256", "exact_prompt_module_sha256",
            "transformers_version"}:
        reasons.append(f"{label}: generator fields are invalid")
    else:
        if (not _is_sha256(generator["runner_sha256"])
                or not _is_sha256(generator["exact_prompt_module_sha256"])
                or not isinstance(generator["transformers_version"], str)
                or not generator["transformers_version"]):
            reasons.append(f"{label}: generator identity is invalid")
    tokenizer = report.get("tokenizer") or {}
    expected_tokenizer_fields = {
        "tokenizer_class", "artifact_set_sha256", "chat_template_sha256",
        "files", "template_kwargs_mode", "thinking_false_prompt_sha256",
        "thinking_true_prompt_sha256", "thinking_modes_distinct",
    }
    if (not isinstance(tokenizer, dict)
            or set(tokenizer) != expected_tokenizer_fields):
        reasons.append(f"{label}: tokenizer identity fields are invalid")
    else:
        files = tokenizer["files"]
        valid_files = (
            isinstance(files, list) and bool(files)
            and all(isinstance(item, dict)
                    and set(item) == {"name", "bytes", "sha256"}
                    and isinstance(item["name"], str) and bool(item["name"])
                    and isinstance(item["bytes"], int)
                    and not isinstance(item["bytes"], bool)
                    and item["bytes"] > 0
                    and _is_sha256(item["sha256"])
                    for item in files)
        )
        if not valid_files:
            reasons.append(f"{label}: tokenizer artifact list is invalid")
        elif tokenizer["artifact_set_sha256"] != _sha256_json(files):
            reasons.append(f"{label}: tokenizer artifact aggregate differs")
        if (not isinstance(tokenizer["tokenizer_class"], str)
                or not tokenizer["tokenizer_class"]
                or not _is_sha256(tokenizer["chat_template_sha256"])
                or not _is_sha256(
                    tokenizer["thinking_false_prompt_sha256"])
                or not _is_sha256(
                    tokenizer["thinking_true_prompt_sha256"])
                or tokenizer["template_kwargs_mode"] not in ("direct", "nested")
                or tokenizer["thinking_modes_distinct"] is not True):
            reasons.append(f"{label}: tokenizer/template contract is invalid")
    selection = report.get("selection") or {}
    if (selection.get("tier") != "extended"
            or selection.get("explicit_cases") != []
            or selection.get("selected_cases") != EXPECTED_CASES):
        reasons.append(f"{label}: report is not the complete extended matrix")
    privacy = report.get("privacy") or {}
    if (privacy.get("contains_raw_requests") is not False
            or privacy.get("contains_raw_model_outputs") is not False
            or privacy.get("contains_credentials") is not False):
        reasons.append(f"{label}: privacy contract is invalid")

    cases = report.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASES:
        reasons.append(f"{label}: report must contain twelve cases")
        cases = []
    case_map = {}
    for expected, case in zip(manifest["cases"], cases):
        case_reasons = _validate_case(
            case,
            expected,
            f"{label}: case {expected['id']}",
            gdn_cache_policy,
        )
        reasons.extend(case_reasons)
        if isinstance(case, dict) and not case_reasons:
            case_map[expected["id"]] = case
    summary = report.get("summary") or {}
    recomputed_passed = sum(
        isinstance(case, dict) and case.get("status") == "pass"
        for case in cases)
    expected_summary = {
        "passed": recomputed_passed,
        "failed": len(cases) - recomputed_passed,
        "total": len(cases),
        "selected_total": EXPECTED_CASES,
        "complete": len(cases) == EXPECTED_CASES,
        "pass_rate": recomputed_passed / len(cases) if cases else 0.0,
    }
    for field, value in expected_summary.items():
        if summary.get(field) != value:
            reasons.append(f"{label}: summary {field} differs from cases")
    wall_s = summary.get("wall_s")
    if (not isinstance(wall_s, (int, float)) or isinstance(wall_s, bool)
            or not math.isfinite(wall_s) or wall_s <= 0):
        reasons.append(f"{label}: summary wall_s is invalid")
    return case_map, reasons


def _report_gdn_cache_policy(report: Any) -> str | None:
    if not isinstance(report, dict):
        return None
    wrapper = report.get("runtime_contract")
    if not isinstance(wrapper, dict):
        return None
    contract = wrapper.get("contract")
    if not isinstance(contract, dict):
        return None
    environment = contract.get("environment")
    if not isinstance(environment, dict):
        return None
    policy = environment.get("BI100_GDN_CACHE_POLICY")
    return policy if isinstance(policy, str) else None


def compare_reports(
    baseline: Any,
    candidate: Any,
    *,
    manifest: Json | None = None,
    manifest_sha: str | None = None,
    manifest_name: str | None = None,
) -> Json:
    if manifest is None or manifest_sha is None:
        manifest, manifest_sha = _load_manifest(DEFAULT_MANIFEST)
        manifest_name = DEFAULT_MANIFEST.name
    assert manifest_name is not None
    baseline_cases, reasons = _validate_report(
        baseline, "baseline", manifest, manifest_sha, manifest_name)
    candidate_cases, candidate_reasons = _validate_report(
        candidate, "candidate", manifest, manifest_sha, manifest_name)
    reasons.extend(candidate_reasons)
    comparisons = []
    baseline_policy = _report_gdn_cache_policy(baseline)
    candidate_policy = _report_gdn_cache_policy(candidate)

    if isinstance(baseline, dict) and isinstance(candidate, dict):
        baseline_runtime = baseline.get("runtime") or {}
        candidate_runtime = candidate.get("runtime") or {}
        for field in (
            "source_revision", "runtime_identity", "runtime_overlay_sha256",
            "instance", "gpu_count", "tensor_parallel_size", "model_path",
            "max_model_len", "served_model_name", "service_command_sha256"):
            if baseline_runtime.get(field) != candidate_runtime.get(field):
                reasons.append(f"runtime contract differs in {field}")
        if baseline.get("generator") != candidate.get("generator"):
            reasons.append("baseline and candidate generator identities differ")
        if baseline.get("tokenizer") != candidate.get("tokenizer"):
            reasons.append("baseline and candidate tokenizer identities differ")
        baseline_wrapper = baseline.get("runtime_contract") or {}
        candidate_wrapper = candidate.get("runtime_contract") or {}
        baseline_contract = (
            baseline_wrapper.get("contract") or {}
            if isinstance(baseline_wrapper, dict) else {})
        candidate_contract = (
            candidate_wrapper.get("contract") or {}
            if isinstance(candidate_wrapper, dict) else {})
        if isinstance(baseline_contract, dict) and isinstance(
                candidate_contract, dict):
            for field in (
                    "source_revision", "runtime_identity",
                    "runtime_overlay_sha256", "instance", "base_image",
                    "command", "gpu_count",
                    "tensor_parallel_size", "max_model_len", "model_path",
                    "tokenizer_path", "served_model_name",
                    "cache_trace_enabled"):
                if baseline_contract.get(field) != candidate_contract.get(field):
                    reasons.append(f"A/B runtime contract differs in {field}")
            baseline_env = baseline_contract.get("environment") or {}
            candidate_env = candidate_contract.get("environment") or {}
            if isinstance(baseline_env, dict) and isinstance(candidate_env, dict):
                changed_env = {
                    key for key in set(baseline_env) | set(candidate_env)
                    if baseline_env.get(key) != candidate_env.get(key)
                }
                disallowed_env = changed_env - ALLOWED_AB_ENV_DIFFERENCES
                if disallowed_env:
                    reasons.append(
                        "A/B changed disallowed runtime environment values: "
                        + ", ".join(sorted(disallowed_env)))

    for expected in manifest["cases"]:
        case_id = expected["id"]
        if case_id not in baseline_cases or case_id not in candidate_cases:
            continue
        base_observation = baseline_cases[case_id]["observation"]
        cand_observation = candidate_cases[case_id]["observation"]
        base_requests = base_observation["requests"]
        cand_requests = cand_observation["requests"]
        case_reasons = []
        for index, (base_request, cand_request) in enumerate(
                zip(base_requests, cand_requests), 1):
            for field in (
                    "status", "model", "local_prompt_tokens", "prompt_tokens",
                    "finish_reason", "request_contract_sha256",
                    "token_accounting"):
                if base_request.get(field) != cand_request.get(field):
                    case_reasons.append(f"request {index} {field} differs")
            if case_id in EXACT_BASELINE_IDS:
                for field in (
                        "semantic_output_sha256", "completion_tokens"):
                    if base_request.get(field) != cand_request.get(field):
                        case_reasons.append(f"request {index} {field} differs")
            elif case_id in NEXT_TOKEN_IDS:
                if base_request.get(
                        "first_generated_token_sha256") != cand_request.get(
                            "first_generated_token_sha256"):
                    case_reasons.append(
                        f"request {index} first generated token differs")
        if (case_id == "235k_partial_branch"
                and baseline_policy != candidate_policy):
            if any(
                    base_observation["facts"].get(fact)
                    != cand_observation["facts"].get(fact)
                    for fact in TRUE_FACTS[case_id]):
                case_reasons.append("independent capability facts differ")
        elif case_id in SEMANTIC_IDS | NEXT_TOKEN_IDS:
            if base_observation["facts"] != cand_observation["facts"]:
                case_reasons.append("independent capability facts differ")
        comparisons.append({
            "ordinal": expected["ordinal"],
            "id": case_id,
            "mode": (
                "exact" if case_id in EXACT_BASELINE_IDS
                else "next_token" if case_id in NEXT_TOKEN_IDS
                else "semantic"),
            "qualified": not case_reasons,
            "reasons": case_reasons,
        })
        reasons.extend(f"{case_id}: {reason}" for reason in case_reasons)

    qualified = not reasons and len(comparisons) == EXPECTED_CASES
    return {
        "schema": COMPARISON_SCHEMA,
        "version": 5,
        "qualified": qualified,
        "long_context_quality_non_regression_authorized": qualified,
        "overall_promotion_authorized": False,
        "reasons": reasons,
        "summary": {
            "compared_cases": len(comparisons),
            "qualified_cases": sum(row["qualified"] for row in comparisons),
            "failed_cases": sum(not row["qualified"] for row in comparisons),
        },
        "cases": comparisons,
        "privacy": {
            "contains_raw_requests": False,
            "contains_raw_model_outputs": False,
            "contains_credentials": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest, manifest_sha = _load_manifest(args.manifest)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    result = compare_reports(
        baseline,
        candidate,
        manifest=manifest,
        manifest_sha=manifest_sha,
        manifest_name=args.manifest.name,
    )
    result["inputs"] = {
        "manifest_sha256": manifest_sha,
        "baseline_file_sha256": hashlib.sha256(
            args.baseline.read_bytes()).hexdigest(),
        "candidate_file_sha256": hashlib.sha256(
            args.candidate.read_bytes()).hexdigest(),
        "baseline_revision": (baseline.get("runtime") or {}).get(
            "source_revision"),
        "candidate_revision": (candidate.get("runtime") or {}).get(
            "source_revision"),
    }
    _atomic_write(args.out, result)
    print(json.dumps({
        "qualified": result["qualified"],
        "reasons": result["reasons"],
        "summary": result["summary"],
        "out": str(args.out),
    }, indent=2, sort_keys=True))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
