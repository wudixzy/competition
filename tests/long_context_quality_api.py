#!/usr/bin/env python3
"""Run the deterministic BI100 long-context capability matrix."""

from __future__ import annotations

import argparse
import base64
import binascii
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable

import exact_chat_prompt as exact_prompt
import quality_gate_api as quality
import quality_runtime_contract as runtime_contract
import validate_quality_data_manifests as manifest_validator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "quality/long_context_matrix.v6.json"
EXPECTED_MANIFEST_SHA256 = (
    "787d603818e5238b8fd45332d30c2991a7b0873d0012e2f0caad0a5c50b40115"
)
SCHEMA = "bi100-long-context-quality-result-v6"
BASE_IMAGE = runtime_contract.BASE_IMAGE
TIER_RANK = {"quick": 0, "full": 1, "extended": 2}
Json = dict[str, Any]
REASONING_DIAGNOSTIC_KEYS = frozenset({
    "content_arithmetic_present",
    "content_contains_expected",
    "content_exact_expected",
    "content_expected_prefix",
    "content_expected_single_occurrence",
    "content_expected_suffix",
    "content_markers_in_order",
    "content_markers_present",
    "reasoning_arithmetic_present",
    "reasoning_contains_expected",
    "reasoning_markers_in_order",
    "reasoning_markers_present",
})


class MatrixFailure(RuntimeError):
    pass


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise MatrixFailure(reason)


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


def _load_manifest(path: Path) -> tuple[Json, str]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    require(digest == EXPECTED_MANIFEST_SHA256,
            "long-context matrix file identity is invalid")
    manifest = json.loads(payload)
    reasons = manifest_validator.validate_matrix(manifest)
    require(not reasons, "long-context matrix contract is invalid")
    return manifest, digest


def _load_runtime_contract(path: Path, args: Any) -> tuple[Json, str]:
    expected = {
        "source_revision": args.source_revision,
        "runtime_identity": args.runtime_identity,
        "instance": args.instance,
        "gpu_count": args.gpu_count,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "model_path": args.model_path,
        "tokenizer_path": args.model_path,
        "served_model_name": args.served_model_name,
    }
    try:
        return runtime_contract.load_runtime_contract(
            path, expected, require_cache_trace=True)
    except runtime_contract.RuntimeContractError as error:
        raise MatrixFailure(str(error)) from error


def _selected_cases(manifest: Json, tier: str, requested: list[str]) -> list[Json]:
    cases = manifest["cases"]
    if requested:
        require(len(requested) == len(set(requested)),
                "explicit case ids must be unique")
        known = {case["id"] for case in cases}
        require(set(requested) <= known, "unknown matrix case requested")
        return [case for case in cases if case["id"] in set(requested)]
    rank = TIER_RANK[tier]
    return [case for case in cases if TIER_RANK[case["tier"]] <= rank]


Recipe = exact_prompt.Recipe


def _fit_recipe(
    tokenizer: Any,
    target_tokens: int,
    recipe: Recipe,
    *,
    namespace: str,
    thinking: bool = False,
    template_kwargs_mode: str = "direct",
) -> tuple[list[Json], list[Json] | None, Json]:
    try:
        return exact_prompt.fit_exact_chat_prompt(
            tokenizer,
            target_tokens,
            recipe,
            seed=20260724,
            namespace=namespace,
            thinking=thinking,
            template_kwargs_mode=template_kwargs_mode,
        )
    except exact_prompt.PromptConstructionError as error:
        raise MatrixFailure(str(error)) from error


def _filler(value: str) -> str:
    midpoint = len(value) // 2
    return (value[:midpoint] + "\nMIDDLE-MARKER-552\n"
            + value[midpoint:])


def _recall_recipe(case_id: str, expected: str) -> Recipe:
    def recipe(filler: str) -> tuple[list[Json], None]:
        content = (
            f"BEGIN-MARKER-731\ncase={case_id}\n"
            + _filler(filler)
            + "\nEND-MARKER-947\n"
            + "Reply with exactly: " + expected
        )
        return [{"role": "user", "content": content}], None
    return recipe


def _branch_recipe(case_id: str, branch: str) -> Recipe:
    expected = f"SHARED-731|BRANCH-{branch}-947"

    def recipe(filler: str) -> tuple[list[Json], None]:
        content = (
            f"SHARED-731\ncase={case_id}\n"
            + _filler(filler)
            + f"\nBRANCH-{branch}-947\nReply with exactly: {expected}"
        )
        return [{"role": "user", "content": content}], None
    return recipe


def _function_tool(name: str, argument: str = "key") -> Json:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Deterministic quality function {name}.",
            "parameters": {
                "type": "object",
                "properties": {argument: {"type": "string"}},
                "required": [argument],
                "additionalProperties": False,
            },
        },
    }


def _large_tools(
    target_name: str,
    namespace: str = "matrix",
    *,
    target_arguments: tuple[str, ...] = ("key", "ordinal"),
) -> list[Json]:
    argument_properties = {
        "key": {"type": "string"},
        "ordinal": {"type": "integer"},
    }
    require(
        bool(target_arguments)
        and len(target_arguments) == len(set(target_arguments))
        and set(target_arguments) <= set(argument_properties),
        "target tool arguments are invalid",
    )
    tools = []
    for index in range(92):
        if index == 0:
            tools.append(_function_tool("load_workspace", "workspace"))
            continue
        elif index == 73:
            name = target_name
        else:
            name = f"auxiliary_{namespace}_{index:03d}"
        is_target = name == target_name
        properties = (
            {argument: argument_properties[argument]
             for argument in target_arguments}
            if is_target else argument_properties
        )
        required = list(target_arguments) if is_target else ["key"]
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": f"Deterministic quality tool {index:03d}.",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        })
    return tools


def _tool_recipe(case_id: str, target_name: str, marker: str) -> Recipe:
    tools = _large_tools(
        target_name, case_id, target_arguments=("key",))

    def recipe(filler: str) -> tuple[list[Json], list[Json]]:
        content = (
            f"case={case_id}\n" + _filler(filler)
            + f"\nCall {target_name} with key={marker}."
        )
        return [{"role": "user", "content": content}], tools
    return recipe


def _tool_result_recipe(case_id: str, marker: str) -> Recipe:
    def recipe(filler: str) -> tuple[list[Json], list[Json]]:
        tool_content = (
            f"case={case_id}\nBEGIN-TOOL-RESULT\n"
            + _filler(filler)
            + f"\nRESULT-MARKER={marker}\nEND-TOOL-RESULT"
        )
        messages = [
            {"role": "user", "content": "Fetch the private test record."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call-quality-fixed",
                    "type": "function",
                    "function": {
                        "name": "fetch_record",
                        "arguments": '{"record":"quality"}',
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call-quality-fixed",
                "name": "fetch_record",
                "content": tool_content,
            },
            {
                "role": "user",
                "content": f"Reply with exactly the RESULT-MARKER value {marker}.",
            },
        ]
        return messages, [_function_tool("fetch_record", "record")]
    return recipe


def _multiturn_recipe(case_id: str, session: str, marker: str) -> Recipe:
    def recipe(filler: str) -> tuple[list[Json], None]:
        messages = [
            {"role": "user", "content": (
                f"session={session}; case={case_id}; remember {marker}.\n"
                + _filler(filler))},
            {"role": "assistant", "content": "Stored."},
            {"role": "user", "content": (
                f"Return only the marker for session {session}.")},
        ]
        return messages, None
    return recipe


def _generated_asset_contract() -> Json:
    return {
        "red_png_data_url_sha256": hashlib.sha256(
            quality._solid_png_data_url((224, 0, 0)).encode("ascii")
        ).hexdigest(),
        "blue_png_data_url_sha256": hashlib.sha256(
            quality._solid_png_data_url((0, 0, 224)).encode("ascii")
        ).hexdigest(),
        "large_tools_65k_sha256": quality._sha256_json(
            _large_tools(
                "lookup_quality_marker", "65k_multiturn_large_tools",
                target_arguments=("key",))),
        "large_tools_235k_sha256": quality._sha256_json(
            _large_tools(
                "report_agent_marker", "235k_agent_large_output_budget",
                target_arguments=("key", "ordinal"))),
        "fetch_record_tool_sha256": quality._sha256_json(
            [_function_tool("fetch_record", "record")]),
    }


class Context:
    def __init__(
        self,
        client: quality.Client,
        tokenizer: Any,
        timeout_s: float,
        *,
        served_model_name: str,
        template_kwargs_mode: str,
        cache_trace_path: Path | None = None,
        gdn_cache_policy: str = "fine32",
    ):
        require(
            gdn_cache_policy in {"fine32", "admission64"},
            "long-context GDN cache policy is invalid",
        )
        self.client = client
        self.tokenizer = tokenizer
        self.timeout_s = timeout_s
        self.served_model_name = served_model_name
        self.template_kwargs_mode = template_kwargs_mode
        self.cache_trace_path = cache_trace_path
        self.gdn_cache_policy = gdn_cache_policy
        self._case_requests: list[Json] = []
        self._case_failure_facts: Json = {}

    def begin_case(self) -> None:
        self._case_requests = []
        self._case_failure_facts = {}

    def record_request(self, summary: Json) -> None:
        self._case_requests.append(dict(summary))

    def record_failure_facts(self, facts: Json) -> None:
        require(
            set(facts) <= REASONING_DIAGNOSTIC_KEYS
            and all(isinstance(value, bool) for value in facts.values()),
            "failure diagnostic facts are invalid",
        )
        self._case_failure_facts.update(facts)

    def failure_observation(self) -> Json:
        return {
            "requests": [dict(summary) for summary in self._case_requests],
            "construction": [],
            "facts": {
                "privacy_safe_requests_captured_before_failure": len(
                    self._case_requests),
                **self._case_failure_facts,
            },
        }


def _payload(
    case: Json,
    messages: list[Json],
    *,
    served_model_name: str,
    tools: list[Json] | None = None,
    thinking: bool = False,
    tool_name: str | None = None,
) -> Json:
    payload: Json = {
        "model": served_model_name,
        "messages": messages,
        "max_tokens": case["max_tokens"],
        "thinking": thinking,
        "temperature": 0,
        "seed": 20260724,
    }
    if case["min_completion_tokens"] > 1:
        payload["min_tokens"] = case["min_completion_tokens"]
    if "baseline_next_token" in case["equivalence"]:
        payload["logprobs"] = True
        payload["top_logprobs"] = 0
    if tools is not None:
        payload["tools"] = tools
    if tool_name is not None:
        payload["tool_choice"] = {
            "type": "function", "function": {"name": tool_name},
        }
    return payload


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _tool_call_structure(message: Json) -> Json:
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return {
            "container_type": type(calls).__name__,
            "count": None,
            "calls": [],
        }
    structures = []
    for call in calls:
        function = call.get("function") if isinstance(call, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        arguments = (
            function.get("arguments") if isinstance(function, dict) else None)
        structure: Json = {
            "call_type": type(call).__name__,
            "function_type": type(function).__name__,
            "name_sha256": (
                hashlib.sha256(name.encode("utf-8")).hexdigest()
                if isinstance(name, str) else None),
            "arguments_type": type(arguments).__name__,
        }
        if isinstance(arguments, str):
            stripped = arguments.strip()
            structure.update({
                "arguments_length": len(arguments),
                "arguments_sha256": hashlib.sha256(
                    arguments.encode("utf-8")).hexdigest(),
                "starts_object": stripped.startswith("{"),
                "ends_object": stripped.endswith("}"),
                "contains_tool_call_tag": "<tool_call>" in arguments,
                "contains_function_prefix": "<function=" in arguments,
                "contains_code_fence": "```" in arguments,
            })
            try:
                decoded = json.loads(arguments)
            except (json.JSONDecodeError, TypeError, ValueError):
                structure["json_type"] = "invalid"
            else:
                structure["json_type"] = type(decoded).__name__
        structures.append(structure)
    return {
        "container_type": "list",
        "count": len(calls),
        "calls": structures,
    }


def _post(
    context: Context,
    payload: Json,
    target_tokens: int,
    *,
    token_accounting: str = "server_exact",
) -> tuple[Json, Json]:
    request_contract_sha256 = quality._sha256_json(payload)
    started = time.perf_counter()
    result = context.client.post(payload, timeout=context.timeout_s)
    elapsed = time.perf_counter() - started
    data = quality._expect_200(result)
    require(_finite(data), "matrix response contains NaN or Inf")
    require(data.get("model") == context.served_model_name,
            "response model differs from requested served model")
    require(len(data["choices"]) == 1,
            "matrix response must contain exactly one choice")
    usage = quality._usage(data)
    if token_accounting == "server_exact":
        require(usage["prompt_tokens"] == target_tokens,
                "server prompt token count differs from matrix target")
    elif token_accounting == "local_template_plus_vision":
        require(usage["prompt_tokens"] >= target_tokens,
                "server multimodal prompt is shorter than local template")
    else:
        raise MatrixFailure("unknown prompt token accounting mode")
    require(usage["prompt_tokens"] + payload["max_tokens"] <= 262144,
            "server prompt plus requested output exceeds max_model_len")
    require(usage["completion_tokens"] > 0,
            "matrix response completion is empty")
    choice = data["choices"][0]
    require(choice.get("finish_reason") in quality.ALLOWED_FINISH_REASONS,
            "matrix response has no terminal finish_reason")
    message = quality._message(data)
    content = message.get("content")
    reasoning = quality._reasoning(message)
    first_token_sha256 = None
    logprobs = choice.get("logprobs") or {}
    logprob_content = logprobs.get("content") or []
    if logprob_content:
        token = logprob_content[0].get("token")
        if isinstance(token, str):
            first_token_sha256 = hashlib.sha256(token.encode("utf-8")).hexdigest()
    if payload.get("logprobs"):
        require(first_token_sha256 is not None,
                "next-token gate received no first-token logprob")
    summary: Json = {
        "status": result[0],
        "model": data["model"],
        "local_prompt_tokens": target_tokens,
        "prompt_tokens": usage["prompt_tokens"],
        "cached_tokens": usage["cached_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": data["usage"]["total_tokens"],
        "finish_reason": choice.get("finish_reason"),
        "content_sha256": quality._sha256_json(content),
        "content_length": len(content) if isinstance(content, str) else 0,
        "reasoning_sha256": quality._sha256_json(reasoning),
        "reasoning_length": len(reasoning),
        "tool_call_structure": _tool_call_structure(message),
        "first_generated_token_sha256": first_token_sha256,
        "request_contract_sha256": request_contract_sha256,
        "token_accounting": token_accounting,
        "protocol_validated": False,
        "elapsed_s": round(elapsed, 6),
    }
    try:
        normalized = quality._normalized_response(data)
        normalized_tools = quality._normalized_tool_calls(message)
    except Exception:
        context.record_request(summary)
        raise
    summary.update({
        "semantic_output_sha256": quality._sha256_json(normalized),
        "tool_calls_sha256": quality._sha256_json(normalized_tools),
        "protocol_validated": True,
    })
    context.record_request(summary)
    return data, summary


def _content(data: Json) -> str:
    return quality._content(data).strip()


def _require_single_tool_call(
    data: Json,
    expected_name: str,
    expected_arguments: Json,
    *,
    allow_content: bool = False,
) -> None:
    choice = data["choices"][0]
    require(choice.get("finish_reason") == "tool_calls",
            "tool response finish_reason must be tool_calls")
    message = quality._message(data)
    content = message.get("content")
    require(content is None or isinstance(content, str),
            "tool response content must be a string or null")
    if not allow_content:
        require(content is None or not content.strip(),
                "forced tool response contains unexpected content")
    raw_calls = message.get("tool_calls")
    require(isinstance(raw_calls, list) and len(raw_calls) == 1,
            "forced tool response must contain exactly one call")
    calls = quality._normalized_tool_calls(message)
    require(len(calls) == 1 and calls[0].get("name") == expected_name,
            "forced tool response name differs")
    actual_arguments = calls[0].get("arguments")
    require(isinstance(actual_arguments, dict),
            "forced tool arguments must decode to an object")
    expected_keys = sorted(expected_arguments)
    actual_keys = sorted(actual_arguments)
    require(actual_keys == expected_keys,
            "forced tool argument keys differ: "
            f"expected={','.join(expected_keys)};"
            f"actual={','.join(actual_keys)}")
    mismatched = sorted(
        key for key in expected_keys
        if actual_arguments.get(key) != expected_arguments[key])
    require(not mismatched,
            "forced tool argument values differ for fields: "
            + ",".join(mismatched))


def _simple_recall(context: Context, case: Json) -> Json:
    expected = "BEGIN-MARKER-731|MIDDLE-MARKER-552|END-MARKER-947"
    messages, _, construction = _fit_recipe(
        context.tokenizer,
        case["target_prompt_tokens"],
        _recall_recipe(case["id"], expected),
        namespace=case["id"],
        template_kwargs_mode=context.template_kwargs_mode,
    )
    payload = _payload(
        case, messages, served_model_name=context.served_model_name)
    first_data, first = _post(
        context, payload, case["target_prompt_tokens"])
    require(expected in _content(first_data), "long-context markers were not recalled")
    requests = [first]
    facts: Json = {"marker_rule_passed": True}
    if case["cache_scenario"] == "cold_warm_identical":
        second_data, second = _post(
            context, payload, case["target_prompt_tokens"])
        require(quality._normalized_response(first_data)
                == quality._normalized_response(second_data),
                "long-context cold/warm output differs")
        require(first["cached_tokens"] == 0 and second["cached_tokens"] > 0,
                "long-context cold/warm cache accounting differs")
        requests.append(second)
        facts["cold_warm_exact"] = True
    constructions = [construction]
    if case["id"] == "near_262k_capacity":
        boundary_target = case["target_prompt_tokens"] - 1
        boundary_messages, _, boundary_construction = _fit_recipe(
            context.tokenizer,
            boundary_target,
            _recall_recipe(case["id"] + "-minus-one", expected),
            namespace=case["id"] + "-minus-one",
            template_kwargs_mode=context.template_kwargs_mode,
        )
        boundary_payload = _payload(
            case,
            boundary_messages,
            served_model_name=context.served_model_name,
        )
        boundary_data, boundary = _post(
            context, boundary_payload, boundary_target)
        require(expected in _content(boundary_data),
                "262144-minus-one boundary marker rule failed")
        requests.append(boundary)
        constructions.append(boundary_construction)
        facts["exact_capacity_boundary_passed"] = True
        facts["minus_one_capacity_boundary_passed"] = True
    return {
        "requests": requests,
        "construction": constructions,
        "facts": facts,
    }


def _partial_branch(context: Context, case: Json) -> Json:
    target = case["target_prompt_tokens"]
    require(context.cache_trace_path is not None,
            "partial-branch quality gate requires a cache trace file")
    require(context.cache_trace_path.is_file(),
            "partial-branch cache trace file is unavailable")
    trace_offset = context.cache_trace_path.stat().st_size
    messages_a, _, construction_a = _fit_recipe(
        context.tokenizer, target, _branch_recipe(case["id"], "A"),
        namespace=case["id"],
        template_kwargs_mode=context.template_kwargs_mode)
    messages_b, _, construction_b = _fit_recipe(
        context.tokenizer, target, _branch_recipe(case["id"], "B"),
        namespace=case["id"],
        template_kwargs_mode=context.template_kwargs_mode)
    messages_c, _, construction_c = _fit_recipe(
        context.tokenizer, target, _branch_recipe(case["id"], "C"),
        namespace=case["id"],
        template_kwargs_mode=context.template_kwargs_mode)
    payload_a = _payload(
        case, messages_a, served_model_name=context.served_model_name)
    payload_b = _payload(
        case, messages_b, served_model_name=context.served_model_name)
    payload_c = _payload(
        case, messages_c, served_model_name=context.served_model_name)
    a_cold_data, a_cold = _post(context, payload_a, target)
    a_warm_data, a_warm = _post(context, payload_a, target)
    b_data, branch_b = _post(context, payload_b, target)
    c_data, branch_c = _post(context, payload_c, target)
    require(_content(a_cold_data) == "SHARED-731|BRANCH-A-947",
            "branch A marker rule failed")
    require(_content(b_data) == "SHARED-731|BRANCH-B-947",
            "branch B marker rule failed")
    require(_content(c_data) == "SHARED-731|BRANCH-C-947",
            "branch C marker rule failed")
    require(quality._normalized_response(a_cold_data)
            == quality._normalized_response(a_warm_data),
            "branch A cold/warm output differs")
    require(a_cold["cached_tokens"] == 0 and a_warm["cached_tokens"] > 0,
            "branch A cold/warm cache accounting differs")
    requests = [a_cold, a_warm, branch_b, branch_c]
    if case["id"] == "235k_partial_branch":
        b_repeat_data, b_repeat = _post(context, payload_b, target)
        require(quality._normalized_response(b_data)
                == quality._normalized_response(b_repeat_data),
                "235K branch warm repeat differs")
        require(b_repeat["cached_tokens"] > 0,
                "235K branch warm repeat did not hit cache")
        requests.append(b_repeat)
    records = _cache_trace_records(
        context.cache_trace_path, trace_offset, len(requests))
    policy_facts = _partial_branch_trace_facts(
        records, requests, context.gdn_cache_policy)
    return {
        "requests": requests,
        "construction": [construction_a, construction_b, construction_c],
        "facts": {
            "branch_markers_correct": True,
            "cold_warm_exact": True,
            "strict_partial_hit": True,
            **policy_facts,
        },
    }


def _cache_trace_records(path: Path, offset: int, expected: int) -> list[Json]:
    marker = "[BI100_CACHE_TRACE] "
    deadline = time.monotonic() + 5
    records: list[Json] = []
    while time.monotonic() < deadline:
        require(path.is_file(), "cache trace file disappeared")
        with path.open("rb") as stream:
            stream.seek(offset)
            tail = stream.read().decode("utf-8", "replace")
        records = []
        for line in tail.splitlines():
            position = line.find(marker)
            if position < 0:
                continue
            try:
                value = json.loads(line[position + len(marker):])
            except json.JSONDecodeError as error:
                raise MatrixFailure("cache trace contains invalid JSON") from error
            require(isinstance(value, dict), "cache trace record is not an object")
            records.append(value)
        if len(records) >= expected:
            break
        time.sleep(0.1)
    require(len(records) == expected,
            "cache gate trace record count differs")
    return records


def _partial_branch_trace_facts(
    records: list[Json],
    requests: list[Json],
    gdn_cache_policy: str,
) -> Json:
    require(
        gdn_cache_policy in {"fine32", "admission64"},
        "partial-branch GDN cache policy is invalid",
    )
    require(
        len(records) == len(requests) and len(records) in {4, 5},
        "partial-branch cache trace sequence length differs",
    )
    trace_sessions = {
        record.get("trace_session_sha256") for record in records
        if isinstance(record, dict)
    }
    require(
        len(trace_sessions) == 1
        and all(isinstance(value, str) and value for value in trace_sessions),
        "partial-branch cache traces do not share one service session",
    )
    for index, (record, request) in enumerate(zip(records, requests), 1):
        require(record.get("version") == 4,
                f"partial-branch trace {index} version differs")
        require(record.get("gdn_policy") == gdn_cache_policy,
                f"partial-branch trace {index} GDN policy differs")
        for field in (
                "initial_raw_kv_contiguous_hit_blocks",
                "effective_gdn_hit_blocks",
                "observed_effective_cached_tokens"):
            value = record.get(field)
            require(
                isinstance(value, int) and not isinstance(value, bool)
                and value >= 0,
                f"partial-branch trace {index} {field} is invalid",
            )
        require(
            record["observed_effective_cached_tokens"]
            == request["cached_tokens"],
            f"partial-branch trace {index} and API cached_tokens differ",
        )
        require(
            record.get("prompt_tokens") == request["prompt_tokens"],
            f"partial-branch trace {index} prompt token count differs",
        )

    branch_b_record = records[2]
    branch_c_record = records[3]
    branch_b = requests[2]
    branch_c = requests[3]
    require(
        0 < branch_c["cached_tokens"] < branch_c["prompt_tokens"],
        "subsequent sibling did not report a strict partial hit",
    )
    require(
        branch_c_record["effective_gdn_hit_blocks"] > 0
        and isinstance(
            branch_c_record.get("gdn_restore_digest_base64"), str)
        and bool(branch_c_record["gdn_restore_digest_base64"]),
        "subsequent sibling did not restore a GDN state",
    )

    if gdn_cache_policy == "fine32":
        require(
            0 < branch_b["cached_tokens"] < branch_b["prompt_tokens"],
            "fine32 first sibling did not report a strict partial hit",
        )
        require(
            branch_b_record["effective_gdn_hit_blocks"] > 0
            and isinstance(
                branch_b_record.get("gdn_restore_digest_base64"), str)
            and bool(branch_b_record["gdn_restore_digest_base64"]),
            "fine32 first sibling did not restore a GDN state",
        )
        return {
            "cache_trace_session_attested": True,
            "first_sibling_strict_partial_hit": True,
            "subsequent_sibling_strict_partial_hit": True,
            "subsequent_sibling_restored": True,
        }

    require(
        branch_b_record["initial_raw_kv_contiguous_hit_blocks"] > 0,
        "admission64 first sibling did not observe a raw KV prefix",
    )
    require(
        branch_b["cached_tokens"] == 0
        and branch_b_record["effective_gdn_hit_blocks"] == 0,
        "admission64 first sibling was incorrectly reported as reusable",
    )
    admissions = branch_b_record.get("gdn_admissions")
    require(
        isinstance(admissions, list),
        "admission64 first-sibling admission evidence is invalid",
    )
    repeated_branch = [
        action for action in admissions
        if isinstance(action, dict)
        and action.get("reason") == "repeated_branch"
        and isinstance(action.get("block_count"), int)
        and not isinstance(action["block_count"], bool)
        and action["block_count"] > 0
    ]
    require(
        bool(repeated_branch),
        "admission64 first sibling did not admit the repeated branch",
    )
    return {
        "cache_trace_session_attested": True,
        "first_sibling_effective_miss": True,
        "repeated_branch_admitted": True,
        "subsequent_sibling_strict_partial_hit": True,
        "subsequent_sibling_restored": True,
    }


def _prompt_trace_hashes(record: Json, expected_cached_tokens: int) -> list[bytes]:
    require(record.get("version") == 4,
            "multimodal cache trace version differs")
    require(record.get("hash_encoding") == "sha256_base64",
            "multimodal cache trace hash encoding differs")
    block_size = record.get("block_size")
    prompt_tokens = record.get("prompt_tokens")
    require(isinstance(block_size, int) and block_size > 0,
            "multimodal cache trace block size is invalid")
    require(isinstance(prompt_tokens, int) and prompt_tokens > 0,
            "multimodal cache trace prompt token count is invalid")
    require(record.get("observed_effective_cached_tokens")
            == expected_cached_tokens,
            "cache trace and API cached_tokens differ")
    encoded = record.get("block_hashes")
    require(isinstance(encoded, str), "cache trace block hashes are missing")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise MatrixFailure("cache trace block hashes are invalid") from error
    require(len(decoded) % 32 == 0,
            "cache trace block hash bytes are truncated")
    hashes = [decoded[index:index + 32]
              for index in range(0, len(decoded), 32)]
    full_prompt_blocks = prompt_tokens // block_size
    require(full_prompt_blocks > 0 and len(hashes) >= full_prompt_blocks,
            "cache trace does not cover complete prompt blocks")
    return hashes[:full_prompt_blocks]


def _multimodal(context: Context, case: Json) -> Json:
    target = case["target_prompt_tokens"]
    require(context.cache_trace_path is not None,
            "multimodal quality gate requires a cache trace file")
    require(context.cache_trace_path.is_file(),
            "multimodal cache trace file is unavailable")
    trace_offset = context.cache_trace_path.stat().st_size

    def recipe(rgb: tuple[int, int, int]) -> Recipe:
        def build(filler: str) -> tuple[list[Json], None]:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {
                        "url": quality._solid_png_data_url(rgb)}},
                    {"type": "text", "text": (
                        f"case={case['id']}\n" + _filler(filler)
                        + "\nName the image center color in Chinese.")},
                ],
            }]
            return messages, None
        return build

    red_messages, _, red_construction = _fit_recipe(
        context.tokenizer, target, recipe((224, 0, 0)),
        namespace=case["id"],
        template_kwargs_mode=context.template_kwargs_mode)
    blue_messages, _, blue_construction = _fit_recipe(
        context.tokenizer, target, recipe((0, 0, 224)),
        namespace=case["id"],
        template_kwargs_mode=context.template_kwargs_mode)
    red_payload = _payload(
        case, red_messages, served_model_name=context.served_model_name)
    blue_payload = _payload(
        case, blue_messages, served_model_name=context.served_model_name)
    red_cold_data, red_cold = _post(
        context, red_payload, target,
        token_accounting="local_template_plus_vision")
    red_warm_data, red_warm = _post(
        context, red_payload, target,
        token_accounting="local_template_plus_vision")
    blue_data, blue = _post(
        context, blue_payload, target,
        token_accounting="local_template_plus_vision")
    require("红" in _content(red_cold_data) and "蓝" in _content(blue_data),
            "multimodal color rule failed")
    require(quality._normalized_response(red_cold_data)
            == quality._normalized_response(red_warm_data),
            "same-image cold/warm output differs")
    require(red_cold["cached_tokens"] == 0 and red_warm["cached_tokens"] > 0,
            "same-image cache accounting differs")
    require(blue["cached_tokens"] == 0,
            "different-image request reused an image-dependent prefix")
    red_image_sha = hashlib.sha256(
        quality._solid_png_data_url((224, 0, 0)).encode("ascii")).hexdigest()
    blue_image_sha = hashlib.sha256(
        quality._solid_png_data_url((0, 0, 224)).encode("ascii")).hexdigest()
    require(red_image_sha != blue_image_sha,
            "multimodal fixtures do not have distinct identities")
    trace_records = _cache_trace_records(
        context.cache_trace_path, trace_offset, 3)
    trace_sessions = {
        record.get("trace_session_sha256") for record in trace_records}
    require(len(trace_sessions) == 1
            and all(isinstance(value, str) and value
                    for value in trace_sessions),
            "multimodal cache traces do not share one service session")
    red_cold_hashes = _prompt_trace_hashes(
        trace_records[0], red_cold["cached_tokens"])
    red_warm_hashes = _prompt_trace_hashes(
        trace_records[1], red_warm["cached_tokens"])
    blue_hashes = _prompt_trace_hashes(
        trace_records[2], blue["cached_tokens"])
    require(red_cold_hashes == red_warm_hashes,
            "same image produced different logical prompt hashes")
    require(red_cold_hashes[0] != blue_hashes[0],
            "different images share the first logical prompt hash")
    trace_proof_sha = quality._sha256_json(trace_records)
    return {
        "requests": [red_cold, red_warm, blue],
        "construction": [red_construction, blue_construction],
        "facts": {
            "red_blue_rules_passed": True,
            "same_image_cold_warm_exact": True,
            "different_image_isolated": True,
            "image_identity_digests_distinct": True,
            "cache_trace_identity_passed": True,
            "cache_trace_records_sha256": trace_proof_sha,
            "cache_trace_version": 4,
            "red_image_sha256": red_image_sha,
            "blue_image_sha256": blue_image_sha,
        },
    }


def _large_tool_call(context: Context, case: Json) -> Json:
    target_name = "lookup_quality_marker"
    marker = "TOOLS-731"
    messages, tools, construction = _fit_recipe(
        context.tokenizer,
        case["target_prompt_tokens"],
        _tool_recipe(case["id"], target_name, marker),
        namespace=case["id"],
        thinking=False,
        template_kwargs_mode=context.template_kwargs_mode,
    )
    payload = _payload(
        case, messages, served_model_name=context.served_model_name,
        tools=tools, thinking=False, tool_name=target_name)
    first_data, first = _post(
        context, payload, case["target_prompt_tokens"])
    second_data, second = _post(
        context, payload, case["target_prompt_tokens"])
    _require_single_tool_call(
        first_data, target_name, {"key": marker})
    _require_single_tool_call(
        second_data, target_name, {"key": marker})
    require(quality._normalized_response(first_data)
            == quality._normalized_response(second_data),
            "large tools cold/warm output differs")
    require(first["cached_tokens"] == 0 and second["cached_tokens"] > 0,
            "large tools cold/warm cache accounting differs")
    return {
        "requests": [first, second],
        "construction": [construction],
        "facts": {
            "tool_count": len(tools or []),
            "tool_call_rule_passed": True,
            "cold_warm_exact": True,
        },
    }


def _long_tool_result(context: Context, case: Json) -> Json:
    marker = "TOOL-RESULT-947"
    messages, tools, construction = _fit_recipe(
        context.tokenizer,
        case["target_prompt_tokens"],
        _tool_result_recipe(case["id"], marker),
        namespace=case["id"],
        template_kwargs_mode=context.template_kwargs_mode,
    )
    payload = _payload(
        case, messages, served_model_name=context.served_model_name,
        tools=tools)
    first_data, first = _post(
        context, payload, case["target_prompt_tokens"])
    second_data, second = _post(
        context, payload, case["target_prompt_tokens"])
    require(_content(first_data) == marker,
            "long tool-result marker rule failed")
    require(quality._normalized_response(first_data)
            == quality._normalized_response(second_data),
            "long tool-result cold/warm output differs")
    require(first["cached_tokens"] == 0 and second["cached_tokens"] > 0,
            "long tool-result cache accounting differs")
    return {
        "requests": [first, second],
        "construction": [construction],
        "facts": {"marker_rule_passed": True, "cold_warm_exact": True},
    }


def _interleaved_sessions(context: Context, case: Json) -> Json:
    target = case["target_prompt_tokens"]
    messages_a, _, construction_a = _fit_recipe(
        context.tokenizer, target,
        _multiturn_recipe(case["id"], "A", "SESSION-A-731"),
        namespace=case["id"] + "-A",
        template_kwargs_mode=context.template_kwargs_mode)
    messages_b, _, construction_b = _fit_recipe(
        context.tokenizer, target,
        _multiturn_recipe(case["id"], "B", "SESSION-B-947"),
        namespace=case["id"] + "-B",
        template_kwargs_mode=context.template_kwargs_mode)
    payload_a = _payload(
        case, messages_a, served_model_name=context.served_model_name)
    payload_b = _payload(
        case, messages_b, served_model_name=context.served_model_name)
    data_a1, a1 = _post(context, payload_a, target)
    data_b1, b1 = _post(context, payload_b, target)
    data_a2, a2 = _post(context, payload_a, target)
    data_b2, b2 = _post(context, payload_b, target)
    require(_content(data_a1) == "SESSION-A-731"
            and _content(data_b1) == "SESSION-B-947",
            "interleaved session marker rule failed")
    require(quality._normalized_response(data_a1)
            == quality._normalized_response(data_a2)
            and quality._normalized_response(data_b1)
            == quality._normalized_response(data_b2),
            "interleaved session warm output differs")
    require(a1["cached_tokens"] == 0 and b1["cached_tokens"] == 0
            and a2["cached_tokens"] > 0 and b2["cached_tokens"] > 0,
            "interleaved session cache isolation differs")
    return {
        "requests": [a1, b1, a2, b2],
        "construction": [construction_a, construction_b],
        "facts": {
            "session_markers_correct": True,
            "warm_outputs_exact": True,
            "session_cache_isolated": True,
        },
    }


def _reasoning_rule_diagnostics(
    content: str,
    reasoning: str,
    expected: str,
) -> Json:
    markers = (
        "BEGIN-MARKER-731",
        "MIDDLE-MARKER-552",
        "END-MARKER-947",
    )

    def markers_present(value: str) -> bool:
        return all(marker in value for marker in markers)

    def markers_in_order(value: str) -> bool:
        positions = [value.find(marker) for marker in markers]
        return all(position >= 0 for position in positions) \
            and positions == sorted(positions) \
            and len(set(positions)) == len(positions)

    return {
        "content_arithmetic_present": "323" in content,
        "content_contains_expected": expected in content,
        "content_exact_expected": content == expected,
        "content_expected_prefix": content.startswith(expected),
        "content_expected_single_occurrence": content.count(expected) == 1,
        "content_expected_suffix": content.endswith(expected),
        "content_markers_in_order": markers_in_order(content),
        "content_markers_present": markers_present(content),
        "reasoning_arithmetic_present": "323" in reasoning,
        "reasoning_contains_expected": expected in reasoning,
        "reasoning_markers_in_order": markers_in_order(reasoning),
        "reasoning_markers_present": markers_present(reasoning),
    }


def _reasoning_semantic_rule_passed(diagnostics: Json) -> bool:
    required = (
        "content_arithmetic_present",
        "content_contains_expected",
        "content_expected_single_occurrence",
        "content_expected_suffix",
        "content_markers_in_order",
        "content_markers_present",
    )
    return all(diagnostics.get(key) is True for key in required)


def _reasoning(context: Context, case: Json) -> Json:
    expected = "BEGIN-MARKER-731|MIDDLE-MARKER-552|END-MARKER-947|323"

    def recipe(filler: str) -> tuple[list[Json], None]:
        content = (
            f"case={case['id']}\nBEGIN-MARKER-731\n"
            + _filler(filler)
            + "\nEND-MARKER-947\nReason about the marker order and compute "
            + "17*19. After reasoning, end the final answer with exactly this "
            + "sequence on the last line: "
            + expected
        )
        return [{"role": "user", "content": content}], None

    messages, _, construction = _fit_recipe(
        context.tokenizer, case["target_prompt_tokens"], recipe,
        namespace=case["id"], thinking=True,
        template_kwargs_mode=context.template_kwargs_mode)
    data, summary = _post(
        context, _payload(
            case, messages, served_model_name=context.served_model_name,
            thinking=True),
        case["target_prompt_tokens"])
    message = quality._message(data)
    reasoning = quality._reasoning(message).strip()
    content = (message.get("content") or "").strip()
    diagnostics = _reasoning_rule_diagnostics(content, reasoning, expected)
    context.record_failure_facts(diagnostics)
    require(
        _reasoning_semantic_rule_passed(diagnostics),
        "long-context reasoning semantic answer differs",
    )
    require(bool(reasoning) and reasoning != content,
            "reasoning/content split failed")
    require(summary["finish_reason"] == "stop"
            and summary["completion_tokens"] < case["max_tokens"],
            "long-context reasoning did not finish naturally")
    return {
        "requests": [summary],
        "construction": [construction],
        "facts": {
            **diagnostics,
            "answer_rule_passed": True,
            "marker_rule_passed": True,
            "natural_finish_before_max_tokens": True,
            "reasoning_content_split": True,
        },
    }


def _agent_large_budget(context: Context, case: Json) -> Json:
    target_name = "report_agent_marker"
    marker = "AGENT-235K-731"
    tools = _large_tools(
        target_name, case["id"], target_arguments=("key", "ordinal"))

    def recipe(filler: str) -> tuple[list[Json], list[Json]]:
        tool_content = (
            f"case={case['id']}\n" + _filler(filler)
            + f"\nFINAL_AGENT_MARKER={marker}"
        )
        messages = [
            {"role": "system", "content": "Use tools exactly as requested."},
            {"role": "user", "content": "Load the prior workspace record."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call-agent-context",
                    "type": "function",
                    "function": {
                        "name": "load_workspace",
                        "arguments": '{"workspace":"quality"}',
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call-agent-context",
                "name": "load_workspace",
                "content": tool_content,
            },
            {"role": "user", "content": (
                "Reason briefly, then call "
                f"{target_name} exactly once with key={marker} and "
                "ordinal=235000. Do not continue after selecting the tool.")},
        ]
        return messages, tools

    messages, fitted_tools, construction = _fit_recipe(
        context.tokenizer, case["target_prompt_tokens"], recipe,
        namespace=case["id"], thinking=True,
        template_kwargs_mode=context.template_kwargs_mode)
    payload = _payload(
        case,
        messages,
        served_model_name=context.served_model_name,
        tools=fitted_tools,
        thinking=True,
    )
    payload["tool_choice"] = "auto"
    first_data, first = _post(
        context, payload, case["target_prompt_tokens"])
    second_data, second = _post(
        context, payload, case["target_prompt_tokens"])
    reasoning = quality._reasoning(quality._message(first_data))
    expected_arguments = {"key": marker, "ordinal": 235000}
    _require_single_tool_call(
        first_data, target_name, expected_arguments, allow_content=True)
    _require_single_tool_call(
        second_data, target_name, expected_arguments, allow_content=True)
    require(bool(reasoning.strip()),
            "235K Agent response contains no reasoning_content")
    require(quality._normalized_response(first_data)
            == quality._normalized_response(second_data),
            "235K Agent cold/warm output differs")
    require(first["cached_tokens"] == 0 and second["cached_tokens"] > 0,
            "235K Agent cache accounting differs")
    require(
        first["completion_tokens"] < case["max_tokens"]
        and second["completion_tokens"] < case["max_tokens"],
        "235K Agent response exhausted the large output budget",
    )
    return {
        "requests": [first, second],
        "construction": [construction],
        "facts": {
            "large_max_tokens_accepted": True,
            "natural_finish_before_max_tokens": True,
            "tool_content_mode": "optional",
            "tool_choice_mode": "auto",
            "tool_count": len(fitted_tools or []),
            "tool_call_rule_passed": True,
            "reasoning_present": True,
            "cold_warm_exact": True,
        },
    }


Handler = Callable[[Context, Json], Json]
HANDLERS: dict[str, Handler] = {
    "short_basic_recall": _simple_recall,
    "4k_cold_warm_recall": _simple_recall,
    "32k_partial_branch": _partial_branch,
    "32k_multimodal_isolation": _multimodal,
    "65k_multiturn_large_tools": _large_tool_call,
    "65k_long_tool_result": _long_tool_result,
    "65k_interleaved_sessions": _interleaved_sessions,
    "131k_cold_warm_recall": _simple_recall,
    "131k_reasoning_recall": _reasoning,
    "235k_agent_large_output_budget": _agent_large_budget,
    "235k_partial_branch": _partial_branch,
    "near_262k_capacity": _simple_recall,
}


def main() -> int:
    import transformers
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", default="llm")
    parser.add_argument(
        "--chat-template-kwargs-mode",
        choices=exact_prompt.TEMPLATE_KWARG_MODES,
        default="direct",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--cache-trace-file",
        type=Path,
        help="service log containing privacy-safe BI100 cache trace v4 records",
    )
    parser.add_argument("--tier", choices=tuple(TIER_RANK), default="quick")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--timeout-s", type=float, default=5400)
    parser.add_argument("--label", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--runtime-identity", required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--fresh-service-attested", action="store_true")
    args = parser.parse_args()
    if not args.fresh_service_attested:
        parser.error("matrix requires an attested fresh service and empty cache")
    if args.gpu_count <= 0:
        parser.error("gpu-count must be positive")
    if args.timeout_s <= 0:
        parser.error("timeout-s must be positive")
    if args.max_model_len != 262144:
        parser.error("long-context quality matrix requires max-model-len=262144")
    if args.tensor_parallel_size <= 0:
        parser.error("tensor-parallel-size must be positive")
    if not runtime_contract.is_git_revision(args.source_revision):
        parser.error("--source-revision must be a fixed Git object id")

    manifest, manifest_sha = _load_manifest(args.manifest)
    loaded_runtime_contract, runtime_contract_sha = _load_runtime_contract(
        args.runtime_contract, args)
    require({case["id"] for case in manifest["cases"]} == set(HANDLERS),
            "matrix handlers do not match frozen cases")
    require(manifest["generated_assets"] == _generated_asset_contract(),
            "generated assets differ from the frozen matrix")
    selected = _selected_cases(manifest, args.tier, args.case)
    requires_cache_trace = any(
        case["id"] in {
            "32k_multimodal_isolation",
            "32k_partial_branch",
            "235k_partial_branch",
        }
        for case in selected)
    if requires_cache_trace and args.cache_trace_file is None:
        parser.error("selected cache matrix case requires --cache-trace-file")
    if args.cache_trace_file is not None and not args.cache_trace_file.is_file():
        parser.error("--cache-trace-file must be an existing regular file")
    model_path = Path(args.model_path)
    require(model_path.is_dir(), "model path is not a local directory")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True)
    client = quality.Client(args.base)
    models_response = client.models(args.served_model_name)
    probe_messages = [{"role": "user", "content": "template probe"}]
    probe_false = exact_prompt.chat_template_token_ids(
        tokenizer,
        probe_messages,
        thinking=False,
        template_kwargs_mode=args.chat_template_kwargs_mode,
    )
    probe_true = exact_prompt.chat_template_token_ids(
        tokenizer,
        probe_messages,
        thinking=True,
        template_kwargs_mode=args.chat_template_kwargs_mode,
    )
    if any("reasoning" in case["capabilities"] for case in selected):
        require(probe_false != probe_true,
                "selected chat-template mode does not distinguish thinking")
    tokenizer_metadata = exact_prompt.tokenizer_identity(model_path, tokenizer)
    context = Context(
        client,
        tokenizer,
        args.timeout_s,
        served_model_name=args.served_model_name,
        template_kwargs_mode=args.chat_template_kwargs_mode,
        cache_trace_path=args.cache_trace_file,
        gdn_cache_policy=loaded_runtime_contract[
            "environment"]["BI100_GDN_CACHE_POLICY"],
    )
    results = []
    report: Json = {
        "schema": SCHEMA,
        "version": 6,
        "qualified": False,
        "quality_run_eligible_for_baseline": False,
        "overall_promotion_authorized": False,
        "label": args.label,
        "run_id_sha256": hashlib.sha256(
            args.run_id.encode("utf-8")).hexdigest(),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest": {
            "path_name": args.manifest.name,
            "sha256": manifest_sha,
            "total_cases": len(manifest["cases"]),
            "seed": manifest["seed"],
        },
        "runtime": {
            "source_revision": args.source_revision,
            "runtime_identity": args.runtime_identity,
            "runtime_overlay_sha256": loaded_runtime_contract[
                "runtime_overlay_sha256"],
            "service_command_sha256": quality._sha256_json(
                loaded_runtime_contract["command"]),
            "service_env_sha256": quality._sha256_json(
                loaded_runtime_contract["environment"]),
            "instance": args.instance,
            "gpu_count": args.gpu_count,
            "tensor_parallel_size": args.tensor_parallel_size,
            "model_path": args.model_path,
            "max_model_len": args.max_model_len,
            "served_model_name": args.served_model_name,
            "fresh_service_attested": True,
            "cache_trace_v4_attested": args.cache_trace_file is not None,
            "model_list_contract_sha256": quality._sha256_json(
                models_response),
        },
        "runtime_contract": {
            "sha256": runtime_contract_sha,
            "contract": loaded_runtime_contract,
        },
        "generator": {
            "runner_sha256": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(),
            "exact_prompt_module_sha256": hashlib.sha256(
                Path(exact_prompt.__file__).read_bytes()).hexdigest(),
            "transformers_version": transformers.__version__,
        },
        "tokenizer": {
            **tokenizer_metadata,
            "template_kwargs_mode": args.chat_template_kwargs_mode,
            "thinking_false_prompt_sha256": quality._sha256_json(probe_false),
            "thinking_true_prompt_sha256": quality._sha256_json(probe_true),
            "thinking_modes_distinct": probe_false != probe_true,
        },
        "selection": {
            "tier": args.tier,
            "explicit_cases": args.case,
            "selected_cases": len(selected),
        },
        "privacy": {
            "contains_raw_requests": False,
            "contains_raw_model_outputs": False,
            "contains_credentials": False,
        },
        "summary": {},
        "cases": results,
    }
    started = time.perf_counter()
    for case in selected:
        context.begin_case()
        case_started = time.perf_counter()
        try:
            observation = HANDLERS[case["id"]](context, case)
            status = "pass"
            error_code = ""
        except (quality.CaseFailure, MatrixFailure) as error:
            observation = context.failure_observation()
            status = "fail"
            error_code = str(error)
        except Exception as error:
            observation = context.failure_observation()
            status = "fail"
            error_code = f"unexpected {type(error).__name__}"
        results.append({
            **case,
            "status": status,
            "ok": status == "pass",
            "elapsed_s": time.perf_counter() - case_started,
            "error_code": error_code,
            "observation": observation,
        })
        _atomic_write(args.out, report)
        print(f"[{status.upper()}] {case['ordinal']:02d} {case['id']}",
              flush=True)
        if status == "fail" and args.fail_fast:
            break

    passed = sum(case["status"] == "pass" for case in results)
    complete = len(results) == len(selected)
    qualified = complete and passed == len(selected)
    report["qualified"] = qualified
    report["quality_run_eligible_for_baseline"] = bool(
        qualified and args.tier == "extended" and not args.case
        and len(results) == len(manifest["cases"])
        and args.gpu_count == 4 and args.tensor_parallel_size == 4
        and args.cache_trace_file is not None)
    report["summary"] = {
        "passed": passed,
        "failed": len(results) - passed,
        "total": len(results),
        "selected_total": len(selected),
        "complete": complete,
        "pass_rate": passed / len(results) if results else 0.0,
        "wall_s": time.perf_counter() - started,
    }
    _atomic_write(args.out, report)
    return 0 if qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
