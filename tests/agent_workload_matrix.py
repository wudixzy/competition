#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import quality_gate_api as quality_api
import quality_runtime_contract as runtime_contract


Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "quality/agent_workload_matrix.v1.json"
EXPECTED_MANIFEST_SHA256 = (
    "962d19f51cfbeb3f414e62444a225029616ed547682e5a97219b0af98c8959ba"
)
REPORT_SCHEMA = "bi100-agent-workload-result-v1"
REPORT_VERSION = 1


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise AssertionError(reason)


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def atomic_write(path: Path, report: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                     dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=True, indent=2,
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


def tool(name: str, required: list[str], properties: Json) -> Json:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Execute the {name} operation.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


CORE_TOOLS = [
    tool("terminal", ["command"], {"command": {"type": "string"}}),
    tool("read", ["path"], {"path": {"type": "string"}}),
    tool("edit", ["file_path", "old_string", "new_string"], {
        "file_path": {"type": "string"},
        "old_string": {"type": "string"},
        "new_string": {"type": "string"},
    }),
    tool("web_search", ["query"], {"query": {"type": "string"}}),
]


def post(base: str, payload: Json, timeout_s: float) -> tuple[Json, float]:
    request = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
            return body, time.monotonic() - started
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        digest = hashlib.sha256(raw).hexdigest()
        raise AssertionError(
            f"HTTP {exc.code}; response_sha256={digest}") from exc


def post_stream(
    base: str,
    payload: Json,
    timeout_s: float,
) -> tuple[int, Json, float]:
    started = time.monotonic()
    status, stream = quality_api.Client(base).stream(
        payload, timeout=timeout_s)
    return status, stream, time.monotonic() - started


def parse_arguments(value: Any) -> Json:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        assert isinstance(parsed, dict), parsed
        return parsed
    raise AssertionError(f"unsupported tool arguments: {value!r}")


def normalize(response: Json, elapsed_s: float) -> Json:
    require(isinstance(response, dict), "response must be an object")
    choices = response.get("choices")
    require(isinstance(choices, list) and len(choices) == 1,
            "response must contain one choice")
    choice = choices[0]
    require(isinstance(choice, dict), "choice must be an object")
    message = choice.get("message")
    require(isinstance(message, dict), "assistant message is missing")
    calls = message.get("tool_calls") or []
    require(isinstance(calls, list), "tool_calls must be a list")
    normalized_calls = []
    for call in calls:
        function = call.get("function") or {}
        normalized_calls.append({
            "name": function.get("name"),
            "arguments": parse_arguments(function.get("arguments") or "{}"),
        })
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    require(isinstance(content, str), "content must be a string")
    require(isinstance(reasoning, str), "reasoning_content must be a string")
    usage = response.get("usage") or {}
    require(isinstance(usage, dict), "usage must be an object")
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(field)
        require(isinstance(value, int) and not isinstance(value, bool)
                and value >= 0, f"usage {field} is invalid")
    require(usage["total_tokens"]
            == usage["prompt_tokens"] + usage["completion_tokens"],
            "usage total_tokens is inconsistent")
    return {
        "elapsed_s": elapsed_s,
        "finish_reason": choice.get("finish_reason"),
        "content": content,
        "reasoning_content": reasoning,
        "tool_calls": normalized_calls,
        "usage": usage,
    }


def normalize_stream(status: int, stream: Json, elapsed_s: float) -> Json:
    require(status == 200, "stream request did not return HTTP 200")
    require(stream.get("done") == 1,
            "stream must contain exactly one DONE event")
    require(stream.get("usage_blocks") == 1,
            "stream must contain exactly one final usage block")
    require(isinstance(stream.get("chunks"), int) and stream["chunks"] >= 2,
            "stream must contain at least two JSON chunks")
    finish_reasons = stream.get("finish_reasons")
    require(finish_reasons == ["tool_calls"],
            "stream must finish exactly once as tool_calls")
    response = {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "content": stream.get("content"),
                "reasoning_content": stream.get("reasoning_content"),
                "tool_calls": [
                    {
                        "function": {
                            "name": call.get("name"),
                            "arguments": call.get("arguments"),
                        },
                    }
                    for call in stream.get("tool_calls") or []
                ],
            },
        }],
        "usage": stream.get("usage"),
    }
    return normalize(response, elapsed_s)


def base_payload(messages: list[Json], *, tools: list[Json] | None = None,
                 max_tokens: int = 128, thinking: Any = False) -> Json:
    payload: Json = {
        "model": "llm",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 20260716,
        "thinking": thinking,
    }
    if tools is not None:
        payload["tools"] = tools
    return payload


def forced_case(name: str, prompt: str, expected_args: Json) -> Json:
    payload = base_payload(
        [{"role": "user", "content": prompt}], tools=CORE_TOOLS)
    payload["tool_choice"] = {
        "type": "function", "function": {"name": name}}
    return {
        "payload": payload,
        "expected_tool": name,
        "expected_args": expected_args,
    }


def as_streaming(case: Json) -> Json:
    value = dict(case)
    value["payload"] = dict(case["payload"])
    value["payload"].update({
        "stream": True,
        "stream_options": {
            "include_usage": True,
            "continuous_usage_stats": False,
        },
    })
    value["stream"] = True
    return value


def build_cases() -> dict[str, Json]:
    cases = {
        "forced_terminal": forced_case(
            "terminal", "Run: grep -R TODO src | head", {
                "command": "grep -R TODO src | head"}),
        "forced_read": forced_case(
            "read", "Read /workspace/project/README.md", {
                "path": "/workspace/project/README.md"}),
        "forced_edit": forced_case(
            "edit", "Replace old_value with new_value in /tmp/config.py", {
                "file_path": "/tmp/config.py",
                "old_string": "old_value",
                "new_string": "new_value",
            }),
        "forced_web_search": forced_case(
            "web_search", "Search for BI100 programming documentation", {
                "query": "BI100 programming documentation"}),
    }

    auto_payload = base_payload([{
        "role": "user",
        "content": "Call terminal to run exactly: pwd && ls -la",
    }], tools=CORE_TOOLS)
    auto_payload["tool_choice"] = "auto"
    cases["auto_terminal"] = {
        "payload": auto_payload,
        "expected_tool": "terminal",
        "required_arg_keys": ["command"],
    }
    cases["stream_forced_terminal"] = as_streaming(forced_case(
        "terminal", "Run exactly: printf STREAM_NAMED_OK", {
            "command": "printf STREAM_NAMED_OK",
        }))
    stream_auto_payload = base_payload([{
        "role": "user",
        "content": "Call terminal to run exactly: printf STREAM_AUTO_OK",
    }], tools=CORE_TOOLS)
    stream_auto_payload["tool_choice"] = "auto"
    cases["stream_auto_terminal"] = as_streaming({
        "payload": stream_auto_payload,
        "expected_tool": "terminal",
        "required_arg_keys": ["command"],
    })

    roundtrip = base_payload([
        {"role": "user", "content": "Read /tmp/value.txt"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_read_1",
                "type": "function",
                "function": {
                    "name": "read",
                    "arguments": json.dumps({"path": "/tmp/value.txt"}),
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call_read_1",
            "name": "read",
            "content": "The marker is TOOL_RESULT_OK.",
        },
        {
            "role": "user",
            "content": "Reply with the marker only.",
        },
    ], tools=CORE_TOOLS, max_tokens=32)
    roundtrip["tool_choice"] = "none"
    cases["tool_result_roundtrip"] = {
        "payload": roundtrip,
        "content_contains": "TOOL_RESULT_OK",
    }

    history: list[Json] = [{
        "role": "system", "content": "Retain the conversation markers."}]
    for index in range(20):
        history.append({
            "role": "user", "content": f"Remember item {index}: VALUE_{index}."})
        history.append({
            "role": "assistant", "content": f"Stored VALUE_{index}."})
    history.append({
        "role": "user", "content": "Reply exactly with VALUE_19."})
    long_history = base_payload(history, tools=CORE_TOOLS, max_tokens=32)
    long_history["tool_choice"] = "none"
    cases["long_history"] = {
        "payload": long_history,
        "content_contains": "VALUE_19",
    }

    many_tools = [
        tool(f"operation_{index}", ["value"], {
            "value": {"type": "integer"},
            "note": {"type": "string"},
        })
        for index in range(92)
    ]
    large_schema = base_payload([{
        "role": "user",
        "content": "Call operation_91 with value 91 and note final.",
    }], tools=many_tools)
    large_schema["tool_choice"] = {
        "type": "function", "function": {"name": "operation_91"}}
    cases["large_tool_schema"] = {
        "payload": large_schema,
        "expected_tool": "operation_91",
        "expected_args": {"value": 91, "note": "final"},
    }

    multi_system = base_payload([
        {"role": "system", "content": "Token A is SYSTEM_A."},
        {"role": "system", "content": "Token B is SYSTEM_B."},
        {"role": "user", "content": "Reply exactly: SYSTEM_A SYSTEM_B"},
    ], max_tokens=32)
    cases["multiple_system"] = {
        "payload": multi_system,
        "content_contains": "SYSTEM_A",
        "content_contains_also": "SYSTEM_B",
    }
    return cases


def validate(case: Json, result: Json) -> Json:
    facts: Json = {
        "http_200": True,
        "usage_valid": True,
    }
    calls = result["tool_calls"]
    expected_tool = case.get("expected_tool")
    if expected_tool:
        require(bool(calls), "expected tool call is missing")
        require(calls[0]["name"] == expected_tool,
                "selected tool name differs")
        require(result["finish_reason"] == "tool_calls",
                "tool request did not finish as tool_calls")
        arguments = calls[0]["arguments"]
        for key, value in (case.get("expected_args") or {}).items():
            require(arguments.get(key) == value,
                    "exact tool argument differs")
        for key in case.get("required_arg_keys") or []:
            require(key in arguments and arguments[key] not in (None, ""),
                    "required tool argument is missing")
        facts.update({
            "tool_call_valid": True,
            "tool_arguments_valid_json": True,
            "tool_argument_rule_passed": True,
            "finish_reason_tool_calls": True,
        })
    if case.get("content_contains"):
        require(case["content_contains"] in result["content"],
                "required content marker is missing")
        facts["primary_content_rule_passed"] = True
    if case.get("content_contains_also"):
        require(case["content_contains_also"] in result["content"],
                "secondary content marker is missing")
        facts["secondary_content_rule_passed"] = True
    return facts


def safe_observation(result: Json, facts: Json) -> Json:
    usage = result["usage"]
    details = usage.get("prompt_tokens_details") or {}
    semantic = {
        "finish_reason": result["finish_reason"],
        "content": result["content"],
        "reasoning_content": result["reasoning_content"],
        "tool_calls": result["tool_calls"],
    }
    return {
        "elapsed_s": result["elapsed_s"],
        "finish_reason": result["finish_reason"],
        "content_chars": len(result["content"]),
        "reasoning_chars": len(result["reasoning_content"]),
        "tool_call_count": len(result["tool_calls"]),
        "prompt_tokens": usage["prompt_tokens"],
        "cached_tokens": details.get("cached_tokens", 0),
        "completion_tokens": usage["completion_tokens"],
        "semantic_output_sha256": sha256_json(semantic),
        "facts": facts,
    }


def load_manifest(path: Path) -> tuple[Json, str]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    require(digest == EXPECTED_MANIFEST_SHA256,
            "agent workload manifest identity is invalid")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == "bi100-agent-workload-manifest-v1"
            and manifest.get("version") == 1,
            "agent workload manifest schema is invalid")
    expected_ids = list(build_cases())
    actual_ids = [case.get("id") for case in manifest.get("cases", [])]
    require(actual_ids == expected_ids,
            "agent workload manifest case order differs")
    return manifest, digest


def select_cases(cases: dict[str, Json],
                 requested: list[str]) -> dict[str, Json]:
    if not requested:
        return cases
    require(len(set(requested)) == len(requested),
            "agent case ids must be unique")
    unknown = set(requested) - set(cases)
    require(not unknown, "unknown agent workload case requested")
    requested_set = set(requested)
    return {
        name: case for name, case in cases.items()
        if name in requested_set
    }


def load_runtime_contract(path: Path, source_revision: str,
                          runtime_identity: str, instance: str) -> tuple[Json, str]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "source_revision": source_revision,
        "runtime_identity": runtime_identity,
        "instance": instance,
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
        raise AssertionError(str(error)) from error
    return contract, digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-s", type=float, default=360)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--runtime-identity", required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest, manifest_sha = load_manifest(args.manifest)
    try:
        selected_cases = select_cases(build_cases(), args.case)
    except AssertionError as error:
        parser.error(str(error))
    contract, contract_sha = load_runtime_contract(
        args.runtime_contract, args.source_revision,
        args.runtime_identity, args.instance)
    report: Json = {
        "schema": REPORT_SCHEMA,
        "version": REPORT_VERSION,
        "qualified": False,
        "promotion_authorized": False,
        "label": args.label,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id_sha256": hashlib.sha256(args.run_id.encode()).hexdigest(),
        "manifest": {
            "path_name": args.manifest.name,
            "sha256": manifest_sha,
            "revision": manifest["revision"],
            "case_count": len(manifest["cases"]),
        },
        "selection": {
            "explicit_cases": args.case,
            "selected_cases": len(selected_cases),
            "promotion_requires": "all 11 cases and baseline comparison",
        },
        "runtime": {
            "source_revision": args.source_revision,
            "runtime_identity": args.runtime_identity,
            "runtime_overlay_sha256": contract["runtime_overlay_sha256"],
            "runtime_contract_sha256": contract_sha,
            "instance": args.instance,
            "gpu_count": 4,
            "tensor_parallel_size": 4,
            "max_model_len": 262144,
        },
        "runtime_contract": {
            "sha256": contract_sha,
            "contract": contract,
        },
        "generator": {
            "runner_sha256": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(),
            "seed": 20260716,
        },
        "privacy": {
            "contains_raw_requests": False,
            "contains_raw_model_outputs": False,
            "contains_tool_arguments": False,
            "contains_credentials": False,
        },
        "summary": {},
        "cases": [],
    }
    atomic_write(args.out, report)
    for name, case in selected_cases.items():
        started = time.monotonic()
        try:
            if case.get("stream"):
                status, stream, elapsed = post_stream(
                    args.base, case["payload"], args.timeout_s)
                result = normalize_stream(status, stream, elapsed)
            else:
                response, elapsed = post(
                    args.base, case["payload"], args.timeout_s)
                result = normalize(response, elapsed)
            facts = validate(case, result)
            if case.get("stream"):
                facts.update({
                    "sse_contract_valid": True,
                    "single_done_event": True,
                    "single_final_usage_block": True,
                })
            report["cases"].append({
                "id": name,
                "status": "pass",
                "error_type": "",
                "error_sha256": None,
                "observation": safe_observation(result, facts),
            })
            print(f"[PASS] {name} {elapsed:.3f}s", flush=True)
        except Exception as exc:
            report["cases"].append({
                "id": name,
                "status": "fail",
                "error_type": type(exc).__name__,
                "error_sha256": hashlib.sha256(
                    str(exc).encode("utf-8", "replace")).hexdigest(),
                "observation": {
                    "elapsed_s": time.monotonic() - started,
                    "finish_reason": None,
                    "content_chars": None,
                    "reasoning_chars": None,
                    "tool_call_count": None,
                    "prompt_tokens": None,
                    "cached_tokens": None,
                    "completion_tokens": None,
                    "semantic_output_sha256": None,
                    "facts": {},
                },
            })
            print(f"[FAIL] {name} {type(exc).__name__}", flush=True)
        passed = sum(row["status"] == "pass" for row in report["cases"])
        failed = sum(row["status"] == "fail" for row in report["cases"])
        report["summary"] = {
            "complete": len(report["cases"]) == len(selected_cases),
            "passed": passed,
            "failed": failed,
            "total": len(report["cases"]),
        }
        atomic_write(args.out, report)
    report["qualified"] = (
        report["summary"]["complete"] and report["summary"]["failed"] == 0)
    atomic_write(args.out, report)
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
