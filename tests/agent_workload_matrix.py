#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


Json = dict[str, Any]


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
        raw = exc.read().decode("utf-8", "replace")
        raise AssertionError(f"HTTP {exc.code}: {raw[:1000]}") from exc


def parse_arguments(value: Any) -> Json:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        assert isinstance(parsed, dict), parsed
        return parsed
    raise AssertionError(f"unsupported tool arguments: {value!r}")


def normalize(response: Json, elapsed_s: float) -> Json:
    choice = response["choices"][0]
    message = choice["message"]
    calls = message.get("tool_calls") or []
    normalized_calls = []
    for call in calls:
        function = call.get("function") or {}
        normalized_calls.append({
            "name": function.get("name"),
            "arguments": parse_arguments(function.get("arguments") or "{}"),
        })
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    return {
        "elapsed_s": elapsed_s,
        "finish_reason": choice.get("finish_reason"),
        "content": content,
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "reasoning_chars": len(reasoning),
        "tool_calls": normalized_calls,
        "usage": response.get("usage") or {},
    }


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


def validate(case: Json, result: Json) -> None:
    calls = result["tool_calls"]
    expected_tool = case.get("expected_tool")
    if expected_tool:
        assert calls, result
        assert calls[0]["name"] == expected_tool, result
        arguments = calls[0]["arguments"]
        for key, value in (case.get("expected_args") or {}).items():
            assert arguments.get(key) == value, (key, value, arguments)
        for key in case.get("required_arg_keys") or []:
            assert key in arguments and arguments[key] not in (None, ""), result
    if case.get("content_contains"):
        assert case["content_contains"] in result["content"], result
    if case.get("content_contains_also"):
        assert case["content_contains_also"] in result["content"], result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-s", type=float, default=360)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report: Json = {"ok": False, "base": args.base, "cases": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for name, case in build_cases().items():
        started = time.monotonic()
        try:
            response, elapsed = post(
                args.base, case["payload"], args.timeout_s)
            result = normalize(response, elapsed)
            validate(case, result)
            report["cases"][name] = {"ok": True, **result}
            print(f"[PASS] {name} {elapsed:.3f}s", flush=True)
        except Exception as exc:
            report["cases"][name] = {
                "ok": False,
                "elapsed_s": time.monotonic() - started,
                "error": repr(exc),
            }
            args.out.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            raise
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    report["ok"] = True
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
