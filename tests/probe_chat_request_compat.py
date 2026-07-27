#!/usr/bin/env python3
"""Probe chat request validation against the installed runtime protocol.

The cases are synthetic and contain no evaluation prompts or user data. Output
contains only bounded case names and validation locations/types.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

Json = dict[str, Any]


def _base(**overrides: Any) -> Json:
    payload: Json = {
        "model": "llm",
        "messages": [{"role": "user", "content": "synthetic request"}],
        "max_tokens": 8,
    }
    payload.update(overrides)
    return payload


def _function_tool(*, strict: bool | None = None) -> Json:
    function: Json = {
        "name": "lookup",
        "description": "Return a synthetic value.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    }
    if strict is not None:
        function["strict"] = strict
    return {"type": "function", "function": function}


CASES: tuple[Json, ...] = (
    {
        "name": "basic_text",
        "expected": "accept",
        "payload": _base(),
    },
    {
        "name": "function_tool_default",
        "expected": "accept",
        "payload": _base(tools=[_function_tool()]),
    },
    {
        "name": "function_tool_strict_false",
        "expected": "accept",
        "payload": _base(tools=[_function_tool(strict=False)]),
    },
    {
        "name": "function_tool_strict_true",
        "expected": "reject",
        "payload": _base(tools=[_function_tool(strict=True)]),
    },
    {
        "name": "tool_choice_required",
        "expected": "reject",
        "payload": _base(
            tools=[_function_tool()],
            tool_choice="required",
        ),
    },
    {
        "name": "assistant_tool_history_null_content",
        "expected": "accept",
        "payload": _base(
            tools=[_function_tool()],
            tool_choice="none",
            messages=[
                {"role": "user", "content": "lookup synthetic key"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_synthetic_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": "{\"key\":\"synthetic\"}",
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_synthetic_1",
                    "content": "synthetic result",
                },
            ],
        ),
    },
    {
        "name": "tool_result_text_parts",
        "expected": "accept",
        "payload": _base(
            messages=[
                {
                    "role": "tool",
                    "tool_call_id": "call_synthetic_2",
                    "content": [{
                        "type": "text",
                        "text": "synthetic result",
                    }],
                },
                {"role": "user", "content": "continue"},
            ],
        ),
    },
    {
        "name": "image_url_data",
        "expected": "accept",
        "payload": _base(messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            "data:image/png;base64,"
                            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
                            "CAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB"
                            "9Wl2nCEAAAAASUVORK5CYII="
                        ),
                    },
                },
                {"type": "text", "text": "describe the synthetic image"},
            ],
        }]),
    },
    {
        "name": "multiple_system_text",
        "expected": "accept",
        "payload": _base(messages=[
            {"role": "system", "content": "synthetic rule A"},
            {"role": "system", "content": "synthetic rule B"},
            {"role": "user", "content": "respond"},
        ]),
    },
    {
        "name": "multiple_system_text_parts",
        "expected": "accept",
        "payload": _base(messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "synthetic rule A1"},
                    {"type": "text", "text": "synthetic rule A2"},
                ],
            },
            {"role": "system", "content": "synthetic rule B"},
            {"role": "user", "content": "respond"},
        ]),
    },
    {
        "name": "assistant_tool_arguments_object",
        "expected": "accept",
        "payload": _base(messages=[{
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_synthetic_3",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": {"key": "synthetic"},
                },
            }],
        }]),
    },
    {
        "name": "assistant_tool_arguments_invalid_json",
        "expected": "reject",
        "payload": _base(messages=[{
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_synthetic_4",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": "{invalid",
                },
            }],
        }]),
    },
)


def _bounded_errors(exc: Exception) -> list[Json]:
    errors_method = getattr(exc, "errors", None)
    if not callable(errors_method):
        return [{"location": [], "type": type(exc).__name__}]
    try:
        errors = errors_method(
            include_url=False,
            include_context=False,
            include_input=False,
        )
    except TypeError:
        errors = errors_method()

    bounded = []
    for error in errors:
        location = error.get("loc", ())
        bounded.append({
            "location": [
                item for item in location if isinstance(item, (str, int))
            ][:12],
            "type": str(error.get("type", "unknown"))[:80],
        })
    return bounded[:16]


def run() -> Json:
    from vllm.entrypoints.chat_utils import _postprocess_messages
    from vllm.entrypoints.openai.protocol import ChatCompletionRequest

    observations = []
    mismatches = []
    for case in CASES:
        try:
            request = ChatCompletionRequest.model_validate(case["payload"])
            _postprocess_messages(request.messages)
        except Exception as exc:
            accepted = False
            errors = _bounded_errors(exc)
            serialized_tools = None
        else:
            accepted = True
            errors = []
            serialized_tools = (
                None if request.tools is None
                else [tool.model_dump() for tool in request.tools]
            )

        actual = "accept" if accepted else "reject"
        matched = actual == case["expected"]
        observation = {
            "name": case["name"],
            "expected": case["expected"],
            "actual": actual,
            "matched": matched,
            "errors": errors,
        }
        if case["name"] == "function_tool_strict_false" and accepted:
            strict_forwarded = any(
                "strict" in tool.get("function", {})
                for tool in serialized_tools or []
            )
            observation["strict_forwarded_to_template"] = strict_forwarded
            matched = matched and not strict_forwarded
            observation["matched"] = matched
        if (case["name"] == "assistant_tool_arguments_object"
                and accepted):
            arguments = request.messages[0]["tool_calls"][0][
                "function"]["arguments"]
            object_preserved = arguments == {"key": "synthetic"}
            observation["argument_object_preserved"] = object_preserved
            matched = matched and object_preserved
            observation["matched"] = matched
        observations.append(observation)
        if not matched:
            mismatches.append(case["name"])

    package_versions = {}
    for package in ("openai", "pydantic", "vllm"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "unknown"

    return {
        "schema": "bi100-chat-request-compat-v1",
        "synthetic_only": True,
        "package_versions": package_versions,
        "case_count": len(observations),
        "matched_count": len(observations) - len(mismatches),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "cases": observations,
        "qualified": not mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = run()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
