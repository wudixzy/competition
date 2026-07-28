#!/usr/bin/env python3
"""Single-GPU HTTP gate for valid Qwen3.6 tool-choice modes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable

from qwen36_compat_http_gate import (
    _canonical_sha256,
    _message_summary,
    _request_json,
)
from qwen36_tool_http_gate import _stream_summary


Json = dict[str, Any]
SCHEMA = "qwen36-diagnostic-tool-choice-http-gate-v1"
VERSION = 1
SEED = 20260728
TOOL_NAME = "get_weather"
USER_TEXT = "Call get_weather to query the weather in Beijing."
_OMITTED = object()


def _post_chat(
    base: str,
    payload: Json,
    *,
    timeout_s: float,
) -> tuple[Json, Json]:
    status, response = _request_json(
        "POST",
        f"{base.rstrip('/')}/v1/chat/completions",
        payload,
        timeout_s=timeout_s,
    )
    if status != 200:
        raise AssertionError(
            f"chat status {status}, expected 200; "
            f"response_sha256={_canonical_sha256(response)}")
    summary = _message_summary(response)
    summary["http_status"] = status
    return response, summary


def _post_stream_chat(
    base: str,
    payload: Json,
    *,
    timeout_s: float,
    request_stream: Callable[[str, Json, float], tuple[int, Json]],
) -> tuple[Json, Json]:
    status, stream = request_stream(base, payload, timeout_s)
    if status != 200:
        raise AssertionError(
            f"stream chat status {status}, expected 200; "
            f"response_sha256={_canonical_sha256(stream)}")
    summary = _stream_summary(stream)
    summary["http_status"] = status
    return stream, summary


def _tool() -> Json:
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Return weather for one city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }


def _payload(
    tool_choice: object,
    *,
    stream: bool,
) -> Json:
    payload: Json = {
        "model": "llm",
        "messages": [{"role": "user", "content": USER_TEXT}],
        "tools": [_tool()],
        "max_tokens": 128,
        "temperature": 0,
        "seed": SEED,
        "thinking": False,
    }
    if tool_choice is not _OMITTED:
        payload["tool_choice"] = tool_choice
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {
            "include_usage": True,
            "continuous_usage_stats": False,
        }
    return payload


def _arguments(value: Any) -> Json:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise AssertionError(
                "tool arguments are not valid JSON") from error
    if not isinstance(value, dict):
        raise AssertionError("tool arguments are not a JSON object")
    city = value.get("city")
    if not isinstance(city, str) or not city.strip():
        raise AssertionError("tool city argument is missing")
    normalized = city.casefold()
    if "beijing" not in normalized and "北京" not in city:
        raise AssertionError("tool city argument differs")
    return value


def _tool_semantics_from_response(response: Json) -> Json:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise AssertionError("tool response must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise AssertionError("tool choice is not an object")
    if choice.get("finish_reason") != "tool_calls":
        raise AssertionError("tool response did not finish as tool_calls")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise AssertionError("tool response message is missing")
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise AssertionError("tool response must contain exactly one call")
    call = calls[0]
    if (
        not isinstance(call, dict)
        or not isinstance(call.get("id"), str)
        or not call["id"]
        or call.get("type") != "function"
    ):
        raise AssertionError("tool call identity is invalid")
    function = call.get("function")
    if (
        not isinstance(function, dict)
        or function.get("name") != TOOL_NAME
    ):
        raise AssertionError("tool function name differs")
    arguments = _arguments(function.get("arguments"))
    content = message.get("content")
    reasoning = message.get(
        "reasoning_content", message.get("reasoning"))
    if content is not None and not isinstance(content, str):
        raise AssertionError("tool response content is invalid")
    if reasoning is not None and not isinstance(reasoning, str):
        raise AssertionError("tool response reasoning is invalid")
    semantic = {
        "finish_reason": "tool_calls",
        "content": content or "",
        "reasoning_content": reasoning or "",
        "tool_calls": [{"name": TOOL_NAME, "arguments": arguments}],
    }
    return {
        "semantic_sha256": _canonical_sha256(semantic),
        "finish_reason_tool_calls": True,
        "tool_call_count": 1,
        "tool_name_sha256": hashlib.sha256(
            TOOL_NAME.encode("utf-8")).hexdigest(),
        "arguments_valid_json_object": True,
        "argument_semantics_valid": True,
    }


def _tool_semantics_from_stream(stream: Json) -> Json:
    summary = _stream_summary(stream)
    if summary.get("finish_reason") != "tool_calls":
        raise AssertionError("stream did not finish as tool_calls")
    calls = stream.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise AssertionError("stream must contain exactly one tool call")
    call = calls[0]
    if not isinstance(call, dict) or call.get("name") != TOOL_NAME:
        raise AssertionError("streamed tool function name differs")
    arguments = _arguments(call.get("arguments"))
    semantic = {
        "finish_reason": "tool_calls",
        "content": stream.get("content") or "",
        "reasoning_content": stream.get("reasoning_content") or "",
        "tool_calls": [{"name": TOOL_NAME, "arguments": arguments}],
    }
    return {
        **summary,
        "http_status": 200,
        "semantic_sha256": _canonical_sha256(semantic),
        "finish_reason_tool_calls": True,
        "tool_call_count": 1,
        "tool_name_sha256": hashlib.sha256(
            TOOL_NAME.encode("utf-8")).hexdigest(),
        "arguments_valid_json_object": True,
        "argument_semantics_valid": True,
    }


def run_gate(
    base: str,
    model_path: Path,
    timeout_s: float,
    *,
    request_stream: Callable[
        [str, Json, float], tuple[int, Json]
    ],
) -> Json:
    cases: list[Json] = []
    semantics: dict[str, Json] = {}

    def run(name: str, function: Callable[[], Json]) -> None:
        started = time.monotonic()
        try:
            evidence = function()
        except Exception as error:
            cases.append({
                "name": name,
                "ok": False,
                "elapsed_s": round(time.monotonic() - started, 4),
                "error_type": type(error).__name__,
                "error_sha256": hashlib.sha256(
                    str(error).encode("utf-8")).hexdigest(),
            })
            return
        semantics[name] = evidence
        cases.append({
            "name": name,
            "ok": True,
            "elapsed_s": round(time.monotonic() - started, 4),
            "evidence": evidence,
        })

    modes = (
        ("omitted", _OMITTED),
        ("auto", "auto"),
        (
            "named",
            {"type": "function", "function": {"name": TOOL_NAME}},
        ),
    )
    for label, choice in modes:
        def nonstream(
            selected: object = choice,
        ) -> Json:
            response, summary = _post_chat(
                base,
                _payload(selected, stream=False),
                timeout_s=timeout_s,
            )
            return {
                **summary,
                **_tool_semantics_from_response(response),
            }

        run(f"tool_choice_{label}_nonstream", nonstream)

    for label, choice in modes:
        def streaming(
            selected: object = choice,
        ) -> Json:
            stream, _ = _post_stream_chat(
                base,
                _payload(selected, stream=True),
                timeout_s=timeout_s,
                request_stream=request_stream,
            )
            return _tool_semantics_from_stream(stream)

        run(f"tool_choice_{label}_stream", streaming)

    def health() -> Json:
        status, response = _request_json(
            "GET", f"{base.rstrip('/')}/health", timeout_s=30)
        if status != 200:
            raise AssertionError(f"health status {status}")
        return {
            "http_status": status,
            "response_sha256": _canonical_sha256(response),
        }

    run("post_tool_choice_health", health)

    generation_names = tuple(
        f"tool_choice_{label}_{transport}"
        for transport in ("nonstream", "stream")
        for label in ("omitted", "auto", "named")
    )
    digest = {
        name: semantics.get(name, {}).get("semantic_sha256")
        for name in generation_names
    }
    transport_exact = all(
        digest[f"tool_choice_{label}_nonstream"] is not None
        and digest[f"tool_choice_{label}_nonstream"]
        == digest[f"tool_choice_{label}_stream"]
        for label in ("omitted", "auto", "named")
    )
    omitted_auto_exact = all(
        digest[f"tool_choice_omitted_{transport}"] is not None
        and digest[f"tool_choice_omitted_{transport}"]
        == digest[f"tool_choice_auto_{transport}"]
        for transport in ("nonstream", "stream")
    )
    report = {
        "schema": SCHEMA,
        "version": VERSION,
        "qualified": (
            len(cases) == 7
            and all(case["ok"] for case in cases)
            and transport_exact
            and omitted_auto_exact
        ),
        "case_count": len(cases),
        "cases": cases,
        "config": {
            "model_path": str(model_path.resolve()),
            "seed": SEED,
            "streaming": True,
            "tool_choice_modes": ["omitted", "auto", "named"],
        },
        "checks": {
            "all_valid_modes_http_200": (
                len(semantics) == 7
                and all(
                    semantics[name].get("http_status") == 200
                    for name in generation_names
                )
            ),
            "nonstream_stream_semantics_exact": transport_exact,
            "omitted_auto_semantics_exact": omitted_auto_exact,
            "tool_calls_structurally_valid": all(
                semantics.get(name, {}).get(
                    "arguments_valid_json_object") is True
                and semantics.get(name, {}).get(
                    "argument_semantics_valid") is True
                and semantics.get(name, {}).get(
                    "finish_reason_tool_calls") is True
                for name in generation_names
            ),
            "post_tool_choice_health_200": semantics.get(
                "post_tool_choice_health", {}).get("http_status") == 200,
        },
        "privacy": {
            "contains_prompt": False,
            "contains_tool_name": False,
            "contains_arguments": False,
            "contains_model_output": False,
        },
        "strict_true_evaluated": False,
        "required_tool_choice_evaluated": False,
        "full_model_evaluated": False,
        "semantic_quality_evaluated": False,
        "production_promotion_authorized": False,
    }
    return report


def _atomic_write(path: Path, value: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(
                value,
                output,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run valid tool-choice modes against a diagnostic service")
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=300)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be positive")

    from qwen36_tool_http_gate import _request_stream

    report = run_gate(
        args.base,
        args.model_path,
        args.timeout_s,
        request_stream=_request_stream,
    )
    _atomic_write(args.json_out, report)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
