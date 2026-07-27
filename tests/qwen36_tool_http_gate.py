#!/usr/bin/env python3
"""HTTP gate for Qwen3.6 tool request compatibility."""

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
    _same_generation,
)
from quality_gate_api import Client


Json = dict[str, Any]
SCHEMA = "qwen36-diagnostic-tool-http-gate-v2"
VERSION = 2
SEED = 20260728
USER_TEXT = "Return one short token for this synthetic diagnostic request."


def _post_chat(
    base: str,
    payload: Json,
    *,
    timeout_s: float,
    expected_status: int,
) -> tuple[Json, Json]:
    status, response = _request_json(
        "POST",
        f"{base.rstrip('/')}/v1/chat/completions",
        payload,
        timeout_s=timeout_s,
    )
    if status != expected_status:
        raise AssertionError(
            f"chat status {status}, expected {expected_status}; "
            f"response_sha256={_canonical_sha256(response)}")
    if status != 200:
        return response, {
            "http_status": status,
            "response_sha256": _canonical_sha256(response),
        }
    summary = _message_summary(response)
    summary["http_status"] = status
    return response, summary


def _request_stream(
    base: str,
    payload: Json,
    timeout_s: float,
) -> tuple[int, Json]:
    return Client(base).stream(payload, timeout=timeout_s)


def _stream_summary(stream: Json) -> Json:
    if not isinstance(stream, dict):
        raise AssertionError("parsed SSE stream must be an object")
    if stream.get("done") != 1:
        raise AssertionError("SSE stream must contain exactly one DONE event")
    if stream.get("usage_blocks") != 1:
        raise AssertionError("SSE stream must contain one final usage block")
    chunks = stream.get("chunks")
    if not isinstance(chunks, int) or chunks < 2:
        raise AssertionError("SSE stream contains too few JSON chunks")

    finish_reasons = stream.get("finish_reasons")
    if not isinstance(finish_reasons, list) or len(finish_reasons) != 1:
        raise AssertionError(
            "SSE stream must contain one terminal finish_reason")
    finish_reason = finish_reasons[0]
    if not isinstance(finish_reason, str) or not finish_reason:
        raise AssertionError("SSE finish_reason is invalid")

    usage = stream.get("usage")
    if not isinstance(usage, dict):
        raise AssertionError("SSE stream has no usage object")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if (
        not isinstance(prompt_tokens, int)
        or isinstance(prompt_tokens, bool)
        or prompt_tokens < 0
    ):
        raise AssertionError("SSE prompt_tokens is invalid")
    if (
        not isinstance(completion_tokens, int)
        or isinstance(completion_tokens, bool)
        or completion_tokens < 1
    ):
        raise AssertionError("SSE request generated no completion tokens")
    if total_tokens != prompt_tokens + completion_tokens:
        raise AssertionError("SSE total_tokens is inconsistent")

    details = usage.get("prompt_tokens_details")
    cached_tokens = (
        details.get("cached_tokens", 0) if isinstance(details, dict) else 0
    )
    if (
        not isinstance(cached_tokens, int)
        or isinstance(cached_tokens, bool)
        or cached_tokens < 0
        or cached_tokens > prompt_tokens
    ):
        raise AssertionError("SSE cached_tokens is invalid")

    content = stream.get("content")
    reasoning = stream.get("reasoning_content")
    tool_calls = stream.get("tool_calls")
    if not isinstance(content, str):
        raise AssertionError("reconstructed SSE content is invalid")
    if not isinstance(reasoning, str):
        raise AssertionError(
            "reconstructed SSE reasoning_content is invalid")
    if not isinstance(tool_calls, list):
        raise AssertionError("reconstructed SSE tool_calls are invalid")
    if not (content or reasoning or tool_calls):
        raise AssertionError("SSE stream has no generated semantic output")

    semantic = {
        "content": content,
        "reasoning_content": reasoning,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
    }
    return {
        "semantic_output_sha256": _canonical_sha256(semantic),
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "has_content": bool(content),
        "has_reasoning_content": bool(reasoning),
        "tool_call_count": len(tool_calls),
        "chunks": chunks,
        "done": 1,
        "usage_blocks": 1,
    }


def _post_stream_chat(
    base: str,
    payload: Json,
    *,
    timeout_s: float,
    expected_status: int,
    stream_request: Callable[[str, Json, float], tuple[int, Json]],
) -> tuple[Json, Json]:
    if expected_status != 200:
        return _post_chat(
            base,
            payload,
            timeout_s=timeout_s,
            expected_status=expected_status,
        )

    status, stream = stream_request(base, payload, timeout_s)
    if status != expected_status:
        raise AssertionError(
            f"stream chat status {status}, expected {expected_status}; "
            f"response_sha256={_canonical_sha256(stream)}")
    summary = _stream_summary(stream)
    summary["http_status"] = status
    return stream, summary


def _same_stream_generation(left: Json, right: Json) -> bool:
    fields = (
        "semantic_output_sha256",
        "finish_reason",
        "prompt_tokens",
        "completion_tokens",
        "has_content",
        "has_reasoning_content",
        "tool_call_count",
    )
    return all(left.get(field) == right.get(field) for field in fields)


def _function_tool(*, strict: bool | None = None) -> Json:
    function: Json = {
        "name": "synthetic_lookup",
        "description": "Return a synthetic diagnostic value.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
    }
    if strict is not None:
        function["strict"] = strict
    return {"type": "function", "function": function}


def _payload(
    messages: list[Json],
    *,
    strict: bool | None = None,
    tool_choice: str = "none",
    stream: bool = False,
) -> Json:
    payload = {
        "model": "llm",
        "messages": messages,
        "tools": [_function_tool(strict=strict)],
        "tool_choice": tool_choice,
        "max_tokens": 8,
        "temperature": 0,
        "seed": SEED,
        "thinking": False,
    }
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {
            "include_usage": True,
            "continuous_usage_stats": False,
        }
    return payload


def _history(arguments: Any) -> list[Json]:
    return [
        {"role": "user", "content": "Look up the synthetic key."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_synthetic_1",
                "type": "function",
                "function": {
                    "name": "synthetic_lookup",
                    "arguments": arguments,
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call_synthetic_1",
            "content": "synthetic result",
        },
        {"role": "user", "content": "Continue with one short token."},
    ]


def run_gate(
    base: str,
    model_path: Path,
    timeout_s: float,
    strict_false_expected_status: int,
    object_history_expected_status: int,
    stream_request: Callable[
        [str, Json, float], tuple[int, Json]
    ] = _request_stream,
) -> Json:
    cases: list[Json] = []

    def run(name: str, function: Callable[[], Json]) -> Json | None:
        started = time.monotonic()
        try:
            evidence = function()
        except BaseException as error:
            cases.append({
                "name": name,
                "ok": False,
                "elapsed_s": round(time.monotonic() - started, 4),
                "error_type": type(error).__name__,
                "error_sha256": hashlib.sha256(
                    str(error).encode("utf-8")).hexdigest(),
            })
            return None
        cases.append({
            "name": name,
            "ok": True,
            "elapsed_s": round(time.monotonic() - started, 4),
            "evidence": evidence,
        })
        return evidence

    def models() -> Json:
        status, response = _request_json(
            "GET", f"{base.rstrip('/')}/v1/models", timeout_s=30)
        if status != 200:
            raise AssertionError(f"models status {status}")
        models_value = response.get("data")
        if not isinstance(models_value, list) or not models_value:
            raise AssertionError("models response is empty")
        model = models_value[0]
        if model.get("max_model_len") != 262144:
            raise AssertionError(
                f"max_model_len is {model.get('max_model_len')!r}")
        return {
            "http_status": status,
            "served_model": model.get("id"),
            "max_model_len": model.get("max_model_len"),
        }

    run("models_262144_contract", models)

    default_tool_response: Json | None = None

    def default_tool() -> Json:
        nonlocal default_tool_response
        default_tool_response, summary = _post_chat(
            base,
            _payload([{"role": "user", "content": USER_TEXT}]),
            timeout_s=timeout_s,
            expected_status=200,
        )
        return summary

    run("function_tool_default", default_tool)

    def strict_false() -> Json:
        response, summary = _post_chat(
            base,
            _payload(
                [{"role": "user", "content": USER_TEXT}],
                strict=False,
            ),
            timeout_s=timeout_s,
            expected_status=strict_false_expected_status,
        )
        if strict_false_expected_status == 200:
            if default_tool_response is None:
                raise AssertionError("default tool request did not run")
            if not _same_generation(default_tool_response, response):
                raise AssertionError(
                    "strict=false changed deterministic output")
            summary["default_generation_exact"] = True
        return summary

    run("function_tool_strict_false", strict_false)

    string_history_response: Json | None = None

    def string_history() -> Json:
        nonlocal string_history_response
        string_history_response, summary = _post_chat(
            base,
            _payload(_history('{"key":"synthetic"}')),
            timeout_s=timeout_s,
            expected_status=200,
        )
        return summary

    run("tool_arguments_json_string", string_history)

    def object_history() -> Json:
        response, summary = _post_chat(
            base,
            _payload(_history({"key": "synthetic"})),
            timeout_s=timeout_s,
            expected_status=object_history_expected_status,
        )
        if object_history_expected_status == 200:
            if string_history_response is None:
                raise AssertionError("string tool history did not run")
            if not _same_generation(string_history_response, response):
                raise AssertionError(
                    "object tool arguments changed deterministic output")
            summary["string_generation_exact"] = True
        return summary

    run("tool_arguments_json_object", object_history)

    default_stream_summary: Json | None = None

    def default_stream() -> Json:
        nonlocal default_stream_summary
        _, default_stream_summary = _post_stream_chat(
            base,
            _payload(
                [{"role": "user", "content": USER_TEXT}],
                stream=True,
            ),
            timeout_s=timeout_s,
            expected_status=200,
            stream_request=stream_request,
        )
        return default_stream_summary

    run("stream_function_tool_default", default_stream)

    def strict_false_stream() -> Json:
        _, summary = _post_stream_chat(
            base,
            _payload(
                [{"role": "user", "content": USER_TEXT}],
                strict=False,
                stream=True,
            ),
            timeout_s=timeout_s,
            expected_status=strict_false_expected_status,
            stream_request=stream_request,
        )
        if strict_false_expected_status == 200:
            if default_stream_summary is None:
                raise AssertionError("default stream request did not run")
            if not _same_stream_generation(
                    default_stream_summary, summary):
                raise AssertionError(
                    "stream strict=false changed deterministic output")
            summary["default_stream_generation_exact"] = True
        return summary

    run("stream_function_tool_strict_false", strict_false_stream)

    string_stream_summary: Json | None = None

    def string_history_stream() -> Json:
        nonlocal string_stream_summary
        _, string_stream_summary = _post_stream_chat(
            base,
            _payload(
                _history('{"key":"synthetic"}'),
                stream=True,
            ),
            timeout_s=timeout_s,
            expected_status=200,
            stream_request=stream_request,
        )
        return string_stream_summary

    run("stream_tool_arguments_json_string", string_history_stream)

    def object_history_stream() -> Json:
        _, summary = _post_stream_chat(
            base,
            _payload(
                _history({"key": "synthetic"}),
                stream=True,
            ),
            timeout_s=timeout_s,
            expected_status=object_history_expected_status,
            stream_request=stream_request,
        )
        if object_history_expected_status == 200:
            if string_stream_summary is None:
                raise AssertionError("string-history stream did not run")
            if not _same_stream_generation(
                    string_stream_summary, summary):
                raise AssertionError(
                    "stream object arguments changed deterministic output")
            summary["string_stream_generation_exact"] = True
        return summary

    run("stream_tool_arguments_json_object", object_history_stream)

    def invalid_json_history() -> Json:
        _, summary = _post_chat(
            base,
            _payload(_history("{invalid")),
            timeout_s=timeout_s,
            expected_status=400,
        )
        return summary

    run("tool_arguments_invalid_json_400", invalid_json_history)

    def strict_true() -> Json:
        _, summary = _post_chat(
            base,
            _payload(
                [{"role": "user", "content": USER_TEXT}],
                strict=True,
            ),
            timeout_s=timeout_s,
            expected_status=400,
        )
        return summary

    run("function_tool_strict_true_400", strict_true)

    def required_tool_choice() -> Json:
        _, summary = _post_chat(
            base,
            _payload(
                [{"role": "user", "content": USER_TEXT}],
                tool_choice="required",
            ),
            timeout_s=timeout_s,
            expected_status=400,
        )
        return summary

    run("tool_choice_required_400", required_tool_choice)

    def post_error_health() -> Json:
        status, response = _request_json(
            "GET", f"{base.rstrip('/')}/health", timeout_s=30)
        if status != 200:
            raise AssertionError(f"health status {status}")
        return {
            "http_status": status,
            "response_sha256": _canonical_sha256(response),
        }

    run("post_4xx_health", post_error_health)

    stream_case_names = (
        "stream_function_tool_default",
        "stream_function_tool_strict_false",
        "stream_tool_arguments_json_string",
        "stream_tool_arguments_json_object",
    )
    stream_cases = {
        case["name"]: case
        for case in cases
        if case["name"] in stream_case_names
    }
    strict_stream_evidence = stream_cases.get(
        "stream_function_tool_strict_false", {}).get("evidence", {})
    object_stream_evidence = stream_cases.get(
        "stream_tool_arguments_json_object", {}).get("evidence", {})
    streaming_contract = {
        "qualified": (
            len(stream_cases) == len(stream_case_names)
            and all(case["ok"] for case in stream_cases.values())
        ),
        "case_count": len(stream_cases),
        "successful_sse_count": sum(
            case.get("evidence", {}).get("http_status") == 200
            for case in stream_cases.values()
        ),
        "accepted_equivalence_qualified": (
            strict_false_expected_status == 200
            and object_history_expected_status == 200
            and strict_stream_evidence.get(
                "default_stream_generation_exact") is True
            and object_stream_evidence.get(
                "string_stream_generation_exact") is True
        ),
    }

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "qualified": all(case["ok"] for case in cases),
        "case_count": len(cases),
        "base": base,
        "model_path": str(model_path.resolve()),
        "config": {
            "seed": SEED,
            "temperature": 0,
            "max_tokens": 8,
            "strict_false_expected_status":
                strict_false_expected_status,
            "object_history_expected_status":
                object_history_expected_status,
        },
        "cases": cases,
        "streaming_contract": streaming_contract,
        "privacy": {
            "contains_raw_request": False,
            "contains_raw_response": False,
            "contains_tool_schema": False,
            "contains_tool_arguments": False,
            "contains_credentials": False,
            "synthetic_inputs_only": True,
        },
        "semantic_quality_evaluated": False,
        "full_model_evaluated": False,
        "production_promotion_authorized": False,
    }


def _atomic_write(path: Path, value: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=300)
    parser.add_argument(
        "--strict-false-expected-status",
        type=int,
        choices=(200, 400),
        required=True,
    )
    parser.add_argument(
        "--object-history-expected-status",
        type=int,
        choices=(200, 400),
        required=True,
    )
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(argv)

    report = run_gate(
        args.base,
        args.model_path,
        args.timeout_s,
        args.strict_false_expected_status,
        args.object_history_expected_status,
    )
    _atomic_write(args.json_out, report)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
