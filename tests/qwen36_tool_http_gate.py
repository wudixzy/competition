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


Json = dict[str, Any]
SCHEMA = "qwen36-diagnostic-tool-http-gate-v1"
VERSION = 1
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
) -> Json:
    return {
        "model": "llm",
        "messages": messages,
        "tools": [_function_tool(strict=strict)],
        "tool_choice": tool_choice,
        "max_tokens": 8,
        "temperature": 0,
        "seed": SEED,
        "thinking": False,
    }


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
