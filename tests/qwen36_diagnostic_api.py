#!/usr/bin/env python3
"""Structural API gate for a reduced-depth Qwen3.6 diagnostic model.

This gate deliberately avoids semantic quality assertions. Reduced depth
changes model capability; the script only attests that important request
surfaces execute without protocol errors or process failure.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import time
from typing import Any, Callable
import urllib.error
import urllib.request
import zlib


Json = dict[str, Any]
SCHEMA = "qwen36-diagnostic-api-gate-v1"


def _request_json(
    method: str,
    url: str,
    payload: Json | None = None,
    *,
    timeout_s: float,
) -> tuple[int, Json]:
    body = None if payload is None else json.dumps(
        payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"} if body is not None else {}
    request = urllib.request.Request(
        url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read()
            parsed = json.loads(raw) if raw else {}
            return response.status, parsed
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {"body_sha256": hashlib.sha256(raw).hexdigest()}
        return error.code, parsed


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _response_summary(response: Json) -> Json:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AssertionError("response has no choices")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise AssertionError("response choice has no message")
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise AssertionError("response has no usage")
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(completion_tokens, int) or completion_tokens < 1:
        raise AssertionError("response generated no completion tokens")
    details = usage.get("prompt_tokens_details")
    cached_tokens = (
        details.get("cached_tokens", 0) if isinstance(details, dict) else 0
    )
    return {
        "message_sha256": _canonical_sha256(message),
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "has_content": isinstance(message.get("content"), str),
        "has_reasoning_content": isinstance(
            message.get("reasoning_content"), str),
        "tool_call_count": len(message.get("tool_calls") or []),
    }


def _post_chat(
    base: str,
    payload: Json,
    *,
    timeout_s: float,
    expected_status: int = 200,
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
    if expected_status != 200:
        return response, {
            "http_status": status,
            "response_sha256": _canonical_sha256(response),
        }
    summary = _response_summary(response)
    summary["http_status"] = status
    return response, summary


def _solid_png_data_url(rgb: tuple[int, int, int]) -> str:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", checksum)
        )

    width = height = 64
    scanline = b"\x00" + bytes(rgb) * width
    pixels = scanline * height
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    image = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(pixels, level=9))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(image).decode("ascii")


def _atomic_write(path: Path, value: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2,
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


def run_gate(base: str, model_path: Path, timeout_s: float) -> Json:
    cases: list[Json] = []

    def run(name: str, function: Callable[[], Json]) -> None:
        started = time.monotonic()
        try:
            evidence = function()
        except BaseException as error:  # Preserve diagnostic failure class.
            cases.append({
                "name": name,
                "ok": False,
                "elapsed_s": round(time.monotonic() - started, 4),
                "error": f"{type(error).__name__}: {error}"[:1000],
            })
        else:
            cases.append({
                "name": name,
                "ok": True,
                "elapsed_s": round(time.monotonic() - started, 4),
                "evidence": evidence,
            })

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

    def deterministic() -> Json:
        payload = {
            "model": "llm",
            "messages": [{
                "role": "user",
                "content": "Return one short token for a diagnostic request.",
            }],
            "max_tokens": 8,
            "temperature": 0,
            "seed": 20260727,
            "thinking": False,
        }
        first, first_summary = _post_chat(
            base, payload, timeout_s=timeout_s)
        second, second_summary = _post_chat(
            base, payload, timeout_s=timeout_s)
        if first["choices"][0]["message"] != second["choices"][0]["message"]:
            raise AssertionError("fixed greedy message changed across replay")
        if first["choices"][0].get(
                "finish_reason") != second["choices"][0].get("finish_reason"):
            raise AssertionError("finish_reason changed across replay")
        return {
            "cold": first_summary,
            "replay": second_summary,
            "exact_message_match": True,
        }

    def tool_message_surface() -> Json:
        _, summary = _post_chat(base, {
            "model": "llm",
            "messages": [{
                "role": "user",
                "content": "Answer directly. Do not call a tool.",
            }],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Diagnostic lookup.",
                    "parameters": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                    },
                },
            }],
            "tool_choice": "none",
            "max_tokens": 8,
            "temperature": 0,
            "seed": 20260727,
            "thinking": False,
        }, timeout_s=timeout_s)
        if summary["tool_call_count"] != 0:
            raise AssertionError("tool_choice=none produced a tool call")
        return summary

    def reasoning_surface() -> Json:
        _, summary = _post_chat(base, {
            "model": "llm",
            "messages": [{
                "role": "user",
                "content": "Think briefly, then provide a short answer.",
            }],
            "max_tokens": 16,
            "temperature": 0,
            "seed": 20260727,
            "thinking": True,
        }, timeout_s=timeout_s)
        return summary

    def structured_output() -> Json:
        response, summary = _post_chat(base, {
            "model": "llm",
            "messages": [{
                "role": "user",
                "content": 'Return a JSON object with integer field "value".',
            }],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "diagnostic_value",
                    "schema": {
                        "type": "object",
                        "properties": {"value": {"type": "integer"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                },
            },
            "max_tokens": 32,
            "temperature": 0,
            "seed": 20260727,
            "thinking": False,
        }, timeout_s=timeout_s)
        content = response["choices"][0]["message"].get("content")
        parsed = json.loads(content)
        if set(parsed) != {"value"} or not isinstance(parsed["value"], int):
            raise AssertionError("structured output does not match schema")
        return summary

    def multimodal_surface() -> Json:
        _, summary = _post_chat(base, {
            "model": "llm",
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _solid_png_data_url((255, 0, 0)),
                        },
                    },
                    {
                        "type": "text",
                        "text": "Provide a short diagnostic response.",
                    },
                ],
            }],
            "max_tokens": 8,
            "temperature": 0,
            "seed": 20260727,
            "thinking": False,
        }, timeout_s=max(timeout_s, 360))
        return summary

    def invalid_request() -> Json:
        _, summary = _post_chat(
            base,
            {"model": "llm", "messages": []},
            timeout_s=30,
            expected_status=400,
        )
        return summary

    run("models_262144_contract", models)
    run("deterministic_replay", deterministic)
    run("tool_message_surface", tool_message_surface)
    run("reasoning_surface", reasoning_surface)
    run("structured_output_surface", structured_output)
    run("multimodal_surface", multimodal_surface)
    run("invalid_empty_messages_400", invalid_request)

    return {
        "schema": SCHEMA,
        "version": 1,
        "base": base,
        "model_path": str(model_path.resolve()),
        "qualified": all(case["ok"] for case in cases),
        "case_count": len(cases),
        "cases": cases,
        "semantic_quality_evaluated": False,
        "production_promotion_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the reduced-depth Qwen3.6 structural API gate")
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=240)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_gate(args.base, args.model_path, args.timeout_s)
    _atomic_write(args.json_out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
