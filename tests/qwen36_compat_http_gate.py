#!/usr/bin/env python3
"""HTTP compatibility gate for the reduced-depth Qwen3.6 diagnostic model."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
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
SCHEMA = "qwen36-diagnostic-compat-http-gate-v1"
VERSION = 1
SEED = 20260728
SYSTEM_TEXT = "synthetic alpha\nsynthetic beta\n\nsynthetic gamma"
USER_TEXT = "Return one short token for this synthetic diagnostic request."


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _request_json(
    method: str,
    url: str,
    payload: Json | None = None,
    *,
    timeout_s: float,
) -> tuple[int, Json]:
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=True).encode("ascii")
        headers["Content-Type"] = "application/json"
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


def _message_summary(response: Json) -> Json:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise AssertionError("response must contain exactly one choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise AssertionError("response choice has no message object")
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise AssertionError("response has no usage object")
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


def _image_messages(count: int) -> list[Json]:
    colors = (
        (255, 0, 0),
        (0, 0, 255),
        (0, 255, 0),
    )
    parts: list[Json] = []
    for index in range(count):
        parts.append({
            "type": "image_url",
            "image_url": {
                "url": _solid_png_data_url(colors[index % len(colors)]),
            },
        })
    parts.append({
        "type": "text",
        "text": "Provide a short synthetic diagnostic response.",
    })
    return [{"role": "user", "content": parts}]


def _tool_contract() -> list[Json]:
    return [{
        "type": "function",
        "function": {
            "name": "synthetic_lookup",
            "description": "A synthetic diagnostic lookup.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
                "additionalProperties": False,
            },
        },
    }]


def _payload(messages: list[Json]) -> Json:
    return {
        "model": "llm",
        "messages": messages,
        "tools": _tool_contract(),
        "tool_choice": "none",
        "max_tokens": 8,
        "temperature": 0,
        "seed": SEED,
        "thinking": False,
    }


def _same_generation(left: Json, right: Json) -> bool:
    return (
        left.get("choices", [{}])[0].get("message")
        == right.get("choices", [{}])[0].get("message")
        and left.get("choices", [{}])[0].get("finish_reason")
        == right.get("choices", [{}])[0].get("finish_reason")
        and left.get("usage", {}).get("prompt_tokens")
        == right.get("usage", {}).get("prompt_tokens")
        and left.get("usage", {}).get("completion_tokens")
        == right.get("usage", {}).get("completion_tokens")
    )


def run_gate(
    base: str,
    model_path: Path,
    timeout_s: float,
    system_parts_expected_status: int,
    image_limit: int,
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

    canonical_response: Json | None = None

    def canonical_system() -> Json:
        nonlocal canonical_response
        canonical_response, summary = _post_chat(
            base,
            _payload([
                {"role": "system", "content": SYSTEM_TEXT},
                {"role": "user", "content": USER_TEXT},
            ]),
            timeout_s=timeout_s,
            expected_status=200,
        )
        if summary["tool_call_count"] != 0:
            raise AssertionError("tool_choice=none produced a tool call")
        return summary

    run("canonical_system_string", canonical_system)

    def single_system_parts() -> Json:
        response, summary = _post_chat(
            base,
            _payload([
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "synthetic alpha"},
                        {
                            "type": "text",
                            "text": "synthetic beta\n\nsynthetic gamma",
                        },
                    ],
                },
                {"role": "user", "content": USER_TEXT},
            ]),
            timeout_s=timeout_s,
            expected_status=system_parts_expected_status,
        )
        if system_parts_expected_status == 200:
            if canonical_response is None:
                raise AssertionError("canonical system request did not run")
            if not _same_generation(canonical_response, response):
                raise AssertionError(
                    "single system text-parts changed deterministic output")
            summary["canonical_generation_exact"] = True
        return summary

    run("single_system_text_parts", single_system_parts)

    def multiple_system_parts() -> Json:
        response, summary = _post_chat(
            base,
            _payload([
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "synthetic alpha"},
                        {"type": "text", "text": "synthetic beta"},
                    ],
                },
                {"role": "system", "content": "synthetic gamma"},
                {"role": "user", "content": USER_TEXT},
            ]),
            timeout_s=timeout_s,
            expected_status=system_parts_expected_status,
        )
        if system_parts_expected_status == 200:
            if canonical_response is None:
                raise AssertionError("canonical system request did not run")
            if not _same_generation(canonical_response, response):
                raise AssertionError(
                    "multi-system text-parts changed deterministic output")
            summary["canonical_generation_exact"] = True
        return summary

    run("multiple_system_text_parts", multiple_system_parts)

    def one_image() -> Json:
        _, summary = _post_chat(
            base,
            _payload(_image_messages(1)),
            timeout_s=max(timeout_s, 360),
            expected_status=200,
        )
        return summary

    run("one_image", one_image)

    def image_at_limit_replay() -> Json:
        payload = _payload(_image_messages(image_limit))
        first, first_summary = _post_chat(
            base,
            payload,
            timeout_s=max(timeout_s, 360),
            expected_status=200,
        )
        second, second_summary = _post_chat(
            base,
            payload,
            timeout_s=max(timeout_s, 360),
            expected_status=200,
        )
        if not _same_generation(first, second):
            raise AssertionError(
                "fixed greedy image request changed across replay")
        return {
            "image_count": image_limit,
            "first": first_summary,
            "replay": second_summary,
            "exact_generation_match": True,
        }

    run("image_at_limit_replay", image_at_limit_replay)

    def over_limit_image() -> Json:
        _, summary = _post_chat(
            base,
            _payload(_image_messages(image_limit + 1)),
            timeout_s=max(timeout_s, 360),
            expected_status=400,
        )
        summary["image_count"] = image_limit + 1
        return summary

    run("over_limit_image_400", over_limit_image)

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
            "system_parts_expected_status": system_parts_expected_status,
            "image_limit": image_limit,
        },
        "cases": cases,
        "privacy": {
            "contains_raw_request": False,
            "contains_raw_response": False,
            "contains_image_url_or_bytes": False,
            "contains_tool_schema": False,
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
        "--system-parts-expected-status",
        type=int,
        choices=(200, 400),
        required=True,
    )
    parser.add_argument(
        "--image-limit", type=int, choices=(1, 2), required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("--timeout-s must be finite and greater than 0")
    report = run_gate(
        args.base,
        args.model_path,
        args.timeout_s,
        args.system_parts_expected_status,
        args.image_limit,
    )
    _atomic_write(args.json_out, report)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
