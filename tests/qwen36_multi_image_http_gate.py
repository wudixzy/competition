#!/usr/bin/env python3
"""Streaming HTTP gate for a Qwen3.6 two-image service candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable

from quality_gate_api import Client
from qwen36_compat_http_gate import (
    _canonical_sha256,
    _request_json,
    _solid_png_data_url,
)
from qwen36_tool_http_gate import _stream_summary


Json = dict[str, Any]
SCHEMA = "qwen36-diagnostic-multi-image-http-gate-v1"
VERSION = 1
SEED = 20260728
CASE_NAMES = (
    "models_262144_contract",
    "stream_one_image_cold",
    "stream_two_images_cold",
    "stream_two_images_warm",
    "stream_two_images_reversed",
    "stream_two_images_reversed_warm",
    "post_request_health",
)
EXACT_FIELDS = (
    "semantic_output_sha256",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "has_content",
    "has_reasoning_content",
    "tool_call_count",
)


def _payload(colors: list[tuple[int, int, int]]) -> Json:
    content: list[Json] = [
        {
            "type": "image_url",
            "image_url": {"url": _solid_png_data_url(color)},
        }
        for color in colors
    ]
    # The repeated material gives the cache gate multiple complete blocks.
    content.append({
        "type": "text",
        "text": (
            "Synthetic multi-image cache and streaming validation material. "
            * 320
        ) + "Return one short token.",
    })
    return {
        "model": "llm",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 8,
        "temperature": 0,
        "seed": SEED,
        "thinking": False,
        "stream": True,
        "stream_options": {
            "include_usage": True,
            "continuous_usage_stats": False,
        },
    }


def _request_stream(
    client: Client,
    payload: Json,
    *,
    timeout_s: float,
    expected_status: int,
) -> Json:
    if expected_status != 200:
        status, response = client.post(payload, timeout=timeout_s)
        if status != expected_status:
            raise AssertionError(
                f"stream status {status}, expected {expected_status}; "
                f"response_sha256={_canonical_sha256(response)}")
        error = response.get("error") if isinstance(response, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        if not isinstance(message, str) or not message.strip():
            raise AssertionError("4xx response lacks a structured error message")
        return {
            "http_status": status,
            "error_fields": sorted(error),
            "error_message_sha256": hashlib.sha256(
                message.encode("utf-8")).hexdigest(),
            "response_sha256": _canonical_sha256(response),
        }

    status, response = client.stream(payload, timeout=timeout_s)
    if status != expected_status:
        raise AssertionError(
            f"stream status {status}, expected {expected_status}; "
            f"response_sha256={_canonical_sha256(response)}")
    summary = _stream_summary(response)
    summary["http_status"] = status
    return summary


def _same_generation(left: Json, right: Json) -> bool:
    return all(left.get(field) == right.get(field) for field in EXACT_FIELDS)


def _models_summary(client: Client) -> Json:
    response = client.models()
    models = response.get("data")
    if not isinstance(models, list) or not models:
        raise AssertionError("models response is empty")
    model = models[0]
    if model.get("id") != "llm":
        raise AssertionError("served model name differs")
    if model.get("max_model_len") != 262144:
        raise AssertionError(
            f"max_model_len is {model.get('max_model_len')!r}")
    return {
        "http_status": 200,
        "served_model": model.get("id"),
        "max_model_len": model.get("max_model_len"),
    }


def _health_summary(
    base: str,
    request_json: Callable[..., tuple[int, Json]],
) -> Json:
    status, response = request_json(
        "GET", f"{base.rstrip('/')}/health", timeout_s=30)
    if status != 200:
        raise AssertionError(f"health status {status}")
    return {
        "http_status": status,
        "response_sha256": _canonical_sha256(response),
    }


def run_gate(
    base: str,
    model_path: Path,
    timeout_s: float,
    expected_two_image_status: int,
    *,
    client: Client | None = None,
    request_json: Callable[..., tuple[int, Json]] = _request_json,
) -> Json:
    if expected_two_image_status not in (200, 400):
        raise ValueError("expected_two_image_status must be 200 or 400")
    client = client or Client(base)
    cases: list[Json] = []

    def run(name: str, function: Callable[[], Json]) -> Json | None:
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
            return None
        cases.append({
            "name": name,
            "ok": True,
            "elapsed_s": round(time.monotonic() - started, 4),
            "evidence": evidence,
        })
        return evidence

    run("models_262144_contract", lambda: _models_summary(client))

    one_image = run(
        "stream_one_image_cold",
        lambda: _request_stream(
            client,
            _payload([(255, 0, 0)]),
            timeout_s=timeout_s,
            expected_status=200,
        ),
    )
    two_cold = run(
        "stream_two_images_cold",
        lambda: _request_stream(
            client,
            _payload([(255, 0, 0), (0, 0, 255)]),
            timeout_s=timeout_s,
            expected_status=expected_two_image_status,
        ),
    )

    if expected_two_image_status == 200:

        def warm() -> Json:
            summary = _request_stream(
                client,
                _payload([(255, 0, 0), (0, 0, 255)]),
                timeout_s=timeout_s,
                expected_status=200,
            )
            if two_cold is None or not _same_generation(two_cold, summary):
                raise AssertionError(
                    "two-image cold/warm generation differs")
            if summary.get("cached_tokens", 0) <= 0:
                raise AssertionError(
                    "two-image warm request has no effective cache hit")
            summary["cold_generation_exact"] = True
            return summary

        run("stream_two_images_warm", warm)

        def reversed_images() -> Json:
            summary = _request_stream(
                client,
                _payload([(0, 0, 255), (255, 0, 0)]),
                timeout_s=timeout_s,
                expected_status=200,
            )
            summary["cache_isolation_deferred_to_trace"] = True
            return summary

        reversed_cold = run("stream_two_images_reversed", reversed_images)

        def reversed_warm() -> Json:
            summary = _request_stream(
                client,
                _payload([(0, 0, 255), (255, 0, 0)]),
                timeout_s=timeout_s,
                expected_status=200,
            )
            if (
                reversed_cold is None
                or not _same_generation(reversed_cold, summary)
            ):
                raise AssertionError(
                    "reversed two-image cold/warm generation differs")
            if summary.get("cached_tokens", 0) <= 0:
                raise AssertionError(
                    "reversed two-image warm request has no effective cache hit")
            summary["cold_generation_exact"] = True
            return summary

        run("stream_two_images_reversed_warm", reversed_warm)
    else:
        run(
            "stream_two_images_warm",
            lambda: {
                "skipped": True,
                "reason": "control_image_limit_one",
            },
        )
        run(
            "stream_two_images_reversed",
            lambda: {
                "skipped": True,
                "reason": "control_image_limit_one",
            },
        )
        run(
            "stream_two_images_reversed_warm",
            lambda: {
                "skipped": True,
                "reason": "control_image_limit_one",
            },
        )

    run(
        "post_request_health",
        lambda: _health_summary(base, request_json),
    )

    case_names = tuple(case["name"] for case in cases)
    qualified = (
        case_names == CASE_NAMES
        and all(case["ok"] for case in cases)
        and one_image is not None
        and two_cold is not None
    )
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "qualified": qualified,
        "case_count": len(cases),
        "base": base,
        "model_path": str(model_path.resolve()),
        "config": {
            "expected_two_image_status": expected_two_image_status,
            "max_tokens": 8,
            "seed": SEED,
            "stream": True,
            "temperature": 0,
            "thinking": False,
        },
        "cases": cases,
        "privacy": {
            "contains_raw_request": False,
            "contains_raw_response": False,
            "contains_image_url_or_bytes": False,
            "contains_prompt_or_generated_text": False,
            "contains_credentials": False,
            "synthetic_images_only": True,
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
        "--expected-two-image-status",
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
        args.expected_two_image_status,
    )
    _atomic_write(args.json_out, report)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
