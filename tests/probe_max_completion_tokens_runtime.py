#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SCHEMA = "bi100-max-completion-tokens-runtime-probe-v1"
PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAusB9Wl2nCEAAAAASUVORK5CYII="
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _output_kind(value: Any) -> str:
    enum_value = getattr(value, "value", None)
    return str(enum_value if enum_value is not None else value)


def _contains_multimodal_message(messages: Any) -> bool:
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        if any(
                isinstance(part, dict)
                and part.get("type") == "image_url"
                for part in content):
            return True
    return False


async def _asgi_post_json(
        app: Any, path: str, payload: dict[str, Any]) -> tuple[int, bytes]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    messages: list[dict[str, Any]] = []
    request_sent = False
    receive_blocker = asyncio.Event()

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {
                "type": "http.request",
                "body": body,
                "more_body": False,
            }
        await receive_blocker.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"runtime.local"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("runtime.local", 80),
        "state": {},
        "extensions": {},
    }
    await app(scope, receive, send)

    starts = [
        message for message in messages
        if message["type"] == "http.response.start"
    ]
    if len(starts) != 1:
        raise RuntimeError(
            f"expected one ASGI response start, observed {len(starts)}")
    response_body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    return int(starts[0]["status"]), response_body


async def run_probe() -> dict[str, Any]:
    from vllm.entrypoints.openai import api_server
    from vllm.entrypoints.openai.protocol import (
        ChatCompletionRequest,
        ChatCompletionResponse,
        ChatCompletionResponseChoice,
        ChatMessage,
        ErrorResponse,
        UsageInfo,
    )

    records: list[dict[str, Any]] = []

    class FakeServingChat:

        @staticmethod
        def create_error_response(message: str) -> ErrorResponse:
            return ErrorResponse(
                message=message,
                type="BadRequestError",
                param=None,
                code=400,
            )

        async def create_chat_completion(self, request, _raw_request):
            record = {
                "stream": bool(request.stream),
                "has_tools": bool(request.tools),
                "has_multimodal": _contains_multimodal_message(
                    request.messages),
                "thinking_present": request.thinking is not None,
            }
            try:
                if request.use_beam_search:
                    sampling = request.to_beam_search_params(4096)
                    record["output_kind"] = "beam"
                else:
                    sampling = request.to_sampling_params(4096)
                    record["output_kind"] = _output_kind(
                        sampling.output_kind)
                record["sampling_max_tokens"] = sampling.max_tokens
            except (TypeError, ValueError) as exc:
                record["sampling_error_type"] = type(exc).__name__
                records.append(record)
                return self.create_error_response(str(exc))

            records.append(record)
            if request.stream:
                async def stream():
                    yield "data: {\"choices\":[]}\n\n"
                    yield "data: [DONE]\n\n"

                return stream()

            return ChatCompletionResponse(
                model=request.model,
                choices=[ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="ok"),
                    finish_reason="stop",
                )],
                usage=UsageInfo(
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                ),
            )

    args = SimpleNamespace(
        disable_fastapi_docs=True,
        root_path=None,
        allowed_origins=["*"],
        allow_credentials=False,
        allowed_methods=["*"],
        allowed_headers=["*"],
        api_key=None,
        middleware=[],
    )
    app = api_server.build_app(args)
    fake_chat = FakeServingChat()
    app.state.openai_serving_chat = fake_chat

    base_payload = {
        "model": "llm",
        "messages": [{"role": "user", "content": "synthetic"}],
        "temperature": 0,
    }
    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "synthetic",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
            },
        },
    }
    cases = [
        {
            "name": "completion_only_nonstream",
            "payload": {**base_payload, "max_completion_tokens": 17},
            "expected_status": 200,
            "expected_budget": 17,
            "expected_enters_serving": True,
        },
        {
            "name": "completion_only_stream",
            "payload": {
                **base_payload,
                "max_completion_tokens": 19,
                "stream": True,
            },
            "expected_status": 200,
            "expected_budget": 19,
            "expected_enters_serving": True,
        },
        {
            "name": "completion_with_tools",
            "payload": {
                **base_payload,
                "max_completion_tokens": 23,
                "tools": [tool],
                "tool_choice": "auto",
            },
            "expected_status": 200,
            "expected_budget": 23,
            "expected_enters_serving": True,
        },
        {
            "name": "completion_with_multimodal",
            "payload": {
                **base_payload,
                "max_completion_tokens": 29,
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "image_url",
                        "image_url": {"url": PNG_DATA_URL},
                    }, {
                        "type": "text",
                        "text": "synthetic",
                    }],
                }],
            },
            "expected_status": 200,
            "expected_budget": 29,
            "expected_enters_serving": True,
        },
        {
            "name": "completion_with_reasoning_switch",
            "payload": {
                **base_payload,
                "max_completion_tokens": 31,
                "thinking": {"type": "disabled"},
            },
            "expected_status": 200,
            "expected_budget": 31,
            "expected_enters_serving": True,
        },
        {
            "name": "legacy_only",
            "payload": {**base_payload, "max_tokens": 37},
            "expected_status": 200,
            "expected_budget": 37,
            "expected_enters_serving": True,
        },
        {
            "name": "both_new_field_precedes",
            "payload": {
                **base_payload,
                "max_tokens": 41,
                "max_completion_tokens": 7,
            },
            "expected_status": 200,
            "expected_budget": 7,
            "expected_enters_serving": True,
        },
        {
            "name": "invalid_completion_type",
            "payload": {
                **base_payload,
                "max_completion_tokens": "not-an-int",
            },
            "expected_status": 400,
            "expected_budget": None,
            "expected_enters_serving": False,
        },
        {
            "name": "invalid_completion_boundary",
            "payload": {**base_payload, "max_completion_tokens": 0},
            "expected_status": 400,
            "expected_budget": None,
            "expected_enters_serving": True,
        },
        {
            "name": "unrelated_unknown_field",
            "payload": {**base_payload, "unknown_completion_budget": 8},
            "expected_status": 400,
            "expected_budget": None,
            "expected_enters_serving": False,
        },
    ]

    observations = []
    for case in cases:
        before = len(records)
        status_code, _ = await _asgi_post_json(
            app, "/v1/chat/completions", case["payload"])
        entered = len(records) == before + 1
        record = records[-1] if entered else None
        actual_budget = (
            record.get("sampling_max_tokens")
            if record is not None else None
        )
        matched = (
            status_code == case["expected_status"]
            and entered == case["expected_enters_serving"]
            and actual_budget == case["expected_budget"]
            and status_code != 500
        )
        observations.append({
            "name": case["name"],
            "status_code": status_code,
            "entered_serving": entered,
            "sampling_max_tokens": actual_budget,
            "stream": None if record is None else record["stream"],
            "has_tools": None if record is None else record["has_tools"],
            "has_multimodal": (
                None if record is None else record["has_multimodal"]),
            "thinking_present": (
                None if record is None else record["thinking_present"]),
            "output_kind": (
                None if record is None else record.get("output_kind")),
            "sampling_error_type": (
                None if record is None
                else record.get("sampling_error_type")),
            "matched": matched,
        })

    protocol_path = Path(
        __import__(
            "vllm.entrypoints.openai.protocol",
            fromlist=["__file__"],
        ).__file__).resolve()
    serving_path = Path(
        __import__(
            "vllm.entrypoints.openai.serving_chat",
            fromlist=["__file__"],
        ).__file__).resolve()
    api_server_path = Path(api_server.__file__).resolve()
    model_fields = sorted(ChatCompletionRequest.model_fields)
    reasons = [
        f"case failed: {row['name']}"
        for row in observations
        if not row["matched"]
    ]
    if "max_completion_tokens" not in model_fields:
        reasons.append("runtime model lacks max_completion_tokens")

    return {
        "schema": SCHEMA,
        "synthetic_only": True,
        "qualified": not reasons,
        "reasons": reasons,
        "case_count": len(observations),
        "http_500_count": sum(
            row["status_code"] == 500 for row in observations),
        "model_has_max_completion_tokens": (
            "max_completion_tokens" in model_fields),
        "runtime_files": {
            "protocol": {
                "path": str(protocol_path),
                "sha256": _sha256(protocol_path),
            },
            "serving_chat": {
                "path": str(serving_path),
                "sha256": _sha256(serving_path),
            },
            "api_server": {
                "path": str(api_server_path),
                "sha256": _sha256(api_server_path),
            },
        },
        "cases": observations,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = asyncio.run(run_probe())
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
