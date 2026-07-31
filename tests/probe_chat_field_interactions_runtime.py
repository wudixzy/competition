#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SCHEMA = "bi100-chat-field-interactions-runtime-probe-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def _asgi_post_json(
        app: Any, path: str, payload: Any) -> tuple[int, bytes]:
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


def _record_request(request: Any, sampling: Any) -> dict[str, Any]:
    template_kwargs = request.chat_template_kwargs or {}
    guided = getattr(sampling, "guided_decoding", None)
    tool_choice = request.tool_choice
    if tool_choice is None:
        tool_choice_mode = "null"
    elif isinstance(tool_choice, str):
        tool_choice_mode = tool_choice
    else:
        tool_choice_mode = "named"
    return {
        "sampling_max_tokens": sampling.max_tokens,
        "sampling_logprobs": sampling.logprobs,
        "stream": bool(request.stream),
        "has_stream_options": request.stream_options is not None,
        "enable_thinking": template_kwargs.get("enable_thinking"),
        "tool_choice_mode": tool_choice_mode,
        "add_generation_prompt": request.add_generation_prompt,
        "continue_final_message": request.continue_final_message,
        "has_guided_json": (
            guided is not None
            and getattr(guided, "json", None) is not None
        ),
    }


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
            try:
                sampling = request.to_sampling_params(4096)
            except (TypeError, ValueError) as exc:
                records.append({
                    "sampling_error_type": type(exc).__name__,
                })
                return self.create_error_response(str(exc))
            records.append(_record_request(request, sampling))
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
    app.state.openai_serving_chat = FakeServingChat()

    base = {
        "model": "llm",
        "messages": [{"role": "user", "content": "synthetic"}],
        "temperature": 0,
        "max_tokens": 8,
    }
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "synthetic",
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
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
            "name": "completion_budget_precedence",
            "payload": {
                **base,
                "max_tokens": 41,
                "max_completion_tokens": 7,
            },
            "status": 200,
            "enters": True,
            "record": {"sampling_max_tokens": 7},
        },
        {
            "name": "thinking_precedence",
            "payload": {
                **base,
                "thinking": False,
                "chat_template_kwargs": {"enable_thinking": True},
            },
            "status": 200,
            "enters": True,
            "record": {"enable_thinking": False},
        },
        {
            "name": "empty_stream_options_without_stream",
            "payload": {**base, "stream_options": {}},
            "status": 400,
            "enters": False,
        },
        {
            "name": "empty_stream_options_with_stream",
            "payload": {
                **base,
                "stream": True,
                "stream_options": {},
            },
            "status": 200,
            "enters": True,
            "record": {
                "stream": True,
                "has_stream_options": True,
            },
        },
        {
            "name": "top_logprobs_zero_is_noop",
            "payload": {
                **base,
                "logprobs": False,
                "top_logprobs": 0,
            },
            "status": 200,
            "enters": True,
            "record": {"sampling_logprobs": None},
        },
        {
            "name": "positive_top_logprobs_requires_logprobs",
            "payload": {
                **base,
                "logprobs": False,
                "top_logprobs": 1,
            },
            "status": 400,
            "enters": False,
        },
        {
            "name": "null_tool_choice",
            "payload": {**base, "tool_choice": None},
            "status": 200,
            "enters": True,
            "record": {"tool_choice_mode": "null"},
        },
        {
            "name": "malformed_tool_choice",
            "payload": {
                **base,
                "tools": [tool],
                "tool_choice": {},
            },
            "status": 400,
            "enters": False,
        },
        {
            "name": "missing_response_schema",
            "payload": {
                **base,
                "response_format": {"type": "json_schema"},
            },
            "status": 400,
            "enters": False,
        },
        {
            "name": "valid_response_schema",
            "payload": {**base, "response_format": schema},
            "status": 200,
            "enters": True,
            "record": {"has_guided_json": True},
        },
        {
            "name": "multiple_output_constraints",
            "payload": {
                **base,
                "response_format": {"type": "json_object"},
                "guided_grammar": "root ::= \"x\"",
            },
            "status": 400,
            "enters": False,
        },
        {
            "name": "continue_uses_default_generation_prompt",
            "payload": {**base, "continue_final_message": True},
            "status": 400,
            "enters": False,
        },
        {
            "name": "continue_with_generation_prompt_disabled",
            "payload": {
                **base,
                "continue_final_message": True,
                "add_generation_prompt": False,
            },
            "status": 200,
            "enters": True,
            "record": {
                "add_generation_prompt": False,
                "continue_final_message": True,
            },
        },
        {
            "name": "legacy_function_call_fail_closed",
            "payload": {**base, "function_call": "auto"},
            "status": 400,
            "enters": False,
        },
        {
            "name": "legacy_functions_fail_closed",
            "payload": {
                **base,
                "functions": [tool["function"]],
            },
            "status": 400,
            "enters": False,
        },
        {
            "name": "responses_max_output_tokens_fail_closed",
            "payload": {**base, "max_output_tokens": 8},
            "status": 400,
            "enters": False,
        },
        {
            "name": "reasoning_effort_fail_closed",
            "payload": {**base, "reasoning_effort": "medium"},
            "status": 400,
            "enters": False,
        },
        {
            "name": "malformed_logprob_type",
            "payload": {
                **base,
                "top_logprobs": {"synthetic": "value"},
            },
            "status": 400,
            "enters": False,
        },
    ]

    observations = []
    for case in cases:
        before = len(records)
        status_code, _ = await _asgi_post_json(
            app,
            "/v1/chat/completions",
            case["payload"],
        )
        entered = len(records) == before + 1
        record = records[-1] if entered else None
        expected_record = case.get("record", {})
        record_matches = (
            record is not None
            and all(record.get(key) == value
                    for key, value in expected_record.items())
        ) if case["enters"] else record is None
        matched = (
            status_code == case["status"]
            and entered == case["enters"]
            and record_matches
            and status_code != 500
        )
        observations.append({
            "name": case["name"],
            "status_code": status_code,
            "entered_serving": entered,
            "record": record,
            "matched": matched,
        })

    protocol_module = __import__(
        "vllm.entrypoints.openai.protocol",
        fromlist=["__file__"],
    )
    protocol_path = Path(protocol_module.__file__).resolve()
    api_server_path = Path(api_server.__file__).resolve()
    model_fields = sorted(ChatCompletionRequest.model_fields)
    reasons = [
        f"case failed: {row['name']}"
        for row in observations
        if not row["matched"]
    ]

    required_fields = {
        "max_tokens",
        "max_completion_tokens",
        "thinking",
        "stream",
        "stream_options",
        "logprobs",
        "top_logprobs",
        "tools",
        "tool_choice",
        "response_format",
        "add_generation_prompt",
        "continue_final_message",
    }
    missing_fields = sorted(required_fields - set(model_fields))
    if missing_fields:
        reasons.append(f"runtime model lacks fields: {missing_fields}")

    return {
        "schema": SCHEMA,
        "synthetic_only": True,
        "qualified": not reasons,
        "reasons": reasons,
        "case_count": len(observations),
        "http_500_count": sum(
            row["status_code"] == 500 for row in observations),
        "required_fields_present": not missing_fields,
        "runtime_files": {
            "protocol": {
                "path": str(protocol_path),
                "sha256": _sha256(protocol_path),
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
