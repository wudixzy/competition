#!/usr/bin/env python3
"""Qualify request compatibility fixes at the tokenizer boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

Json = dict[str, Any]


def _digest_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _token_digest(tokens: list[int]) -> str:
    payload = json.dumps(
        tokens, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _tool(*, strict: bool | None = None) -> Json:
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


def _tool_history(arguments: Any) -> Json:
    return {
        "model": "llm",
        "messages": [
            {"role": "user", "content": "lookup synthetic key"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_synthetic",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": arguments,
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_synthetic",
                "content": "synthetic result",
            },
            {"role": "user", "content": "continue"},
        ],
        "tools": [_tool()],
        "tool_choice": "none",
        "max_tokens": 8,
    }


def _strict_payload(strict: bool | None) -> Json:
    return {
        "model": "llm",
        "messages": [{"role": "user", "content": "lookup synthetic key"}],
        "tools": [_tool(strict=strict)],
        "tool_choice": "auto",
        "max_tokens": 8,
    }


def _system_parts_payload(*, normalized: bool) -> Json:
    if normalized:
        messages = [
            {
                "role": "system",
                "content": "synthetic rule A1\nsynthetic rule A2\n\n"
                           "synthetic rule B",
            },
            {"role": "user", "content": "respond"},
        ]
    else:
        messages = [
            {"role": "user", "content": "respond"},
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "synthetic rule A1"},
                    {"type": "text", "text": "synthetic rule A2"},
                ],
            },
            {"role": "system", "content": "synthetic rule B"},
        ]
    return {
        "model": "llm",
        "messages": messages,
        "max_tokens": 8,
    }


def _render(tokenizer, payload: Json) -> tuple[list[int], Json]:
    from vllm.entrypoints.chat_utils import _postprocess_messages
    from vllm.entrypoints.openai.protocol import ChatCompletionRequest

    request = ChatCompletionRequest.model_validate(payload)
    _postprocess_messages(request.messages)
    tools = (
        None if request.tools is None
        else [tool.model_dump() for tool in request.tools]
    )
    tokens = tokenizer.apply_chat_template(
        request.messages,
        tokenize=True,
        add_generation_prompt=True,
        tools=tools,
        **(request.chat_template_kwargs or {}),
    )
    return list(tokens), {"tools": tools}


def _is_rejected(payload: Json) -> bool:
    from vllm.entrypoints.openai.protocol import ChatCompletionRequest

    try:
        ChatCompletionRequest.model_validate(payload)
    except Exception:
        return True
    return False


def qualify(model_path: Path) -> Json:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )

    string_tokens, _ = _render(
        tokenizer, _tool_history('{"key":"synthetic"}'))
    object_tokens, _ = _render(
        tokenizer, _tool_history({"key": "synthetic"}))
    default_tokens, default_meta = _render(
        tokenizer, _strict_payload(None))
    false_tokens, false_meta = _render(
        tokenizer, _strict_payload(False))
    system_parts_tokens, _ = _render(
        tokenizer, _system_parts_payload(normalized=False))
    system_normalized_tokens, _ = _render(
        tokenizer, _system_parts_payload(normalized=True))

    history_exact = string_tokens == object_tokens
    strict_exact = default_tokens == false_tokens
    system_parts_exact = system_parts_tokens == system_normalized_tokens
    strict_not_forwarded = (
        default_meta["tools"] == false_meta["tools"]
        and all(
            "strict" not in tool.get("function", {})
            for tool in false_meta["tools"] or []
        )
    )
    strict_true_rejected = _is_rejected(_strict_payload(True))
    required_payload = _strict_payload(None)
    required_payload["tool_choice"] = "required"
    required_rejected = _is_rejected(required_payload)

    checks = {
        "object_history_token_exact": history_exact,
        "strict_false_token_exact": strict_exact,
        "strict_not_forwarded_to_template": strict_not_forwarded,
        "strict_true_rejected": strict_true_rejected,
        "tool_choice_required_rejected": required_rejected,
        "system_text_parts_token_exact": system_parts_exact,
    }
    qualified = all(checks.values())
    return {
        "schema": "bi100-chat-request-template-compat-v1",
        "synthetic_only": True,
        "qualified": qualified,
        "checks": checks,
        "pairs": {
            "object_history": {
                "string_token_count": len(string_tokens),
                "object_token_count": len(object_tokens),
                "string_sha256": _token_digest(string_tokens),
                "object_sha256": _token_digest(object_tokens),
            },
            "strict_false": {
                "omitted_token_count": len(default_tokens),
                "false_token_count": len(false_tokens),
                "omitted_sha256": _token_digest(default_tokens),
                "false_sha256": _token_digest(false_tokens),
            },
            "system_text_parts": {
                "parts_token_count": len(system_parts_tokens),
                "normalized_token_count": len(system_normalized_tokens),
                "parts_sha256": _token_digest(system_parts_tokens),
                "normalized_sha256": _token_digest(
                    system_normalized_tokens),
            },
        },
        "model_identity": {
            "config_sha256": _digest_file(model_path / "config.json"),
            "tokenizer_config_sha256": _digest_file(
                model_path / "tokenizer_config.json"),
        },
        "privacy": {
            "contains_prompt_or_response_text": False,
            "contains_tool_schema": False,
            "contains_model_weights": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()

    report = qualify(args.model_path)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
