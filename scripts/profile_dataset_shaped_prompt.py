#!/usr/bin/env python3
"""Build and submit one deterministic dataset-shaped chat request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.request

from transformers import AutoTokenizer

from bench_perf import post_stream, request_metrics


def make_tools(count: int) -> list[dict]:
    tool_names = ("read_file", "search_code", "run_command", "edit_file",
                  "web_search", "list_directory", "inspect_process")
    tools = []
    for index in range(count):
        name = f"{tool_names[index % len(tool_names)]}_{index}"
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": (
                    "Agent engineering tool for repository inspection, exact "
                    "edits, command execution, and structured result capture."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "query": {"type": "string"},
                        "timeout_seconds": {
                            "type": "integer",
                            "minimum": 1,
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        })
    return tools


def load_corpus(paths: list[Path]) -> str:
    parts = []
    for path in paths:
        parts.append(f"\n===== {path.name} =====\n")
        parts.append(path.read_text(errors="replace"))
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--target-tokens", type=int, default=7800)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--tools", type=int, default=29)
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--prompt-salt", default="")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("corpus", type=Path, nargs="+")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True)
    tools = make_tools(args.tools)
    corpus_ids = tokenizer.encode(
        load_corpus(args.corpus), add_special_tokens=False)
    system = (
        (f"RUN_ID={args.prompt_salt}. " if args.prompt_salt else "")
        + "You are a coding agent. Inspect the supplied repository material, "
        "preserve exact identifiers, and produce a concise implementation plan."
    )

    def build_messages(content_token_count: int) -> list[dict]:
        content = tokenizer.decode(corpus_ids[:content_token_count])
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]

    low, high = 0, len(corpus_ids)
    while low < high:
        mid = (low + high + 1) // 2
        rendered = tokenizer.apply_chat_template(
            build_messages(mid),
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if len(rendered) <= args.target_tokens:
            low = mid
        else:
            high = mid - 1

    messages = build_messages(low)
    rendered_tokens = len(tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    ))
    payload = {
        "model": "llm",
        "messages": messages,
        "tools": tools,
        "tool_choice": "none",
        "thinking": False,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": args.stream,
    }
    if args.stream:
        payload["stream_options"] = {"include_usage": True}
        response = post_stream(args.base, payload, args.timeout_s)
        if not response.ok:
            raise RuntimeError(response.error)
        body = {}
        timing = request_metrics(response)
    else:
        request = urllib.request.Request(
            f"{args.base.rstrip('/')}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(
                request, timeout=args.timeout_s) as response:
            body = json.load(response)
        timing = None

    result = {
        "target_tokens": args.target_tokens,
        "rendered_tokens_local": rendered_tokens,
        "content_tokens": low,
        "tool_count": len(tools),
        "prompt_salt": args.prompt_salt,
        "usage": body.get("usage"),
        "finish_reason": (body.get("choices")
                           or [{}])[0].get("finish_reason"),
        "timing": timing,
    }
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
