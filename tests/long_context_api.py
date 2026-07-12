#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


Json = dict[str, Any]


def prompt_token_count(tokenizer: Any, content: str) -> int:
    return len(tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    ))


def build_exact_prompt(tokenizer: Any, target_tokens: int, run_id: str) -> str:
    prefix = f"Long-context contract test {run_id}. Remember marker FINAL-99500.\n"
    suffix = "\nReply with the marker only."
    low = 0
    high = target_tokens * 2
    while low <= high:
        filler_count = (low + high) // 2
        content = prefix + (" x" * filler_count) + suffix
        count = prompt_token_count(tokenizer, content)
        if count == target_tokens:
            return content
        if count < target_tokens:
            low = filler_count + 1
        else:
            high = filler_count - 1
    raise RuntimeError(f"could not construct exactly {target_tokens} prompt tokens")


def post_chat(base: str, payload: Json, timeout_s: float) -> tuple[Json, float]:
    request = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result, time.monotonic() - started


def summarize(response: Json, elapsed_s: float) -> Json:
    usage = response.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    choice = response["choices"][0]
    message = choice["message"]
    encoded = json.dumps(message, ensure_ascii=False, sort_keys=True).encode()
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens", 0),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "message_sha256": hashlib.sha256(encoded).hexdigest(),
        "elapsed_s": round(elapsed_s, 3),
    }


def assert_equivalent(first: Json, second: Json) -> None:
    first_choice = first["choices"][0]
    second_choice = second["choices"][0]
    assert second_choice["message"] == first_choice["message"]
    assert second_choice.get("finish_reason") == first_choice.get("finish_reason")
    assert (second.get("usage") or {}).get("completion_tokens") == (
        first.get("usage") or {}).get("completion_tokens")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--model-path",
        default="/root/public-storage/models/Qwen/Qwen3.6-35B-A3B",
    )
    parser.add_argument("--target-prompt-tokens", type=int, default=99500)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=100000)
    parser.add_argument("--timeout-s", type=float, default=1800)
    parser.add_argument("--run-id", default=str(time.time_ns()))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.target_prompt_tokens + args.max_tokens > args.max_model_len:
        parser.error("prompt plus max tokens exceeds --max-model-len")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True)
    content = build_exact_prompt(
        tokenizer, args.target_prompt_tokens, args.run_id)
    payload = {
        "model": "llm",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": args.max_tokens,
        "thinking": False,
        "temperature": 0,
        "seed": 20260712,
    }

    first, first_elapsed = post_chat(args.base, payload, args.timeout_s)
    second, second_elapsed = post_chat(args.base, payload, args.timeout_s)
    first_summary = summarize(first, first_elapsed)
    second_summary = summarize(second, second_elapsed)
    assert first_summary["prompt_tokens"] == args.target_prompt_tokens, first_summary
    assert second_summary["prompt_tokens"] == args.target_prompt_tokens, second_summary
    assert first_summary["cached_tokens"] == 0, first_summary
    assert second_summary["cached_tokens"] >= 98304, second_summary
    assert_equivalent(first, second)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "long_context_response1.json").write_text(
        json.dumps(first, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "long_context_response2.json").write_text(
        json.dumps(second, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "target_prompt_tokens": args.target_prompt_tokens,
        "max_tokens": args.max_tokens,
        "first": first_summary,
        "second": second_summary,
    }
    (args.output_dir / "long_context_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
