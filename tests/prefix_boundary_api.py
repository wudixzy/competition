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


def encode_chat(tokenizer: Any, content: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def common_prefix_len(first: list[int], second: list[int]) -> int:
    for index, (left, right) in enumerate(zip(first, second)):
        if left != right:
            return index
    return min(len(first), len(second))


def build_boundary_prompts(
    tokenizer: Any,
    block_context_len: int,
    prefix_query_len: int,
    block_size: int,
    run_id: str,
) -> tuple[str, str, int, int]:
    target_total = block_context_len + prefix_query_len + 1
    header = f"BI100 prefix boundary regression {run_id}.\n"

    selected: tuple[str, str, int] | None = None
    low = max(0, block_context_len - 256)
    high = block_context_len + 256
    while low <= high:
        filler_count = (low + high) // 2
        common = header + (" x" * filler_count)
        first = common + "\nBranch A." + (" a" * 384)
        second = common + "\nBranch B." + (" b" * 384)
        shared = common_prefix_len(
            encode_chat(tokenizer, first), encode_chat(tokenizer, second))
        cached = shared // block_size * block_size
        if cached == block_context_len:
            selected = (first, common, shared)
            break
        if cached < block_context_len:
            low = filler_count + 1
        else:
            high = filler_count - 1
    if selected is None:
        raise RuntimeError(
            f"could not construct cached prefix of {block_context_len} tokens")

    first, common, _ = selected
    low = 0
    high = prefix_query_len * 3 + 256
    second = ""
    while low <= high:
        suffix_count = (low + high) // 2
        candidate = common + "\nBranch B." + (" b" * suffix_count)
        count = len(encode_chat(tokenizer, candidate))
        if count == target_total:
            second = candidate
            break
        if count < target_total:
            low = suffix_count + 1
        else:
            high = suffix_count - 1
    if not second:
        raise RuntimeError(
            f"could not construct exactly {target_total} prompt tokens")

    first_ids = encode_chat(tokenizer, first)
    second_ids = encode_chat(tokenizer, second)
    shared = common_prefix_len(first_ids, second_ids)
    cached = shared // block_size * block_size
    if cached != block_context_len:
        raise RuntimeError(
            f"constructed cached prefix {cached}, expected {block_context_len}")
    return first, second, shared, len(second_ids)


def post_chat(base: str, content: str, max_tokens: int, timeout_s: float) -> Json:
    payload = {
        "model": "llm",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "thinking": False,
        "temperature": 0,
        "seed": 20260713,
    }
    request = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def summarize(response: Json, elapsed_s: float) -> Json:
    usage = response.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    choice = response["choices"][0]
    encoded = json.dumps(
        choice["message"], ensure_ascii=False, sort_keys=True).encode()
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens", 0),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "message_sha256": hashlib.sha256(encoded).hexdigest(),
        "elapsed_s": round(elapsed_s, 3),
    }


def request_once(
    base: str,
    content: str,
    max_tokens: int,
    timeout_s: float,
) -> tuple[Json, Json]:
    started = time.monotonic()
    response = post_chat(base, content, max_tokens, timeout_s)
    return response, summarize(response, time.monotonic() - started)


def assert_equivalent(first: Json, second: Json) -> None:
    assert second["choices"][0]["message"] == first["choices"][0]["message"]
    assert second["choices"][0].get("finish_reason") == (
        first["choices"][0].get("finish_reason"))
    assert (second.get("usage") or {}).get("completion_tokens") == (
        (first.get("usage") or {}).get("completion_tokens"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--model-path",
        default="/root/public-storage/models/Qwen/Qwen3.6-35B-A3B",
    )
    parser.add_argument("--block-context-len", type=int, default=11296)
    parser.add_argument("--prefix-query-len", type=int, default=320)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--min-partial-cached-tokens", type=int, default=8176)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=900)
    parser.add_argument("--run-id", default=str(time.time_ns()))
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True)
    first_content, boundary_content, shared_tokens, total_tokens = (
        build_boundary_prompts(
            tokenizer,
            args.block_context_len,
            args.prefix_query_len,
            args.block_size,
            args.run_id,
        ))

    primer, primer_summary = request_once(
        args.base, first_content, 1, args.timeout_s)
    partial, partial_summary = request_once(
        args.base, boundary_content, args.max_tokens, args.timeout_s)
    warm, warm_summary = request_once(
        args.base, boundary_content, args.max_tokens, args.timeout_s)

    report = {
        "block_context_len": args.block_context_len,
        "prefix_query_len": args.prefix_query_len,
        "min_partial_cached_tokens": args.min_partial_cached_tokens,
        "shared_tokens_before_block_rounding": shared_tokens,
        "target_prompt_tokens": total_tokens,
        "primer": primer_summary,
        "partial_cache": partial_summary,
        "warm_cache": warm_summary,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    for name, response in (
        ("primer", primer),
        ("partial_cache", partial),
        ("warm_cache", warm),
    ):
        (args.json_out.parent / f"{name}_response.json").write_text(
            json.dumps(response, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    args.json_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    assert total_tokens == (
        args.block_context_len + args.prefix_query_len + 1), total_tokens
    assert partial_summary["prompt_tokens"] == total_tokens, partial_summary
    assert (args.min_partial_cached_tokens
            <= partial_summary["cached_tokens"]
            < total_tokens - args.block_size), partial_summary
    assert warm_summary["cached_tokens"] >= total_tokens - 2 * args.block_size, (
        warm_summary)
    assert_equivalent(partial, warm)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
