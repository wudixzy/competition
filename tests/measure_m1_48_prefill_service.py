#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any


SCHEMA = "bi100-m1-48-prefill-service-v1"
VERSION = 1
SEED = 20260722


def _post_stream(
    base: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    ttft_s = None
    usage: dict[str, Any] = {}
    finish_reason = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_deltas: list[Any] = []
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        for raw_line in response:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            value = line[5:].strip()
            if value == "[DONE]":
                break
            event = json.loads(value)
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta") or {}
            has_output = bool(
                delta.get("content")
                or delta.get("reasoning_content")
                or delta.get("tool_calls")
            )
            if not has_output:
                continue
            if ttft_s is None:
                ttft_s = time.perf_counter() - started
            if delta.get("content"):
                content_parts.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning_parts.append(delta["reasoning_content"])
            if delta.get("tool_calls"):
                tool_deltas.extend(delta["tool_calls"])
    elapsed_s = time.perf_counter() - started
    if ttft_s is None:
        raise RuntimeError("stream completed without an output delta")
    details = usage.get("prompt_tokens_details") or {}
    normalized_output = {
        "content": "".join(content_parts),
        "reasoning_content": "".join(reasoning_parts),
        "tool_deltas": tool_deltas,
        "finish_reason": finish_reason,
    }
    output_sha256 = hashlib.sha256(json.dumps(
        normalized_output,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return {
        "elapsed_s": elapsed_s,
        "ttft_s": ttft_s,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "cached_tokens": int(details.get("cached_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "finish_reason": finish_reason,
        "output_sha256": output_sha256,
    }


def validate_measurement(
    request: dict[str, Any],
    target_prompt_tokens: int,
) -> list[str]:
    reasons = []
    if request.get("prompt_tokens") != target_prompt_tokens:
        reasons.append("prompt token count does not match the fixed target")
    if request.get("cached_tokens") != 0:
        reasons.append("profile request was not cold")
    if request.get("completion_tokens") != 1:
        reasons.append("profile request did not produce exactly one token")
    for field in ("elapsed_s", "ttft_s"):
        value = request.get(field)
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value <= 0):
            reasons.append(f"request {field} is not finite and positive")
    digest = request.get("output_sha256")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)):
        reasons.append("output digest is invalid")
    return reasons


def _write_atomic(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> int:
    from transformers import AutoTokenizer
    from long_context_api import build_exact_prompt

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--model-path",
        default="/root/public-storage/models/Qwen/Qwen3.6-35B-A3B",
    )
    parser.add_argument("--target-prompt-tokens", type=int, default=235000)
    parser.add_argument("--max-model-len", type=int, default=262144)
    parser.add_argument("--timeout-s", type=float, default=4200.0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("control", "profile"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.target_prompt_tokens + 1 > args.max_model_len:
        parser.error("prompt plus one output token exceeds --max-model-len")
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("--timeout-s must be finite and positive")
    if not args.run_id or len(args.run_id) > 64:
        parser.error("--run-id must contain 1..64 characters")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True)
    content = build_exact_prompt(
        tokenizer, args.target_prompt_tokens, args.run_id)
    payload = {
        "model": "llm",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1,
        "min_tokens": 1,
        "temperature": 0,
        "seed": SEED,
        "thinking": False,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = _post_stream(args.base, payload, args.timeout_s)
    reasons = validate_measurement(request, args.target_prompt_tokens)
    report = {
        "schema": SCHEMA,
        "version": VERSION,
        "mode": args.mode,
        "run_id": args.run_id,
        "protocol": {
            "stream": True,
            "max_tokens": 1,
            "min_tokens": 1,
            "temperature": 0,
            "seed": SEED,
            "thinking": False,
            "target_prompt_tokens": args.target_prompt_tokens,
            "max_model_len": args.max_model_len,
        },
        "request": request,
        "qualified_measurement": not reasons,
        "reasons": reasons,
    }
    _write_atomic(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not reasons else 1


if __name__ == "__main__":
    raise SystemExit(main())
