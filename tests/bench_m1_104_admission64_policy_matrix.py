#!/usr/bin/env python3
"""Run the fixed 18-request admission64 dataset-shaped measurement."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SHAPES = (4096, 7800, 16000)
PAIR_COUNT = 3
REQUEST_COUNT = len(SHAPES) * PAIR_COUNT * 2
SEED = 20260728
SALT_NAMESPACE = "m1-104-admission64-policy-matrix-v1"
TOOL_COUNT = 29
MAX_TOKENS = 8
TOKEN_ERROR_LIMIT = 16


def make_tools(count: int = TOOL_COUNT) -> list[dict[str, Any]]:
    names = ("read_file", "search_code", "run_command", "edit_file",
             "web_search", "list_directory", "inspect_process")
    return [{
        "type": "function",
        "function": {
            "name": f"{names[i % len(names)]}_{i}",
            "description": ("Agent engineering tool for repository inspection, "
                            "exact edits, command execution, and structured result capture."),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "query": {"type": "string"},
                    "timeout_seconds": {"type": "integer", "minimum": 1},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    } for i in range(count)]


def normalized_output(body: dict[str, Any]) -> dict[str, Any]:
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    tool_calls = message.get("tool_calls") or []
    normalized = {
        "content": message.get("content") or "",
        "reasoning_content": message.get("reasoning_content") or "",
        "tool_calls": tool_calls,
    }
    encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(),
            "content_sha256": hashlib.sha256(
                normalized["content"].encode("utf-8")).hexdigest(),
            "reasoning_sha256": hashlib.sha256(
                normalized["reasoning_content"].encode("utf-8")).hexdigest(),
            "tool_calls_sha256": hashlib.sha256(
                json.dumps(tool_calls, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")).encode("utf-8")).hexdigest()}


def build_prompt(tokenizer: Any, corpus: str, target: int, salt: str,
                 tools: list[dict[str, Any]]) -> tuple[list[dict[str, str]], int]:
    ids = tokenizer.encode(corpus, add_special_tokens=False)
    system = (f"RUN_ID={salt}. You are a coding agent. Inspect the supplied "
              "repository material, preserve exact identifiers, and produce a concise "
              "implementation plan.")

    def messages(n: int) -> list[dict[str, str]]:
        return [{"role": "system", "content": system},
                {"role": "user", "content": tokenizer.decode(ids[:n])}]

    low, high = 0, len(ids)
    while low < high:
        mid = (low + high + 1) // 2
        rendered = tokenizer.apply_chat_template(
            messages(mid), tools=tools, tokenize=True,
            add_generation_prompt=True, enable_thinking=False)
        if len(rendered) <= target:
            low = mid
        else:
            high = mid - 1
    final = messages(low)
    return final, len(tokenizer.apply_chat_template(
        final, tools=tools, tokenize=True, add_generation_prompt=True,
        enable_thinking=False))


def stream_request(base: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    req = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    status = 0
    usage: dict[str, Any] = {}
    content, reasoning, tool_calls = [], [], []
    finish_reason = None
    ttft = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = response.status
            if status != 200:
                return {"ok": False, "http_status": status,
                        "error": f"http {status}"}
            for raw in response:
                if not raw.decode("utf-8", "replace").strip().startswith("data:"):
                    continue
                value = raw.decode("utf-8", "replace").strip()[5:].strip()
                if value == "[DONE]":
                    break
                event = json.loads(value)
                if event.get("usage"):
                    usage = event["usage"]
                choice = (event.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                if delta.get("tool_calls"):
                    tool_calls.extend(delta["tool_calls"])
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
                if ttft is None and (delta.get("content") or delta.get("reasoning_content")
                                     or delta.get("tool_calls")):
                    ttft = time.perf_counter() - started
        body = {"choices": [{"message": {
            "content": "".join(content),
            "reasoning_content": "".join(reasoning),
            "tool_calls": tool_calls}, "finish_reason": finish_reason}]}
        elapsed = time.perf_counter() - started
        completion = int(usage.get("completion_tokens") or 0)
        return {"ok": True, "http_status": status, "usage": usage,
                "ttft_s": ttft, "latency_s": elapsed,
                "output_tps": completion / elapsed if elapsed else 0.0,
                "finish_reason": finish_reason, "completion_tokens": completion,
                "cached_tokens": int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
                "output_sha256": normalized_output(body)}
    except (OSError, urllib.error.URLError, ValueError) as exc:
        return {"ok": False, "http_status": status, "error": repr(exc)}


def service_health(base: str, timeout: float) -> bool:
    request = urllib.request.Request(f"{base.rstrip('/')}/health", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def validate_report(report: dict[str, Any]) -> None:
    records = report.get("requests") or []
    if len(records) != REQUEST_COUNT:
        raise ValueError(f"expected {REQUEST_COUNT} requests, got {len(records)}")
    if not report.get("service_healthy"):
        raise ValueError("service health check failed")
    for shape in SHAPES:
        for pair in range(PAIR_COUNT):
            rows = [r for r in records if r["target_tokens"] == shape and r["pair"] == pair]
            if len(rows) != 2 or {r["phase"] for r in rows} != {"cold", "warm"}:
                raise ValueError(f"incomplete shape={shape} pair={pair}")
            cold, warm = sorted(rows, key=lambda r: r["phase"])
            if cold["cached_tokens"] != 0:
                raise ValueError(f"cold cache is nonzero shape={shape} pair={pair}")
            if cold["output_sha256"] != warm["output_sha256"]:
                raise ValueError(f"cold/warm output mismatch shape={shape} pair={pair}")
            if abs(cold["prompt_tokens"] - shape) >= TOKEN_ERROR_LIMIT:
                raise ValueError(f"prompt token target mismatch shape={shape}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=600)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tools = make_tools()
    corpus = args.corpus.read_text(encoding="utf-8", errors="replace")
    records = []
    for target in SHAPES:
        for pair in range(PAIR_COUNT):
            salt = f"{SALT_NAMESPACE}:shape-{target}:pair-{pair}"
            messages, rendered = build_prompt(tokenizer, corpus, target, salt, tools)
            for phase in ("cold", "warm"):
                payload = {"model": "llm", "messages": messages, "tools": tools,
                           "tool_choice": "none", "thinking": False, "temperature": 0,
                           "seed": SEED, "max_tokens": MAX_TOKENS, "stream": True,
                           "stream_options": {"include_usage": True}}
                result = stream_request(args.base, payload, args.timeout_s)
                usage = result.get("usage") or {}
                result.update({"target_tokens": target, "pair": pair, "phase": phase,
                               "rendered_tokens_local": rendered,
                               "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                               "seed": SEED, "salt_namespace": SALT_NAMESPACE,
                               "tool_count": TOOL_COUNT})
                records.append(result)
    report = {"schema": "m1-104.v1", "request_count": REQUEST_COUNT,
              "service_healthy": service_health(args.base, args.timeout_s)
              and all(r.get("ok") for r in records),
              "fixed": {"shapes": SHAPES, "pairs": PAIR_COUNT, "seed": SEED,
                        "tools": TOOL_COUNT, "max_tokens": MAX_TOKENS,
                        "temperature": 0, "thinking": False, "tool_choice": "none",
                        "stream_usage": True, "salt_namespace": SALT_NAMESPACE},
              "requests": records}
    validate_report(report)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
