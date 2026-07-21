#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any

def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _post_stream(base: str, payload: dict[str, Any], timeout_s: float) -> dict:
    started = time.perf_counter()
    request = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    ttft_s = None
    last_output_s = None
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
                or delta.get("tool_calls"))
            if not has_output:
                continue
            elapsed = time.perf_counter() - started
            if ttft_s is None:
                ttft_s = elapsed
            last_output_s = elapsed
            if delta.get("content"):
                content_parts.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning_parts.append(delta["reasoning_content"])
            if delta.get("tool_calls"):
                tool_deltas.extend(delta["tool_calls"])
    elapsed_s = time.perf_counter() - started
    if ttft_s is None or last_output_s is None:
        raise RuntimeError("stream completed without an output delta")
    details = usage.get("prompt_tokens_details") or {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    decode_window_s = max(last_output_s - ttft_s, 0.0)
    normalized_output = {
        "content": "".join(content_parts),
        "reasoning_content": "".join(reasoning_parts),
        "tool_deltas": tool_deltas,
        "finish_reason": finish_reason,
    }
    output_sha256 = hashlib.sha256(json.dumps(
        normalized_output, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "ok": True,
        "elapsed_s": elapsed_s,
        "ttft_s": ttft_s,
        "last_output_s": last_output_s,
        "decode_window_s": decode_window_s,
        "output_tps": (
            completion_tokens / decode_window_s
            if decode_window_s > 0 else 0.0),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "cached_tokens": int(details.get("cached_tokens") or 0),
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "output_sha256": output_sha256,
    }


def main() -> int:
    from transformers import AutoTokenizer
    from long_context_api import build_exact_prompt

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--model-path",
        default="/root/public-storage/models/Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--targets", default="65536,235000")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--run-id", default="m1-47-service-ab-v1")
    parser.add_argument("--mode", choices=("control", "candidate"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    targets = [int(value) for value in args.targets.split(",")]
    if (not targets or any(value <= 32 for value in targets)
            or any(value + args.max_tokens > 262144 for value in targets)):
        parser.error("invalid --targets for the 262144-token service contract")
    if args.max_tokens < 2:
        parser.error("--max-tokens must be at least 2")
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("--timeout-s must be finite and positive")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True)
    cases = []
    reasons = []
    all_output_tps = []
    for target in targets:
        content = build_exact_prompt(
            tokenizer, target, f"{args.run_id}-{target}")
        payload = {
            "model": "llm",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": args.max_tokens,
            "min_tokens": args.max_tokens,
            "temperature": 0,
            "seed": 20260721,
            "thinking": False,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        requests = [
            _post_stream(args.base, payload, args.timeout_s)
            for _ in range(3)
        ]
        for index, request in enumerate(requests):
            if request["prompt_tokens"] != target:
                reasons.append(
                    f"target {target} request {index} prompt_tokens mismatch")
            if request["completion_tokens"] < args.max_tokens:
                reasons.append(
                    f"target {target} request {index} completion truncated")
            for field in ("elapsed_s", "ttft_s", "output_tps"):
                if not math.isfinite(request[field]) or request[field] <= 0:
                    reasons.append(
                        f"target {target} request {index} invalid {field}")
            all_output_tps.append(request["output_tps"])
        if requests[0]["cached_tokens"] != 0:
            reasons.append(f"target {target} cold request was not cold")
        for index in (1, 2):
            if requests[index]["cached_tokens"] < target - 32:
                reasons.append(
                    f"target {target} warm request {index} cache miss")
        if requests[1]["output_sha256"] != requests[2]["output_sha256"]:
            reasons.append(f"target {target} warm outputs differ")
        if (target <= 131000
                and requests[0]["output_sha256"] != requests[1]["output_sha256"]):
            reasons.append(f"target {target} cold/warm outputs differ")
        cases.append({
            "target_prompt_tokens": target,
            "cold": requests[0],
            "warm_1": requests[1],
            "warm_2": requests[2],
            "warm_ttft_median_s": statistics.median(
                [requests[1]["ttft_s"], requests[2]["ttft_s"]]),
        })

    report = {
        "schema": "bi100-m1-47-service-measurement-v1",
        "mode": args.mode,
        "run_id": args.run_id,
        "max_tokens": args.max_tokens,
        "cases": cases,
        "output_tps_p10": _percentile(all_output_tps, 10),
        "qualified_measurement": not reasons,
        "reasons": reasons,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not reasons else 1


if __name__ == "__main__":
    raise SystemExit(main())
