#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class RequestResult:
    ok: bool
    latency_s: float
    ttft_s: float | None
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    error: str = ""


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def post_stream(base: str, payload: dict[str, Any], timeout: float) -> RequestResult:
    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    ttft: float | None = None
    usage: dict[str, Any] = {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return RequestResult(False, time.perf_counter() - t0, None, 0, 0, 0,
                                     f"http {resp.status}")
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                value = line[5:].strip()
                if value == "[DONE]":
                    break
                chunk = json.loads(value)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if ttft is None and (delta.get("content") or delta.get("reasoning_content")):
                    ttft = time.perf_counter() - t0
        latency = time.perf_counter() - t0
        details = usage.get("prompt_tokens_details") or {}
        return RequestResult(
            True,
            latency,
            ttft,
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
            int(details.get("cached_tokens") or 0),
        )
    except Exception as exc:  # noqa: BLE001 - benchmark result must retain failure text.
        return RequestResult(False, time.perf_counter() - t0, ttft, 0, 0, 0, repr(exc))


def make_payload(prompt: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": "llm",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "thinking": False,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def score(output_tps_p10: float, input_tps: float, cache_tps: float) -> float:
    return output_tps_p10 * 16.796 + input_tps * 2.799 + cache_tps * 0.56


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stagger-s", type=float, default=0.25)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt-repeat", type=int, default=126)
    parser.add_argument("--prompt-salt", default="")
    parser.add_argument("--timeout-s", type=float, default=360)
    parser.add_argument("--label", default="custom")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    prompt_salt = args.prompt_salt or f"{args.label}-{time.time_ns()}"
    prefix = (
        f"请基于以下材料回答问题。RUN_ID={prompt_salt}\n" +
        ("BI100 Qwen3.6 prefix cache benchmark material. " * args.prompt_repeat)
    )
    prompt = prefix + "\n问题：请用一小段话概括材料主题。"
    payloads = [make_payload(prompt, args.max_tokens) for _ in range(args.requests)]

    wall_t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for payload in payloads:
            futures.append(pool.submit(post_stream, args.base, payload, args.timeout_s))
            time.sleep(args.stagger_s)
        results = [future.result() for future in futures]
    wall_s = time.perf_counter() - wall_t0

    ok = [result for result in results if result.ok]
    ttfts = [result.ttft_s for result in ok if result.ttft_s is not None]
    output_rates = [
        result.completion_tokens / result.latency_s
        for result in ok
        if result.latency_s > 0 and result.completion_tokens > 0
    ]
    prompt_tokens = sum(result.prompt_tokens for result in ok)
    completion_tokens = sum(result.completion_tokens for result in ok)
    cached_tokens = sum(result.cached_tokens for result in ok)
    success_rate = len(ok) / args.requests if args.requests else 0.0
    input_tps = prompt_tokens / wall_s if wall_s > 0 else 0.0
    output_tps_total = completion_tokens / wall_s if wall_s > 0 else 0.0
    cache_tps = cached_tokens / wall_s if wall_s > 0 else 0.0
    cache_hit_rate = cached_tokens / prompt_tokens if prompt_tokens else 0.0
    output_tps_p10 = percentile(output_rates, 10)
    ttft_p90 = percentile(ttfts, 90)
    weighted = score(output_tps_p10, input_tps, cache_tps)

    report = {
        "label": args.label,
        "requests": args.requests,
        "workers": args.workers,
        "success_rate": success_rate,
        "wall_s": wall_s,
        "ttft_p90_s": ttft_p90,
        "output_tps_p10": output_tps_p10,
        "input_tps": input_tps,
        "output_tps_total": output_tps_total,
        "cache_tps": cache_tps,
        "cache_hit_rate": cache_hit_rate,
        "weighted_score": weighted,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "errors": [result.error for result in results if not result.ok],
        "latency_s_mean": statistics.mean([result.latency_s for result in ok]) if ok else 0.0,
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
