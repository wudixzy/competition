#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request
from dataclasses import asdict, dataclass, field
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
    output_event_times_s: list[float] = field(default_factory=list)
    sse_event_times_s: list[float] = field(default_factory=list)


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


def has_output_delta(delta: dict[str, Any]) -> bool:
    return bool(delta.get("content") or delta.get("reasoning_content")
                or delta.get("tool_calls"))


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
    output_event_times: list[float] = []
    sse_event_times: list[float] = []
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
                event_time = time.perf_counter() - t0
                sse_event_times.append(event_time)
                if value == "[DONE]":
                    break
                chunk = json.loads(value)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if has_output_delta(delta):
                    if ttft is None:
                        ttft = event_time
                    output_event_times.append(event_time)
        latency = time.perf_counter() - t0
        details = usage.get("prompt_tokens_details") or {}
        return RequestResult(
            True,
            latency,
            ttft,
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
            int(details.get("cached_tokens") or 0),
            output_event_times_s=output_event_times,
            sse_event_times_s=sse_event_times,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark result must retain failure text.
        return RequestResult(False, time.perf_counter() - t0, ttft, 0, 0, 0, repr(exc))


def make_payload(prompt: str, max_tokens: int,
                 seed: int | None = None) -> dict[str, Any]:
    payload = {
        "model": "llm",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "thinking": False,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if seed is not None:
        payload["seed"] = seed
    return payload


def score(output_tps_p10: float, input_tps: float, cache_tps: float) -> float:
    return output_tps_p10 * 16.796 + input_tps * 2.799 + cache_tps * 0.56


def request_metrics(result: RequestResult) -> dict[str, Any]:
    metrics = asdict(result)
    first_event = result.ttft_s
    decode_window = max(0.0, result.latency_s - first_event) \
        if first_event is not None else 0.0
    intervals = [
        later - earlier
        for earlier, later in zip(result.output_event_times_s,
                                  result.output_event_times_s[1:])
    ]
    metrics.update({
        "request_e2e_s": result.latency_s,
        "first_event_s": first_event,
        "stream_decode_window_s": decode_window,
        "output_rate_e2e": (
            result.completion_tokens / result.latency_s
            if result.latency_s > 0 else 0.0),
        "output_tps_decode": (
            result.completion_tokens / decode_window
            if decode_window > 0 else 0.0),
        "inter_token_latencies_s": intervals,
        "inter_token_latency_p50_s": percentile(intervals, 50),
        "inter_token_latency_p90_s": percentile(intervals, 90),
    })
    return metrics


def summarize_results(results: list[RequestResult], request_count: int,
                      workers: int, wall_s: float, label: str) -> dict[str, Any]:
    ok = [result for result in results if result.ok]
    per_request = [request_metrics(result) for result in results]
    ttfts = [result.ttft_s for result in ok if result.ttft_s is not None]
    e2e_rates = [
        metric["output_rate_e2e"] for result, metric in zip(results, per_request)
        if result.ok and metric["output_rate_e2e"] > 0
    ]
    decode_rates = [
        metric["output_tps_decode"]
        for result, metric in zip(results, per_request)
        if result.ok and metric["output_tps_decode"] > 0
    ]
    inter_token_latencies = [
        interval
        for result, metric in zip(results, per_request)
        if result.ok
        for interval in metric["inter_token_latencies_s"]
    ]
    prompt_tokens = sum(result.prompt_tokens for result in ok)
    completion_tokens = sum(result.completion_tokens for result in ok)
    cached_tokens = sum(result.cached_tokens for result in ok)
    uncached_tokens = max(0, prompt_tokens - cached_tokens)
    success_rate = len(ok) / request_count if request_count else 0.0
    prompt_tps_total = prompt_tokens / wall_s if wall_s > 0 else 0.0
    prompt_tps_uncached = uncached_tokens / wall_s if wall_s > 0 else 0.0
    output_tps_total = completion_tokens / wall_s if wall_s > 0 else 0.0
    cache_tps = cached_tokens / wall_s if wall_s > 0 else 0.0
    cache_hit_rate = cached_tokens / prompt_tokens if prompt_tokens else 0.0
    output_tps_decode_p10 = percentile(decode_rates, 10)
    output_rate_e2e_p10 = percentile(e2e_rates, 10)
    ttft_p90 = percentile(ttfts, 90)
    score_overlap = score(output_tps_decode_p10, prompt_tps_total, cache_tps)
    score_disjoint = score(output_tps_decode_p10, prompt_tps_uncached, cache_tps)

    return {
        "label": label,
        "requests": request_count,
        "workers": workers,
        "success_rate": success_rate,
        "wall_s": wall_s,
        "ttft_p90_s": ttft_p90,
        "first_event_p90_s": ttft_p90,
        "output_tps_p10": output_tps_decode_p10,
        "output_tps_decode_p10": output_tps_decode_p10,
        "output_rate_e2e_p10": output_rate_e2e_p10,
        "input_tps": prompt_tps_total,
        "prompt_tps_total": prompt_tps_total,
        "prompt_tps_uncached": prompt_tps_uncached,
        "output_tps_total": output_tps_total,
        "cache_tps": cache_tps,
        "cache_hit_rate": cache_hit_rate,
        "weighted_score": score_overlap,
        "score_overlap": score_overlap,
        "score_disjoint": score_disjoint,
        "inter_token_latency_p50_s": percentile(inter_token_latencies, 50),
        "inter_token_latency_p90_s": percentile(inter_token_latencies, 90),
        "prompt_tokens": prompt_tokens,
        "uncached_prompt_tokens": uncached_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "errors": [result.error for result in results if not result.ok],
        "latency_s_mean": (
            statistics.mean([result.latency_s for result in ok]) if ok else 0.0),
        "request_metrics": per_request,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--stagger-s", type=float, default=0.25)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt-repeat", type=int, default=126)
    parser.add_argument("--prompt-salt", default="")
    parser.add_argument("--seed", type=int, default=None)
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
    payloads = [
        make_payload(prompt, args.max_tokens, args.seed)
        for _ in range(args.requests)
    ]

    wall_t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for index, payload in enumerate(payloads):
            futures.append(pool.submit(post_stream, args.base, payload, args.timeout_s))
            if index + 1 < len(payloads):
                time.sleep(args.stagger_s)
        results = [future.result() for future in futures]
    wall_s = time.perf_counter() - wall_t0
    report = summarize_results(
        results, args.requests, args.workers, wall_s, args.label)

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
