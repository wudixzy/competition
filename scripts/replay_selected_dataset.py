#!/usr/bin/env python3
"""Replay the selected chat dataset and emit privacy-redacted metrics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence
import urllib.error
import urllib.request


@dataclass
class StreamResult:
    ok: bool
    content: str
    latency_s: float
    ttft_s: float | None
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    finish_reason: str | None
    error_kind: str = ""


def percentile(values: Sequence[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _error_kind(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTPError:{exc.code}"
    return type(exc).__name__


def post_stream(
    base: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> StreamResult:
    started = time.perf_counter()
    request = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    content_parts: list[str] = []
    usage: dict[str, Any] = {}
    ttft_s: float | None = None
    finish_reason: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").strip()
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
                choice = choices[0]
                if choice.get("finish_reason") is not None:
                    finish_reason = str(choice["finish_reason"])
                delta = choice.get("delta") or {}
                has_output = bool(
                    delta.get("content")
                    or delta.get("reasoning_content")
                    or delta.get("tool_calls")
                )
                if has_output and ttft_s is None:
                    ttft_s = time.perf_counter() - started
                if isinstance(delta.get("content"), str):
                    content_parts.append(delta["content"])
        latency_s = time.perf_counter() - started
        details = usage.get("prompt_tokens_details") or {}
        content = "".join(content_parts)
        ok = bool(content) and ttft_s is not None
        return StreamResult(
            ok=ok,
            content=content,
            latency_s=latency_s,
            ttft_s=ttft_s,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            cached_tokens=int(details.get("cached_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            finish_reason=finish_reason,
            error_kind="" if ok else "EmptyAssistantContent",
        )
    except Exception as exc:  # noqa: BLE001 - retain a redacted failure class.
        return StreamResult(
            ok=False,
            content="",
            latency_s=time.perf_counter() - started,
            ttft_s=ttft_s,
            prompt_tokens=0,
            cached_tokens=0,
            completion_tokens=0,
            finish_reason=None,
            error_kind=_error_kind(exc),
        )


def redacted_turn(
    conversation_index: int,
    turn_index: int,
    request_message_count: int,
    result: StreamResult,
) -> dict[str, Any]:
    decode_window_s = (
        max(0.0, result.latency_s - result.ttft_s)
        if result.ttft_s is not None else 0.0
    )
    return {
        "conversation": conversation_index,
        "turn": turn_index,
        "request_message_count": request_message_count,
        "ok": result.ok,
        "error_kind": result.error_kind,
        "finish_reason": result.finish_reason,
        "prompt_tokens": result.prompt_tokens,
        "cached_tokens": result.cached_tokens,
        "uncached_prompt_tokens": max(
            0, result.prompt_tokens - result.cached_tokens),
        "completion_tokens": result.completion_tokens,
        "ttft_s": result.ttft_s,
        "latency_s": result.latency_s,
        "output_tps_decode": (
            result.completion_tokens / decode_window_s
            if decode_window_s > 0 else 0.0
        ),
        "content_sha256": (
            hashlib.sha256(result.content.encode("utf-8")).hexdigest()
            if result.content else None
        ),
    }


def summarize(
    label: str,
    dataset_path: Path,
    dataset_sha256: str,
    expected_conversations: int,
    expected_turns: int,
    turns: list[dict[str, Any]],
    wall_s: float,
) -> dict[str, Any]:
    successful = [turn for turn in turns if turn["ok"]]
    ttfts = [
        float(turn["ttft_s"])
        for turn in successful if turn["ttft_s"] is not None
    ]
    output_rates = [
        float(turn["output_tps_decode"])
        for turn in successful if turn["output_tps_decode"] > 0
    ]
    prompt_tokens = sum(int(turn["prompt_tokens"]) for turn in successful)
    cached_tokens = sum(int(turn["cached_tokens"]) for turn in successful)
    uncached_tokens = sum(
        int(turn["uncached_prompt_tokens"]) for turn in successful)
    ttft_total_s = sum(ttfts)
    output_tps_p10 = percentile(output_rates, 10)
    input_tps_proxy = (
        uncached_tokens / ttft_total_s if ttft_total_s > 0 else 0.0)
    cache_tps_proxy = (
        cached_tokens / ttft_total_s if ttft_total_s > 0 else 0.0)
    weighted_proxy = (
        output_tps_p10 * 16.796
        + input_tps_proxy * 2.799
        + cache_tps_proxy * 0.56
    )
    success_rate = (
        len(successful) / expected_turns if expected_turns else 0.0)
    return {
        "schema": "bi100-selected-dataset-replay-v1",
        "label": label,
        "dataset": {
            "path_name": dataset_path.name,
            "sha256": dataset_sha256,
            "conversation_count": expected_conversations,
            "turn_count": expected_turns,
        },
        "privacy": {
            "contains_raw_messages": False,
            "contains_raw_model_output": False,
        },
        "validation": {
            "complete_replay": len(turns) == expected_turns,
            "success_rate": success_rate,
            "all_successful": len(successful) == expected_turns,
        },
        "aggregate": {
            "wall_s": wall_s,
            "ttft_p90_s": percentile(ttfts, 90),
            "output_tps_p10": output_tps_p10,
            "cache_hit_rate": (
                cached_tokens / prompt_tokens if prompt_tokens else 0.0),
            "input_tps_residual_proxy": input_tps_proxy,
            "cache_tps_proxy": cache_tps_proxy,
            "weighted_score_proxy": weighted_proxy,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "uncached_prompt_tokens": uncached_tokens,
            "completion_tokens": sum(
                int(turn["completion_tokens"]) for turn in successful),
        },
        "turns": turns,
    }


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("dataset root must be a list")
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"conversation {index} must be an object")
        if not isinstance(item.get("user_questions"), list):
            raise ValueError(
                f"conversation {index} user_questions must be a list")
        if not all(isinstance(value, str)
                   for value in item["user_questions"]):
            raise ValueError(
                f"conversation {index} questions must be strings")
        if not isinstance(item.get("system_prompt", ""), str):
            raise ValueError(
                f"conversation {index} system_prompt must be a string")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
                        default=Path("chat_dataset_v0.json"))
    parser.add_argument("--label", required=True)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--timeout-s", type=float, default=900)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    dataset_bytes = args.dataset.read_bytes()
    dataset = load_dataset(args.dataset)
    expected_turns = sum(
        len(item["user_questions"]) for item in dataset)
    turns: list[dict[str, Any]] = []
    wall_started = time.perf_counter()
    for conversation_index, item in enumerate(dataset):
        messages: list[dict[str, Any]] = []
        if item.get("system_prompt"):
            messages.append({
                "role": "system",
                "content": item["system_prompt"],
            })
        for turn_index, question in enumerate(item["user_questions"]):
            messages.append({"role": "user", "content": question})
            payload = {
                "model": "llm",
                "messages": messages,
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "thinking": False,
                "seed": args.seed,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            result = post_stream(args.base, payload, args.timeout_s)
            turns.append(redacted_turn(
                conversation_index,
                turn_index,
                len(messages),
                result,
            ))
            if not result.ok:
                break
            messages.append({"role": "assistant", "content": result.content})

    report = summarize(
        args.label,
        args.dataset,
        hashlib.sha256(dataset_bytes).hexdigest(),
        len(dataset),
        expected_turns,
        turns,
        time.perf_counter() - wall_started,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "label": args.label,
        "validation": report["validation"],
        "aggregate": report["aggregate"],
        "out": str(args.out),
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["validation"]["all_successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
