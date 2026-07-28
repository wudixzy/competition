#!/usr/bin/env python3
"""Run the fixed privacy-safe M1-104 dataset-shaped policy matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any
import urllib.error
import urllib.request


SCHEMA = "bi100-m1-104-admission64-policy-matrix-v2"
VERSION = 2
SHAPES = (4096, 7800, 16000)
PAIRS = (1, 2, 3)
PHASES = ("cold", "warm")
REQUEST_COUNT = len(SHAPES) * len(PAIRS) * len(PHASES)
SEED = 20260721
TOOL_COUNT = 29
MAX_TOKENS = 64
TOKEN_ERROR_LIMIT = 16
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = (
    ROOT / "qwen3_6_scripts" / "qwen3_5.py",
    ROOT / "docs" / "HANDOFF_SUMMARY.md",
    ROOT / "tests" / "bench_perf.py",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8"))


def _percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def make_tools(count: int = TOOL_COUNT) -> list[dict[str, Any]]:
    names = (
        "read_file",
        "search_code",
        "run_command",
        "edit_file",
        "web_search",
        "list_directory",
        "inspect_process",
    )
    return [
        {
            "type": "function",
            "function": {
                "name": f"{names[index % len(names)]}_{index}",
                "description": (
                    "Agent engineering tool for repository inspection, exact "
                    "edits, command execution, and structured result capture."
                ),
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
        }
        for index in range(count)
    ]


def load_corpus(paths: list[Path]) -> tuple[str, list[dict[str, str]]]:
    pieces: list[str] = []
    manifest = []
    for path in paths:
        value = path.read_bytes()
        pieces.extend((
            f"\n===== {path.name} =====\n",
            value.decode("utf-8", "replace"),
        ))
        manifest.append({
            "name": path.name,
            "sha256": _sha256_bytes(value),
        })
    return "".join(pieces), manifest


def build_prompt(
    tokenizer: Any,
    corpus: str,
    target: int,
    salt: str,
    tools: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], int]:
    corpus_ids = tokenizer.encode(corpus, add_special_tokens=False)
    system = (
        f"RUN_ID={salt}. You are a coding agent. Inspect the supplied "
        "repository material, preserve exact identifiers, and produce a "
        "concise implementation plan."
    )

    def messages(count: int) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": tokenizer.decode(corpus_ids[:count]),
            },
        ]

    low, high = 0, len(corpus_ids)
    while low < high:
        middle = (low + high + 1) // 2
        rendered = tokenizer.apply_chat_template(
            messages(middle),
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if len(rendered) <= target:
            low = middle
        else:
            high = middle - 1
    result = messages(low)
    rendered_tokens = len(tokenizer.apply_chat_template(
        result,
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    ))
    return result, rendered_tokens


def service_health(base: str, timeout_s: float) -> bool:
    request = urllib.request.Request(
        f"{base.rstrip('/')}/health",
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def stream_request(
    base: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    request = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    status = 0
    usage: dict[str, Any] = {}
    finish_reason = None
    content: list[str] = []
    reasoning: list[str] = []
    tool_deltas: list[Any] = []
    first_identity = None
    ttft_s = None
    done_seen = False
    terminal_choice_seen = False
    usage_seen = False
    data_event_count = 0
    malformed_sse_count = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status = response.status
            if status != 200:
                return {
                    "ok": False,
                    "http_status": status,
                    "error_type": "HttpStatus",
                }
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line or line.startswith(":"):
                    continue
                if done_seen:
                    malformed_sse_count += 1
                    continue
                if not line.startswith("data:"):
                    malformed_sse_count += 1
                    continue
                value = line[5:].strip()
                if value == "[DONE]":
                    done_seen = True
                    continue
                event = json.loads(value)
                if not isinstance(event, dict):
                    raise ValueError("SSE data event is not an object")
                data_event_count += 1
                if event.get("usage"):
                    if not isinstance(event["usage"], dict):
                        raise ValueError("SSE usage is not an object")
                    usage = event["usage"]
                    usage_seen = True
                choices = event.get("choices") or []
                if not isinstance(choices, list):
                    raise ValueError("SSE choices is not a list")
                if not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, dict):
                    raise ValueError("SSE choice is not an object")
                if choice.get("finish_reason") is not None:
                    terminal_choice_seen = True
                finish_reason = (
                    choice.get("finish_reason")
                    if choice.get("finish_reason") is not None
                    else finish_reason
                )
                delta = choice.get("delta") or {}
                output = (
                    delta.get("content")
                    or delta.get("reasoning_content")
                    or delta.get("tool_calls")
                )
                if not output:
                    continue
                elapsed = time.perf_counter() - started
                if ttft_s is None:
                    ttft_s = elapsed
                    first_identity = {
                        "content": delta.get("content") or "",
                        "reasoning_content":
                            delta.get("reasoning_content") or "",
                        "tool_calls": delta.get("tool_calls") or [],
                    }
                if delta.get("content"):
                    content.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                if delta.get("tool_calls"):
                    tool_deltas.extend(delta["tool_calls"])
        elapsed_s = time.perf_counter() - started
        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_tokens = int(usage.get("completion_tokens") or 0)
        normalized = {
            "content": "".join(content),
            "reasoning_content": "".join(reasoning),
            "tool_call_deltas": tool_deltas,
        }
        decode_window_s = (
            max(elapsed_s - ttft_s, 0.0)
            if ttft_s is not None
            else 0.0
        )
        return {
            "ok": True,
            "http_status": status,
            "done_seen": done_seen,
            "terminal_choice_seen": terminal_choice_seen,
            "usage_seen": usage_seen,
            "data_event_count": data_event_count,
            "malformed_sse_count": malformed_sse_count,
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "cached_tokens": int(
                prompt_details.get("cached_tokens") or 0),
            "completion_tokens": completion_tokens,
            "finish_reason": finish_reason,
            "ttft_s": ttft_s,
            "latency_s": elapsed_s,
            "decode_window_s": decode_window_s,
            "output_tps": (
                completion_tokens / decode_window_s
                if decode_window_s > 0 else 0.0
            ),
            "first_token_sha256": (
                _sha256_json(first_identity)
                if first_identity is not None else None
            ),
            "output_sha256": _sha256_json(normalized),
            "content_sha256": _sha256_bytes(
                normalized["content"].encode("utf-8")),
            "reasoning_sha256": _sha256_bytes(
                normalized["reasoning_content"].encode("utf-8")),
            "tool_calls_sha256": _sha256_json(tool_deltas),
            "tool_call_delta_count": len(tool_deltas),
        }
    except urllib.error.HTTPError as error:
        return {
            "ok": False,
            "http_status": error.code,
            "error_type": type(error).__name__,
        }
    except (OSError, urllib.error.URLError, ValueError) as error:
        return {
            "ok": False,
            "http_status": status,
            "error_type": type(error).__name__,
        }


def request_contract(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record.get(field) for field in (
        "request_id",
        "target_prompt_tokens",
        "pair",
        "phase",
        "salt_sha256",
        "rendered_tokens_local",
        "seed",
    ))


def validate_requests(records: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    expected = {
        (target, pair, phase)
        for target in SHAPES
        for pair in PAIRS
        for phase in PHASES
    }
    observed = {
        (
            record.get("target_prompt_tokens"),
            record.get("pair"),
            record.get("phase"),
        )
        for record in records
    }
    if len(records) != REQUEST_COUNT or observed != expected:
        reasons.append("the fixed 18-request matrix is incomplete")
    if len({record.get("request_id") for record in records}) != len(records):
        reasons.append("request ids are not unique")

    for index, record in enumerate(records):
        label = f"request[{index}]"
        if (
            record.get("ok") is not True
            or record.get("http_status") != 200
            or record.get("done_seen") is not True
            or record.get("terminal_choice_seen") is not True
            or record.get("usage_seen") is not True
            or record.get("malformed_sse_count") != 0
            or not isinstance(record.get("data_event_count"), int)
            or record["data_event_count"] <= 0
            or record.get("health_after") is not True
        ):
            reasons.append(f"{label} request or health failed")
            continue
        target = record.get("target_prompt_tokens")
        rendered = record.get("rendered_tokens_local")
        prompt = record.get("prompt_tokens")
        if (
            not isinstance(target, int)
            or not isinstance(rendered, int)
            or not isinstance(prompt, int)
            or prompt != rendered
            or abs(prompt - target) >= TOKEN_ERROR_LIMIT
        ):
            reasons.append(f"{label} prompt token contract differs")
        if (
            not isinstance(record.get("completion_tokens"), int)
            or not 0 < record["completion_tokens"] <= MAX_TOKENS
            or record.get("finish_reason") not in {"stop", "length"}
            or record.get("tool_call_delta_count") != 0
            or record.get("tool_calls_sha256") != _sha256_json([])
        ):
            reasons.append(f"{label} completion contract differs")
        for field in ("ttft_s", "latency_s", "decode_window_s", "output_tps"):
            value = record.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                reasons.append(f"{label} {field} is not finite and positive")
        ttft_s = record.get("ttft_s")
        latency_s = record.get("latency_s")
        completion_tokens = record.get("completion_tokens")
        if all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in (ttft_s, latency_s, completion_tokens)
        ):
            expected_window = float(latency_s) - float(ttft_s)
            observed_window = record.get("decode_window_s")
            observed_tps = record.get("output_tps")
            if (
                expected_window <= 0
                or not isinstance(observed_window, (int, float))
                or isinstance(observed_window, bool)
                or not math.isclose(
                    float(observed_window),
                    expected_window,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                or not isinstance(observed_tps, (int, float))
                or isinstance(observed_tps, bool)
                or not math.isclose(
                    float(observed_tps),
                    int(completion_tokens) / expected_window,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                reasons.append(f"{label} decode timing formula differs")
        for field in (
            "salt_sha256",
            "first_token_sha256",
            "output_sha256",
            "content_sha256",
            "reasoning_sha256",
            "tool_calls_sha256",
        ):
            if not SHA256_RE.fullmatch(str(record.get(field) or "")):
                reasons.append(f"{label} {field} is invalid")

    for target in SHAPES:
        for pair in PAIRS:
            rows = {
                record.get("phase"): record
                for record in records
                if record.get("target_prompt_tokens") == target
                and record.get("pair") == pair
            }
            if set(rows) != set(PHASES):
                continue
            cold, warm = rows["cold"], rows["warm"]
            if (
                not isinstance(cold.get("cached_tokens"), int)
                or not isinstance(warm.get("cached_tokens"), int)
                or cold["cached_tokens"] < 0
                or warm["cached_tokens"] < cold["cached_tokens"]
            ):
                reasons.append(
                    f"target={target} pair={pair} cache progression differs")
            for field in (
                "salt_sha256",
                "first_token_sha256",
                "output_sha256",
                "finish_reason",
                "completion_tokens",
            ):
                if cold.get(field) != warm.get(field):
                    reasons.append(
                        f"target={target} pair={pair} cold/warm {field} differs")
    if records and records[0].get("cached_tokens") != 0:
        reasons.append("the first request of a fresh service is not cold")
    return reasons


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record.get("ok") is True]
    cold = [
        record for record in successful if record.get("phase") == "cold"]
    warm = [
        record for record in successful if record.get("phase") == "warm"]
    cold_ttft = sum(float(record.get("ttft_s") or 0) for record in cold)
    warm_ttft = sum(float(record.get("ttft_s") or 0) for record in warm)
    prompt_tokens = sum(
        int(record.get("prompt_tokens") or 0) for record in successful)
    cached_tokens = sum(
        int(record.get("cached_tokens") or 0) for record in successful)
    input_tps = (
        sum(int(record.get("prompt_tokens") or 0) for record in cold)
        / cold_ttft if cold_ttft > 0 else 0.0
    )
    cache_tps = (
        sum(int(record.get("cached_tokens") or 0) for record in warm)
        / warm_ttft if warm_ttft > 0 else 0.0
    )
    output_tps_p10 = _percentile([
        float(record.get("output_tps") or 0) for record in successful
    ], 10)
    weighted = (
        output_tps_p10 * 16.796
        + input_tps * 2.799
        + cache_tps * 0.56
    )
    return {
        "success_rate": (
            len(successful) / REQUEST_COUNT if REQUEST_COUNT else 0.0),
        "output_tps_p10": output_tps_p10,
        "input_tps": input_tps,
        "cache_tps": cache_tps,
        "ttft_p90_s": _percentile([
            float(record.get("ttft_s") or 0) for record in successful
        ], 90),
        "effective_hit_rate": (
            cached_tokens / prompt_tokens if prompt_tokens else 0.0),
        "weighted": weighted,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "cold_cached_tokens": sum(
            int(record.get("cached_tokens") or 0) for record in cold),
        "first_request_cached_tokens": (
            int(records[0].get("cached_tokens") or 0)
            if records else 0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=("fine32", "admission64"),
        required=True,
    )
    parser.add_argument(
        "--ab-pair",
        type=int,
        choices=PAIRS,
        required=True,
    )
    parser.add_argument("--salt-namespace", required=True)
    parser.add_argument("--corpus", type=Path, nargs="+")
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("--timeout-s must be finite and positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}",
                        args.salt_namespace):
        parser.error("--salt-namespace must be a short non-sensitive label")

    from transformers import AutoTokenizer

    corpus_paths = args.corpus or list(DEFAULT_CORPUS)
    corpus, corpus_manifest = load_corpus(corpus_paths)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    tools = make_tools()
    records = []
    for target in SHAPES:
        for pair in PAIRS:
            salt = f"{args.salt_namespace}:{target}:{pair}"
            messages, rendered_tokens = build_prompt(
                tokenizer, corpus, target, salt, tools)
            for phase in PHASES:
                payload = {
                    "model": "llm",
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "none",
                    "thinking": False,
                    "temperature": 0,
                    "seed": SEED,
                    "max_tokens": MAX_TOKENS,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                record = stream_request(args.base, payload, args.timeout_s)
                record.update({
                    "request_id": f"{target}_pair{pair}_{phase}",
                    "target_prompt_tokens": target,
                    "pair": pair,
                    "phase": phase,
                    "salt_sha256": _sha256_bytes(salt.encode("utf-8")),
                    "rendered_tokens_local": rendered_tokens,
                    "seed": SEED,
                    "health_after": service_health(args.base, 5.0),
                })
                records.append(record)

    reasons = validate_requests(records)
    contract = [request_contract(record) for record in records]
    report = {
        "schema": SCHEMA,
        "version": VERSION,
        "mode": (
            "control" if args.policy == "fine32" else "candidate"),
        "policy": args.policy,
        "ab_pair": args.ab_pair,
        "request_count": len(records),
        "request_manifest_sha256": _sha256_json(contract),
        "target_order": [
            record["request_id"] for record in records],
        "fixed": {
            "shapes": list(SHAPES),
            "pairs": list(PAIRS),
            "phases": list(PHASES),
            "seed": SEED,
            "tool_count": TOOL_COUNT,
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "thinking": False,
            "tool_choice": "none",
            "stream_usage": True,
            "salt_namespace_sha256": _sha256_bytes(
                args.salt_namespace.encode("utf-8")),
            "corpus": corpus_manifest,
        },
        "aggregate": aggregate(records),
        "qualified_measurement": not reasons,
        "reasons": reasons,
        "requests": records,
        "privacy": {
            "contains_raw_prompt": False,
            "contains_raw_output": False,
            "contains_tools": False,
            "contains_credentials": False,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report["qualified_measurement"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
