#!/usr/bin/env python3
"""Run a privacy-safe deterministic decode probe for the M1-65 service A/B."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import time
from typing import Any
import urllib.request

import quality_runtime_contract as runtime_contract


Json = dict[str, Any]
SCHEMA = "bi100-gdn-combined-qk-decode-v1"
VERSION = 1
PROMPT = (
    "Write a continuous sequence of short English words separated by spaces. "
    "Continue until the requested token budget is exhausted."
)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write(path: Path, value: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile_value / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _error_row(index: int, started: float, error: BaseException) -> Json:
    return {
        "index": index,
        "ok": False,
        "http_status": None,
        "elapsed_s": time.perf_counter() - started,
        "ttft_s": None,
        "decode_s": None,
        "output_tps": None,
        "prompt_tokens": None,
        "cached_tokens": None,
        "completion_tokens": None,
        "finish_reason": None,
        "content_chars": None,
        "reasoning_chars": None,
        "tool_call_fragments": None,
        "first_output_sha256": None,
        "semantic_output_sha256": None,
        "error_type": type(error).__name__,
        "error_sha256": hashlib.sha256(
            str(error).encode("utf-8", "replace")).hexdigest(),
    }


def stream_once(
    base: str,
    *,
    index: int,
    tokens: int,
    seed: int,
    timeout_s: float,
) -> Json:
    payload = {
        "model": "llm",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": tokens,
        "min_tokens": tokens,
        "temperature": 0,
        "seed": seed,
        "thinking": False,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    content: list[str] = []
    reasoning: list[str] = []
    tool_fragments: list[Any] = []
    first_output: Json | None = None
    first_output_time: float | None = None
    last_output_time: float | None = None
    finish_reason: str | None = None
    usage: Json = {}
    done_count = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status = int(response.status)
            if status != 200:
                raise RuntimeError(f"unexpected HTTP status {status}")
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done_count += 1
                    continue
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason") is not None:
                    finish_reason = str(choice["finish_reason"])
                delta = choice.get("delta") or {}
                safe_delta = {
                    "content": delta.get("content") or "",
                    "reasoning_content": delta.get("reasoning_content") or "",
                    "tool_calls": delta.get("tool_calls") or [],
                }
                if not any(safe_delta.values()):
                    continue
                now = time.perf_counter()
                if first_output_time is None:
                    first_output_time = now
                    first_output = safe_delta
                last_output_time = now
                content.append(safe_delta["content"])
                reasoning.append(safe_delta["reasoning_content"])
                tool_fragments.extend(safe_delta["tool_calls"])
        elapsed_s = time.perf_counter() - started
        if done_count != 1:
            raise RuntimeError(f"expected one DONE event, got {done_count}")
        if first_output_time is None or last_output_time is None:
            raise RuntimeError("stream produced no output event")
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        cached_tokens = int(
            (usage.get("prompt_tokens_details") or {}).get(
                "cached_tokens") or 0)
        decode_s = max(0.0, last_output_time - first_output_time)
        output_tps = (
            (completion_tokens - 1) / decode_s
            if completion_tokens > 1 and decode_s > 0 else 0.0)
        semantic = {
            "content": "".join(content),
            "reasoning_content": "".join(reasoning),
            "tool_calls": tool_fragments,
        }
        return {
            "index": index,
            "ok": True,
            "http_status": status,
            "elapsed_s": elapsed_s,
            "ttft_s": first_output_time - started,
            "decode_s": decode_s,
            "output_tps": output_tps,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "completion_tokens": completion_tokens,
            "finish_reason": finish_reason,
            "content_chars": len(semantic["content"]),
            "reasoning_chars": len(semantic["reasoning_content"]),
            "tool_call_fragments": len(tool_fragments),
            "first_output_sha256": _sha256_json(first_output),
            "semantic_output_sha256": _sha256_json(semantic),
            "error_type": "",
            "error_sha256": None,
        }
    except BaseException as error:  # noqa: BLE001 - persist safe failure data.
        return _error_row(index, started, error)


def summarize(rows: list[Json], tokens: int) -> Json:
    valid_rates = [
        float(row["output_tps"]) for row in rows
        if row.get("ok") and isinstance(row.get("output_tps"), (int, float))
        and math.isfinite(float(row["output_tps"]))
    ]
    digests = {
        row.get("semantic_output_sha256") for row in rows if row.get("ok")
    }
    qualified_rows = [
        row for row in rows
        if (
            row.get("ok") is True
            and row.get("http_status") == 200
            and row.get("completion_tokens") == tokens
            and row.get("finish_reason") == "length"
            and isinstance(row.get("output_tps"), (int, float))
            and math.isfinite(float(row["output_tps"]))
            and float(row["output_tps"]) > 0
        )
    ]
    return {
        "requests": len(rows),
        "successful_requests": len(qualified_rows),
        "success_rate": (
            len(qualified_rows) / len(rows) if rows else 0.0),
        "repeated_output_exact": len(digests) == 1 and bool(digests),
        "output_tps_p10": percentile(valid_rates, 10),
        "output_tps_median": (
            statistics.median(valid_rates) if valid_rates else 0.0),
        "output_tps_mean": (
            statistics.mean(valid_rates) if valid_rates else 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--runtime-identity", required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--timeout-s", type=float, default=1200.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.requests < 3:
        parser.error("requests must be at least 3")
    if args.warmup < 1:
        parser.error("warmup must be at least 1")
    if args.tokens < 256:
        parser.error("tokens must be at least 256")
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("timeout-s must be finite and positive")

    expected_runtime = {
        "source_revision": args.source_revision,
        "runtime_identity": args.runtime_identity,
        "instance": args.instance,
        "gpu_count": 4,
        "tensor_parallel_size": 4,
        "max_model_len": 262144,
    }
    contract, contract_sha = runtime_contract.load_runtime_contract(
        args.runtime_contract,
        expected_runtime,
        require_cache_trace=True,
    )
    report: Json = {
        "schema": SCHEMA,
        "version": VERSION,
        "qualified": False,
        "production_promotion_authorized": False,
        "label": args.label,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runtime": {
            "source_revision": args.source_revision,
            "runtime_identity": args.runtime_identity,
            "runtime_overlay_sha256": contract["runtime_overlay_sha256"],
            "instance": args.instance,
            "gpu_count": 4,
            "tensor_parallel_size": 4,
            "max_model_len": 262144,
        },
        "runtime_contract": {
            "sha256": contract_sha,
            "contract": contract,
        },
        "config": {
            "requests": args.requests,
            "warmup": args.warmup,
            "tokens": args.tokens,
            "seed": args.seed,
            "timeout_s": args.timeout_s,
            "prompt_sha256": hashlib.sha256(PROMPT.encode("ascii")).hexdigest(),
        },
        "generator": {
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "privacy": {
            "contains_raw_requests": False,
            "contains_raw_model_outputs": False,
            "contains_credentials": False,
        },
        "warmup": [],
        "requests": [],
        "summary": {},
    }
    _atomic_write(args.out, report)
    for index in range(args.warmup):
        row = stream_once(
            args.base,
            index=index,
            tokens=args.tokens,
            seed=args.seed,
            timeout_s=args.timeout_s,
        )
        report["warmup"].append(row)
        _atomic_write(args.out, report)
        if not row["ok"]:
            return 1
    for index in range(args.requests):
        row = stream_once(
            args.base,
            index=index,
            tokens=args.tokens,
            seed=args.seed,
            timeout_s=args.timeout_s,
        )
        report["requests"].append(row)
        report["summary"] = summarize(report["requests"], args.tokens)
        _atomic_write(args.out, report)
    report["qualified"] = (
        report["summary"]["successful_requests"] == args.requests
        and report["summary"]["repeated_output_exact"] is True)
    _atomic_write(args.out, report)
    print(json.dumps({
        "qualified": report["qualified"],
        "label": report["label"],
        "summary": report["summary"],
        "out": str(args.out),
    }, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
