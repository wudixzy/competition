#!/usr/bin/env python3
"""One-startup short TP4 integration screen for a mature operator candidate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time

from bench_fused_prefill_service import _post_stream
from long_context_api import build_exact_prompt


def main() -> int:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--model-path",
        default="/root/public-storage/models/Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--targets", default="4096,32768,65536")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--selector", choices=("control", "candidate"),
                        required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    targets = [int(value) for value in args.targets.split(",")]
    if (
        targets != [4096, 32768, 65536]
        or args.max_tokens != 8
        or not math.isfinite(args.timeout_s)
        or args.timeout_s <= 0
    ):
        parser.error("short TP4 screen parameters differ from the contract")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True)
    started = time.monotonic()
    cases = []
    reasons = []
    for target in targets:
        content = build_exact_prompt(
            tokenizer, target, f"{args.run_id}-{target}")
        payload = {
            "model": "llm",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": args.max_tokens,
            "min_tokens": args.max_tokens,
            "temperature": 0,
            "seed": 20260730,
            "thinking": False,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        cold = _post_stream(
            args.base, payload, args.timeout_s, tokenizer)
        warm = _post_stream(
            args.base, payload, args.timeout_s, tokenizer)
        if cold["prompt_tokens"] != target or warm["prompt_tokens"] != target:
            reasons.append(f"{target}: prompt token count differs")
        if cold["cached_tokens"] != 0:
            reasons.append(f"{target}: cold request was not cold")
        if warm["cached_tokens"] < target - 32:
            reasons.append(f"{target}: warm prefix was not retained")
        for field in (
            "first_token_sha256",
            "output_sha256",
            "completion_tokens",
            "finish_reason",
        ):
            if cold[field] != warm[field]:
                reasons.append(f"{target}: cold/warm {field} differs")
        cases.append({
            "target_prompt_tokens": target,
            "cold": cold,
            "warm": warm,
        })
    report = {
        "schema": "bi100-short-tp4-funnel-service-v1",
        "version": 1,
        "run_id": args.run_id,
        "selector": args.selector,
        "targets": targets,
        "max_tokens": args.max_tokens,
        "elapsed_s": time.monotonic() - started,
        "qualified": not reasons,
        "reasons": reasons,
        "cases": cases,
        "cold_ttft_median_s": statistics.median(
            case["cold"]["ttft_s"] for case in cases),
        "warm_ttft_median_s": statistics.median(
            case["warm"]["ttft_s"] for case in cases),
        "privacy": {
            "contains_prompts": False,
            "contains_model_outputs": False,
            "contains_token_ids": False,
            "contains_credentials": False,
        },
        "authorization": {
            "long_context_authorized": False,
            "main_or_yaml_change_authorized": False,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="ascii",
    )
    print(json.dumps({
        "qualified": report["qualified"],
        "selector": args.selector,
        "elapsed_s": report["elapsed_s"],
        "cold_ttft_median_s": report["cold_ttft_median_s"],
    }, ensure_ascii=True, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
