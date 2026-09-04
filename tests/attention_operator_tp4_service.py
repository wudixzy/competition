#!/usr/bin/env python3
"""Focused cold-only TP4 service population for an attention operator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

from bench_fused_prefill_service import _post_stream, stream_timing_metrics
from long_context_api import build_exact_prompt


SCHEMA = "bi100-attention-operator-tp4-service-v1"
TARGETS = (16384, 32768, 65536)
REPETITIONS = 3
MAX_TOKENS = 8
SEED = 20260904
WORKLOAD_ORDER = "target_ascending_then_repetition_ascending"


def _payload(content: str) -> dict[str, Any]:
    return {
        "model": "llm",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": MAX_TOKENS,
        "min_tokens": MAX_TOKENS,
        "temperature": 0,
        "seed": SEED,
        "thinking": False,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def summarize_response(raw: dict[str, Any]) -> dict[str, Any]:
    completion = raw["completion_tokens"]
    timing = stream_timing_metrics(
        completion, raw["ttft_s"], raw["last_output_s"])
    return {
        "ok": raw["ok"],
        "http_status": 200,
        "sse_complete": True,
        "usage_complete": True,
        "elapsed_s": raw["elapsed_s"],
        "ttft_s": raw["ttft_s"],
        "last_output_s": raw["last_output_s"],
        "decode_window_s": timing["decode_window_s"],
        "tpot_s": timing["tpot_s"],
        "output_tps": timing["output_tps"],
        "prompt_tokens": raw["prompt_tokens"],
        "completion_tokens": completion,
        "finish_reason": raw["finish_reason"],
    }


def response_reasons(value: Any, target: int) -> list[str]:
    if not isinstance(value, dict):
        return ["response is not an object"]
    reasons = []
    for name in ("elapsed_s", "ttft_s", "last_output_s", "decode_window_s",
                 "tpot_s", "output_tps"):
        item = value.get(name)
        if (not isinstance(item, (int, float)) or isinstance(item, bool)
                or not math.isfinite(float(item)) or float(item) < 0.0):
            reasons.append(f"{name} is not finite and non-negative")
    if value.get("ttft_s", 0.0) <= 0.0:
        reasons.append("TTFT is not positive")
    if (value.get("ok") is not True or value.get("http_status") != 200
            or value.get("sse_complete") is not True
            or value.get("usage_complete") is not True):
        reasons.append("HTTP/SSE/usage contract differs")
    if value.get("prompt_tokens") != target:
        reasons.append("prompt token count differs")
    if value.get("completion_tokens") != MAX_TOKENS:
        reasons.append("completion token count differs")
    if value.get("finish_reason") != "length":
        reasons.append("finish reason differs")
    return reasons


def evaluate(report: Any) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {"qualified": False, "reasons": ["report is not an object"]}
    cases = report.get("cases")
    expected = len(TARGETS) * REPETITIONS
    if (report.get("schema") != SCHEMA or report.get("version") != 1
            or report.get("change_scope") != "attention_operator"
            or report.get("targets") != list(TARGETS)
            or report.get("repetitions") != REPETITIONS
            or report.get("max_tokens") != MAX_TOKENS
            or report.get("seed") != SEED
            or report.get("workload_order") != WORKLOAD_ORDER
            or report.get("expected_requests") != expected
            or report.get("attempted_requests") != expected
            or report.get("completed_requests") != expected
            or report.get("failed_requests") != 0
            or not isinstance(cases, list) or len(cases) != expected):
        return {"qualified": False,
                "reasons": ["focused request population is incomplete"]}
    reasons = []
    for index, case in enumerate(cases):
        target = TARGETS[index // REPETITIONS]
        repetition = index % REPETITIONS
        if (not isinstance(case, dict)
                or case.get("target_prompt_tokens") != target
                or case.get("repetition") != repetition):
            reasons.append(f"case {index} identity differs")
            continue
        reasons.extend(
            f"{target}/rep-{repetition}: {reason}"
            for reason in response_reasons(case.get("response"), target))
    return {"qualified": not reasons, "reasons": reasons}


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True)
    started = time.monotonic()
    cases = []
    for target in TARGETS:
        for repetition in range(REPETITIONS):
            content = build_exact_prompt(
                tokenizer, target,
                f"{args.workload_id}-{target}-{repetition}")
            raw = _post_stream(
                args.base, _payload(content), args.timeout_s, tokenizer)
            cases.append({
                "target_prompt_tokens": target,
                "repetition": repetition,
                "response": summarize_response(raw),
            })
    elapsed = time.monotonic() - started
    ttft = [case["response"]["ttft_s"] for case in cases]
    report = {
        "schema": SCHEMA,
        "version": 1,
        "change_scope": "attention_operator",
        "selector": args.selector,
        "run_id": args.run_id,
        "workload_id": args.workload_id,
        "targets": list(TARGETS),
        "repetitions": REPETITIONS,
        "max_tokens": MAX_TOKENS,
        "seed": SEED,
        "workload_order": WORKLOAD_ORDER,
        "expected_requests": len(TARGETS) * REPETITIONS,
        "attempted_requests": len(cases),
        "completed_requests": len(cases),
        "failed_requests": 0,
        "elapsed_s": elapsed,
        "raw_ttft_s": ttft,
        "ttft_median_s": statistics.median(ttft),
        "cases": cases,
        "metric_definitions": {
            "ttft": "request_start_to_first_output_token",
            "tpot": "first_to_last_output_time_divided_by_completion_tokens_minus_one",
            "output_tps": "completion_tokens_minus_one_divided_by_first_to_last_output_time",
        },
        "privacy": {
            "prompts_recorded": False,
            "model_outputs_recorded": False,
            "token_ids_recorded": False,
            "credentials_recorded": False,
        },
    }
    report["evaluation"] = evaluate(report)
    report["qualified"] = report["evaluation"]["qualified"]
    report["reasons"] = report["evaluation"]["reasons"]
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workload-id", required=True)
    parser.add_argument("--selector", choices=("control", "candidate"),
                        required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("timeout must be finite and positive")
    return args


def main() -> int:
    args = parse_args()
    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "qualified": report["qualified"],
        "selector": args.selector,
        "requests": report["completed_requests"],
        "ttft_median_s": report["ttft_median_s"],
    }, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
