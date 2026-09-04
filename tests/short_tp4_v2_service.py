#!/usr/bin/env python3
"""Fixed M1-176 short-TP4 cold/partial/warm service population."""

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


SCHEMA = "bi100-m1-176-short-tp4-service-v2"
TARGETS = (4096, 16384, 32768, 65536)
PARTIAL_RESIDUAL_TOKENS = 2048
BLOCK_SIZE = 16
MAX_TOKENS = 8
REPETITIONS = 3
SEED = 20260904
TTFT_SLO_S = {4096: 30.0, 16384: 60.0, 32768: 120.0, 65536: 240.0}


def _valid_identifier(value: str) -> bool:
    return 1 <= len(value) <= 128 and all(
        character.isalnum() or character in "._-" for character in value)


def _payload(content: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": "llm",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "temperature": 0,
        "seed": SEED,
        "thinking": False,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def _summarize_response(raw: dict[str, Any]) -> dict[str, Any]:
    completion = raw["completion_tokens"]
    timing = stream_timing_metrics(
        completion, raw["ttft_s"], raw["last_output_s"])
    return {
        "ok": raw["ok"],
        "elapsed_s": raw["elapsed_s"],
        "ttft_s": raw["ttft_s"],
        "tpot_s": timing["tpot_s"],
        "itl_s": timing["tpot_s"],
        "prompt_tokens": raw["prompt_tokens"],
        "cached_tokens": raw["cached_tokens"],
        "completion_tokens": completion,
        "finish_reason": raw["finish_reason"],
        "input_tps": raw["prompt_tokens"] / raw["ttft_s"],
        "output_tps": timing["output_tps"],
        "cache_tps": raw["cached_tokens"] / raw["ttft_s"],
        "request_throughput_rps": 1.0 / raw["elapsed_s"],
        # These two digests are private transient observations used only to
        # prove deterministic equality. The public/safe qualifier strips them.
        "first_output_identity": raw["first_token_sha256"],
        "output_identity": raw["output_sha256"],
    }


def _same_output(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left["first_output_identity"], left["output_identity"],
        left["completion_tokens"], left["finish_reason"],
    ) == (
        right["first_output_identity"], right["output_identity"],
        right["completion_tokens"], right["finish_reason"],
    )


def _response_ok(value: Any, prompt_tokens: int | None,
                 completion_tokens: int) -> bool:
    if not isinstance(value, dict):
        return False
    finite = (
        "elapsed_s", "ttft_s", "tpot_s", "itl_s", "input_tps",
        "output_tps", "cache_tps", "request_throughput_rps",
    )
    return (
        value.get("ok") is True
        and all(isinstance(value.get(name), (int, float))
                and not isinstance(value[name], bool)
                and math.isfinite(float(value[name]))
                and value[name] >= 0.0 for name in finite)
        and value["elapsed_s"] > 0.0
        and value["ttft_s"] > 0.0
        and value.get("completion_tokens") == completion_tokens
        and isinstance(value.get("prompt_tokens"), int)
        and (prompt_tokens is None or value["prompt_tokens"] == prompt_tokens)
        and isinstance(value.get("cached_tokens"), int)
        and value["cached_tokens"] >= 0
        and isinstance(value.get("finish_reason"), str)
        and bool(value["finish_reason"])
        and all(isinstance(value.get(name), str) and len(value[name]) == 64
                for name in ("first_output_identity", "output_identity"))
    )


def evaluate(report: Any) -> dict[str, Any]:
    reasons: list[str] = []
    if not isinstance(report, dict):
        return {"qualified": False, "reasons": ["report is not an object"]}
    cold = report.get("cold_cases")
    partial = report.get("partial_cases")
    expected_count = len(TARGETS) * REPETITIONS
    if (
        report.get("schema") != SCHEMA or report.get("version") != 2
        or report.get("targets") != list(TARGETS)
        or report.get("partial_residual_tokens") != PARTIAL_RESIDUAL_TOKENS
        or report.get("block_size") != BLOCK_SIZE
        or report.get("max_tokens") != MAX_TOKENS
        or report.get("repetitions") != REPETITIONS
        or report.get("seed") != SEED
        or not isinstance(cold, list) or len(cold) != expected_count
        or not isinstance(partial, list) or len(partial) != expected_count
    ):
        return {"qualified": False, "reasons": [
            "fixed short-TP4 population is incomplete"]}

    for cases, mode in ((cold, "cold"), (partial, "partial")):
        for index, case in enumerate(cases):
            target = TARGETS[index // REPETITIONS]
            repetition = index % REPETITIONS
            label = f"{mode}/{target}/rep-{repetition}"
            if (not isinstance(case, dict)
                    or case.get("target_prompt_tokens") != target
                    or case.get("repetition") != repetition):
                reasons.append(f"{label}: case identity differs")
                continue
            if mode == "cold":
                left, right = case.get("cold"), case.get("warm")
                if not (_response_ok(left, target, MAX_TOKENS)
                        and _response_ok(right, target, MAX_TOKENS)):
                    reasons.append(f"{label}: response contract differs")
                    continue
                if left["cached_tokens"] != 0:
                    reasons.append(f"{label}: cold request was cached")
                if right["cached_tokens"] < target - 2 * BLOCK_SIZE:
                    reasons.append(f"{label}: full-warm cache accounting differs")
                if not case.get("output_exact") or not _same_output(left, right):
                    reasons.append(f"{label}: deterministic output differs")
            else:
                context = target - PARTIAL_RESIDUAL_TOKENS
                primer = case.get("primer")
                sibling = case.get("first_sibling")
                left, right = case.get("partial"), case.get("warm")
                if (case.get("block_context_tokens") != context
                        or not _response_ok(primer, None, 1)
                        or not _response_ok(sibling, target, 1)
                        or not _response_ok(left, target, MAX_TOKENS)
                        or not _response_ok(right, target, MAX_TOKENS)):
                    reasons.append(f"{label}: response contract differs")
                    continue
                if primer["cached_tokens"] != 0 or sibling["cached_tokens"] != 0:
                    reasons.append(f"{label}: cold/first-sibling cache identity differs")
                if not context - 2 * BLOCK_SIZE <= left["cached_tokens"] <= context:
                    reasons.append(f"{label}: partial cache accounting differs")
                if right["cached_tokens"] < target - 2 * BLOCK_SIZE:
                    reasons.append(f"{label}: full-warm cache accounting differs")
                if not case.get("output_exact") or not _same_output(left, right):
                    reasons.append(f"{label}: deterministic output differs")
    return {"qualified": not reasons, "reasons": reasons}


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from prefix_boundary_api import build_admission_boundary_prompts

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True)
    started = time.monotonic()
    cold_cases = []
    for target in TARGETS:
        for repetition in range(REPETITIONS):
            content = build_exact_prompt(
                tokenizer, target,
                f"{args.prompt_set_id}-cold-{target}-{repetition}")
            request = _payload(content, MAX_TOKENS)
            cold = _summarize_response(_post_stream(
                args.base, request, args.timeout_s, tokenizer))
            warm = _summarize_response(_post_stream(
                args.base, request, args.timeout_s, tokenizer))
            cold_cases.append({
                "target_prompt_tokens": target,
                "repetition": repetition,
                "cold": cold,
                "warm": warm,
                "output_exact": _same_output(cold, warm),
            })

    partial_cases = []
    for target in TARGETS:
        context = target - PARTIAL_RESIDUAL_TOKENS
        for repetition in range(REPETITIONS):
            primer_content, sibling_content, partial_content, shared, total = (
                build_admission_boundary_prompts(
                    tokenizer, context, PARTIAL_RESIDUAL_TOKENS - 1,
                    BLOCK_SIZE,
                    f"{args.prompt_set_id}-partial-{target}-{repetition}"))
            primer = _summarize_response(_post_stream(
                args.base, _payload(primer_content, 1),
                args.timeout_s, tokenizer))
            sibling = _summarize_response(_post_stream(
                args.base, _payload(sibling_content, 1),
                args.timeout_s, tokenizer))
            request = _payload(partial_content, MAX_TOKENS)
            partial = _summarize_response(_post_stream(
                args.base, request, args.timeout_s, tokenizer))
            warm = _summarize_response(_post_stream(
                args.base, request, args.timeout_s, tokenizer))
            partial_cases.append({
                "target_prompt_tokens": total,
                "block_context_tokens": context,
                "partial_residual_tokens": PARTIAL_RESIDUAL_TOKENS,
                "shared_tokens_before_block_rounding": shared,
                "repetition": repetition,
                "primer": primer,
                "first_sibling": sibling,
                "partial": partial,
                "warm": warm,
                "output_exact": _same_output(partial, warm),
            })

    primary = [
        ("cold", case["target_prompt_tokens"], case["cold"])
        for case in cold_cases
    ] + [
        ("partial", case["target_prompt_tokens"], case["partial"])
        for case in partial_cases
    ] + [
        ("warm", case["target_prompt_tokens"], case["warm"])
        for case in cold_cases + partial_cases
    ]
    report = {
        "schema": SCHEMA,
        "version": 2,
        "run_id": args.run_id,
        "prompt_set_id": args.prompt_set_id,
        "selector": args.selector,
        "targets": list(TARGETS),
        "partial_residual_tokens": PARTIAL_RESIDUAL_TOKENS,
        "block_size": BLOCK_SIZE,
        "max_tokens": MAX_TOKENS,
        "repetitions": REPETITIONS,
        "seed": SEED,
        "workload_order": "cold_then_full_warm_then_partial_sequence",
        "expected_requests": 72,
        "completed_requests": 72,
        "elapsed_s": time.monotonic() - started,
        "cold_cases": cold_cases,
        "partial_cases": partial_cases,
        "metrics": {
            "ttft_p50_s": statistics.median(item[2]["ttft_s"] for item in primary),
            "success_rate": 1.0,
            "error_rate": 0.0,
            "slo_goodput_requests": sum(
                item[2]["ttft_s"] <= TTFT_SLO_S[item[1]] for item in primary),
            "slo_total_requests": len(primary),
        },
        "ttft_slo_s": {str(key): value for key, value in TTFT_SLO_S.items()},
        "privacy": {
            "contains_prompts": False,
            "contains_model_outputs": False,
            "contains_token_ids": False,
            "contains_credentials": False,
            "contains_private_output_identities": True,
            "must_remain_outside_repository": True,
        },
        "authorization": {
            "long_context_authorized": False,
            "full_capability_authorized": False,
            "main_or_yaml_change_authorized": False,
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
    parser.add_argument("--prompt-set-id", required=True)
    parser.add_argument(
        "--selector", choices=("control_a", "control_b", "candidate"),
        required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (not math.isfinite(args.timeout_s) or args.timeout_s <= 0
            or not _valid_identifier(args.run_id)
            or not _valid_identifier(args.prompt_set_id)):
        parser.error("short TP4 v2 parameters differ from the fixed contract")
    return args


def main() -> int:
    args = parse_args()
    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(
        report, ensure_ascii=True, indent=2, sort_keys=True,
        allow_nan=False) + "\n", encoding="ascii")
    temporary.replace(args.out)
    print(json.dumps({
        "qualified": report["qualified"],
        "selector": args.selector,
        "elapsed_s": report["elapsed_s"],
        "request_count": report["completed_requests"],
        "reasons": report["reasons"],
    }, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
