#!/usr/bin/env python3
"""Measure cold and partial-prefix TTFT shapes that drive platform P90."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

from bench_fused_prefill_service import _percentile, _post_stream
from long_context_api import build_exact_prompt


SCHEMA = "bi100-short-tp4-p90-funnel-service-v2"
TARGETS = (8192, 16384, 24576, 32768, 49152, 65536)
PARTIAL_TARGETS = (16384, 32768, 49152, 65536)
PARTIAL_RESIDUAL_TOKENS = 8192
BLOCK_SIZE = 16
MAX_TOKENS = 8
REPETITIONS = 1
SEED = 20260730


def _valid_identifier(value: str) -> bool:
    return (
        1 <= len(value) <= 128
        and all(
            character in (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            )
            for character in value
        )
    )


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


def _output_identity(response: dict[str, Any]) -> tuple[Any, ...]:
    return (
        response["first_token_sha256"],
        response["output_sha256"],
        response["completion_tokens"],
        response["finish_reason"],
    )


def _response_ok(
    response: Any,
    *,
    prompt_tokens: int | None,
    completion_tokens: int,
) -> bool:
    if not isinstance(response, dict) or response.get("ok") is not True:
        return False
    finite_fields = (
        "elapsed_s", "ttft_s", "last_output_s",
        "decode_window_s", "output_tps",
    )
    if any(
        not isinstance(response.get(name), (int, float))
        or isinstance(response.get(name), bool)
        or not math.isfinite(float(response[name]))
        or float(response[name]) < 0.0
        for name in finite_fields
    ):
        return False
    if (
        response["elapsed_s"] <= 0.0
        or response["ttft_s"] <= 0.0
        or response["last_output_s"] < response["ttft_s"]
        or response["elapsed_s"] < response["last_output_s"]
        or response.get("completion_tokens") != completion_tokens
        or not isinstance(response.get("prompt_tokens"), int)
        or isinstance(response.get("prompt_tokens"), bool)
        or response["prompt_tokens"] <= 0
        or not isinstance(response.get("cached_tokens"), int)
        or isinstance(response.get("cached_tokens"), bool)
        or response["cached_tokens"] < 0
        or not isinstance(response.get("finish_reason"), str)
        or not response["finish_reason"]
    ):
        return False
    if (
        prompt_tokens is not None
        and response["prompt_tokens"] != prompt_tokens
    ):
        return False
    for name in ("first_token_sha256", "output_sha256"):
        value = response.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            return False
    return True


def evaluate(report: Any) -> dict[str, Any]:
    reasons: list[str] = []
    if not isinstance(report, dict):
        return {"qualified": False, "reasons": ["report must be an object"]}
    cold_cases = report.get("cold_cases")
    partial_cases = report.get("partial_cases")
    if (
        report.get("schema") != SCHEMA
        or report.get("version") != 2
        or report.get("targets") != list(TARGETS)
        or report.get("partial_targets") != list(PARTIAL_TARGETS)
        or report.get("partial_residual_tokens")
        != PARTIAL_RESIDUAL_TOKENS
        or report.get("block_size") != BLOCK_SIZE
        or report.get("max_tokens") != MAX_TOKENS
        or report.get("repetitions") != REPETITIONS
        or report.get("seed") != SEED
        or not isinstance(cold_cases, list)
        or len(cold_cases) != len(TARGETS)
        or not isinstance(partial_cases, list)
        or len(partial_cases) != len(PARTIAL_TARGETS)
    ):
        reasons.append("report structure differs from the P90 contract")
        return {"qualified": False, "reasons": reasons}

    for expected_target, case in zip(TARGETS, cold_cases):
        label = f"cold/{expected_target}"
        if (
            not isinstance(case, dict)
            or case.get("target_prompt_tokens") != expected_target
            or case.get("repetition") != 0
            or not _response_ok(
                case.get("cold"),
                prompt_tokens=expected_target,
                completion_tokens=MAX_TOKENS,
            )
            or not _response_ok(
                case.get("warm"),
                prompt_tokens=expected_target,
                completion_tokens=MAX_TOKENS,
            )
        ):
            reasons.append(f"{label}: response contract differs")
            continue
        cold = case["cold"]
        warm = case["warm"]
        if cold["cached_tokens"] != 0:
            reasons.append(f"{label}: cold request was cached")
        if warm["cached_tokens"] < expected_target - 2 * BLOCK_SIZE:
            reasons.append(f"{label}: warm prefix was not retained")
        if _output_identity(cold) != _output_identity(warm):
            reasons.append(f"{label}: cold/warm output differs")

    for expected_target, case in zip(PARTIAL_TARGETS, partial_cases):
        label = f"partial/{expected_target}"
        context_tokens = expected_target - PARTIAL_RESIDUAL_TOKENS
        if (
            not isinstance(case, dict)
            or case.get("target_prompt_tokens") != expected_target
            or case.get("block_context_tokens") != context_tokens
            or case.get("partial_residual_tokens")
            != PARTIAL_RESIDUAL_TOKENS
            or case.get("repetition") != 0
            or not _response_ok(
                case.get("primer"),
                prompt_tokens=None,
                completion_tokens=1,
            )
            or not _response_ok(
                case.get("partial"),
                prompt_tokens=expected_target,
                completion_tokens=MAX_TOKENS,
            )
            or not _response_ok(
                case.get("warm"),
                prompt_tokens=expected_target,
                completion_tokens=MAX_TOKENS,
            )
        ):
            reasons.append(f"{label}: response contract differs")
            continue
        primer = case["primer"]
        partial = case["partial"]
        warm = case["warm"]
        if primer["cached_tokens"] != 0:
            reasons.append(f"{label}: primer was unexpectedly cached")
        if not (
            context_tokens - 2 * BLOCK_SIZE
            <= partial["cached_tokens"]
            <= context_tokens
        ):
            reasons.append(f"{label}: partial cached-token boundary differs")
        if warm["cached_tokens"] < expected_target - 2 * BLOCK_SIZE:
            reasons.append(f"{label}: full warm prefix was not retained")
        if _output_identity(partial) != _output_identity(warm):
            reasons.append(f"{label}: partial/warm output differs")

    return {"qualified": not reasons, "reasons": reasons}


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from prefix_boundary_api import build_boundary_prompts

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    started = time.monotonic()
    cold_cases = []
    for target in TARGETS:
        content = build_exact_prompt(
            tokenizer,
            target,
            f"{args.prompt_set_id}-cold-{target}",
        )
        prompt_sha256 = hashlib.sha256(
            content.encode("utf-8")).hexdigest()
        request = _payload(content, MAX_TOKENS)
        cold = _post_stream(args.base, request, args.timeout_s, tokenizer)
        warm = _post_stream(args.base, request, args.timeout_s, tokenizer)
        cold_cases.append({
            "target_prompt_tokens": target,
            "repetition": 0,
            "prompt_sha256": prompt_sha256,
            "cold": cold,
            "warm": warm,
        })

    partial_cases = []
    for target in PARTIAL_TARGETS:
        context_tokens = target - PARTIAL_RESIDUAL_TOKENS
        primer_content, partial_content, shared_tokens, total_tokens = (
            build_boundary_prompts(
                tokenizer,
                context_tokens,
                PARTIAL_RESIDUAL_TOKENS - 1,
                BLOCK_SIZE,
                f"{args.prompt_set_id}-partial-{target}",
            )
        )
        primer = _post_stream(
            args.base,
            _payload(primer_content, 1),
            args.timeout_s,
            tokenizer,
        )
        request = _payload(partial_content, MAX_TOKENS)
        partial = _post_stream(
            args.base, request, args.timeout_s, tokenizer)
        warm = _post_stream(
            args.base, request, args.timeout_s, tokenizer)
        partial_cases.append({
            "target_prompt_tokens": total_tokens,
            "block_context_tokens": context_tokens,
            "partial_residual_tokens": PARTIAL_RESIDUAL_TOKENS,
            "shared_tokens_before_block_rounding": shared_tokens,
            "repetition": 0,
            "primer_prompt_sha256": hashlib.sha256(
                primer_content.encode("utf-8")).hexdigest(),
            "partial_prompt_sha256": hashlib.sha256(
                partial_content.encode("utf-8")).hexdigest(),
            "primer": primer,
            "partial": partial,
            "warm": warm,
        })

    cold_ttfts = [case["cold"]["ttft_s"] for case in cold_cases]
    partial_ttfts = [
        case["partial"]["ttft_s"] for case in partial_cases]
    warm_ttfts = (
        [case["warm"]["ttft_s"] for case in cold_cases]
        + [case["warm"]["ttft_s"] for case in partial_cases]
    )
    report = {
        "schema": SCHEMA,
        "version": 2,
        "run_id": args.run_id,
        "prompt_set_id": args.prompt_set_id,
        "selector": args.selector,
        "targets": list(TARGETS),
        "partial_targets": list(PARTIAL_TARGETS),
        "partial_residual_tokens": PARTIAL_RESIDUAL_TOKENS,
        "block_size": BLOCK_SIZE,
        "max_tokens": MAX_TOKENS,
        "repetitions": REPETITIONS,
        "seed": SEED,
        "elapsed_s": time.monotonic() - started,
        "cold_cases": cold_cases,
        "partial_cases": partial_cases,
        "cold_ttft_median_s": statistics.median(cold_ttfts),
        "partial_ttft_median_s": statistics.median(partial_ttfts),
        "uncached_ttft_p90_s": _percentile(
            cold_ttfts + partial_ttfts, 90.0),
        "warm_ttft_median_s": statistics.median(warm_ttfts),
        "privacy": {
            "contains_prompts": False,
            "contains_model_outputs": False,
            "contains_token_ids": False,
            "contains_credentials": False,
        },
        "authorization": {
            "long_context_confirmation_authorized": False,
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
    parser.add_argument(
        "--model-path",
        default="/root/public-storage/models/Qwen/Qwen3.6-35B-A3B",
        type=Path,
    )
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prompt-set-id", required=True)
    parser.add_argument(
        "--selector",
        choices=("control", "candidate"),
        required=True,
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if (
        not math.isfinite(args.timeout_s)
        or args.timeout_s <= 0.0
        or not _valid_identifier(args.run_id)
        or not _valid_identifier(args.prompt_set_id)
    ):
        parser.error("P90 service parameters differ from the contract")
    return args


def main() -> int:
    args = parse_args()
    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(
        f".{args.out.name}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(
            report,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n",
        encoding="ascii",
    )
    temporary.replace(args.out)
    print(json.dumps({
        "qualified": report["qualified"],
        "selector": args.selector,
        "elapsed_s": report["elapsed_s"],
        "cold_ttft_median_s": report["cold_ttft_median_s"],
        "partial_ttft_median_s": report["partial_ttft_median_s"],
        "uncached_ttft_p90_s": report["uncached_ttft_p90_s"],
        "reasons": report["reasons"],
    }, ensure_ascii=True, sort_keys=True, allow_nan=False))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
