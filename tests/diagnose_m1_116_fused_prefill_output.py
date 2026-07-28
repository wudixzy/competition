#!/usr/bin/env python3
"""Collect privacy-safe 65K output-divergence evidence for fused prefill."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import bench_fused_prefill_service as bench
from long_context_api import build_exact_prompt
import quality_runtime_contract as runtime_contract


SCHEMA = "bi100-m1-116-fused-prefill-output-diagnostic-v1"
VERSION = 1
TARGET_PROMPT_TOKENS = 65536
MAX_TOKENS_LADDER = (1, 2, 4, 8, 16, 32)
REPRODUCTION_MAX_TOKENS = 32
SEED = 20260721
MINIMUM_WARM_CACHED_TOKENS = TARGET_PROMPT_TOKENS - 32
Json = dict[str, Any]
REQUEST_FIELDS = {
    "status",
    "elapsed_s",
    "ttft_s",
    "decode_window_s",
    "output_tps",
    "prompt_tokens",
    "cached_tokens",
    "completion_tokens",
    "finish_reason",
    "first_token_hmac_sha256",
    "output_hmac_sha256",
    "request_contract_sha256",
}


def _atomic_write(path: Path, value: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=True, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _identity_hmac(identity_key: bytes, value: str) -> str:
    return hmac.new(
        identity_key, value.encode("ascii"), hashlib.sha256).hexdigest()


def _request_summary(
    value: Json,
    request_contract_sha256: str,
    identity_key: bytes,
) -> Json:
    return {
        "status": 200,
        "elapsed_s": value["elapsed_s"],
        "ttft_s": value["ttft_s"],
        "decode_window_s": value["decode_window_s"],
        "output_tps": value["output_tps"],
        "prompt_tokens": value["prompt_tokens"],
        "cached_tokens": value["cached_tokens"],
        "completion_tokens": value["completion_tokens"],
        "finish_reason": value["finish_reason"],
        "first_token_hmac_sha256": _identity_hmac(
            identity_key, value["first_token_sha256"]),
        "output_hmac_sha256": _identity_hmac(
            identity_key, value["output_sha256"]),
        "request_contract_sha256": request_contract_sha256,
    }


def _same_output(left: Json, right: Json) -> bool:
    return all(
        left[field] == right[field]
        for field in (
            "first_token_hmac_sha256",
            "output_hmac_sha256",
            "completion_tokens",
            "finish_reason",
        )
    )


def _validate_request(
    request: Json,
    *,
    max_tokens: int,
    expected_cached: bool,
    label: str,
) -> list[str]:
    reasons = []
    if not isinstance(request, dict) or set(request) != REQUEST_FIELDS:
        return [f"{label} fields are invalid"]
    if request.get("status") != 200:
        reasons.append(f"{label} status differs")
    if request.get("prompt_tokens") != TARGET_PROMPT_TOKENS:
        reasons.append(f"{label} prompt_tokens differs")
    if request.get("completion_tokens") != max_tokens:
        reasons.append(f"{label} completion_tokens differs")
    if request.get("finish_reason") != "length":
        reasons.append(f"{label} finish_reason differs")
    for field in ("first_token_hmac_sha256", "output_hmac_sha256",
                  "request_contract_sha256"):
        if not runtime_contract.is_sha256(request.get(field)):
            reasons.append(f"{label} {field} is invalid")
    for field in ("elapsed_s", "ttft_s"):
        value = request.get(field)
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value <= 0):
            reasons.append(f"{label} {field} is invalid")
    for field in ("decode_window_s", "output_tps"):
        value = request.get(field)
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value < 0):
            reasons.append(f"{label} {field} is invalid")
    cached_tokens = request.get("cached_tokens")
    if not isinstance(cached_tokens, int) or isinstance(cached_tokens, bool):
        reasons.append(f"{label} cached_tokens is invalid")
    elif expected_cached:
        if cached_tokens < MINIMUM_WARM_CACHED_TOKENS:
            reasons.append(f"{label} effective cache restore is too short")
    elif cached_tokens != 0:
        reasons.append(f"{label} cold request was not cold")
    return reasons


def _validate_observations(
    cold: Json,
    cold_repeat: Json,
    ladder: list[Json],
) -> list[str]:
    reasons = []
    reasons.extend(_validate_request(
        cold,
        max_tokens=REPRODUCTION_MAX_TOKENS,
        expected_cached=False,
        label="reproduction cold",
    ))
    reasons.extend(_validate_request(
        cold_repeat,
        max_tokens=REPRODUCTION_MAX_TOKENS,
        expected_cached=True,
        label="reproduction warm",
    ))
    if not _same_output(cold, cold_repeat):
        reasons.append("reproduction cold/warm output differs")
    if len(ladder) != len(MAX_TOKENS_LADDER):
        return reasons + ["ladder row count differs"]
    for expected_budget, row in zip(MAX_TOKENS_LADDER, ladder):
        label = f"ladder max_tokens={expected_budget}"
        if not isinstance(row, dict) or set(row) != {
                "max_tokens", "warm_1", "warm_2"}:
            reasons.append(f"{label} fields are invalid")
            continue
        if row["max_tokens"] != expected_budget:
            reasons.append(f"{label} budget differs")
        for repeat in ("warm_1", "warm_2"):
            reasons.extend(_validate_request(
                row[repeat],
                max_tokens=expected_budget,
                expected_cached=True,
                label=f"{label} {repeat}",
            ))
        if not _same_output(row["warm_1"], row["warm_2"]):
            reasons.append(f"{label} repeated warm output differs")
    return reasons


def _payload(content: str, max_tokens: int) -> Json:
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


def _post(
    base: str,
    content: str,
    max_tokens: int,
    timeout_s: float,
    tokenizer: Any,
    identity_key: bytes,
) -> Json:
    payload = _payload(content, max_tokens)
    result = bench._post_stream(base, payload, timeout_s, tokenizer)
    return _request_summary(
        result, runtime_contract.sha256_json(payload), identity_key)


def main() -> int:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--runtime-identity", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--mode", choices=("control", "candidate"),
                        required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not runtime_contract.is_git_revision(args.source_revision):
        parser.error("--source-revision must be a fixed Git object id")
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("--timeout-s must be finite and positive")
    identity_key_hex = os.environ.pop(
        "BI100_FUSED_OUTPUT_DIAGNOSTIC_HMAC_KEY", "")
    if (len(identity_key_hex) != 64
            or any(character not in "0123456789abcdef"
                   for character in identity_key_hex)):
        parser.error(
            "BI100_FUSED_OUTPUT_DIAGNOSTIC_HMAC_KEY must be 32-byte hex")
    identity_key = bytes.fromhex(identity_key_hex)

    expected_contract = {
        "source_revision": args.source_revision,
        "runtime_identity": args.runtime_identity,
        "instance": args.instance,
        "gpu_count": 4,
        "tensor_parallel_size": 4,
        "max_model_len": 262144,
        "model_path": args.model_path,
        "tokenizer_path": args.model_path,
        "served_model_name": "llm",
    }
    contract, contract_sha256 = runtime_contract.load_runtime_contract(
        args.runtime_contract,
        expected_contract,
        require_cache_trace=True,
    )
    expected_selector = "0" if args.mode == "control" else "1"
    if (contract["environment"].get("BI100_ATTN_COREX_FUSED_PREFILL")
            != expected_selector):
        parser.error("runtime contract fused-prefill selector differs")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True)
    content = build_exact_prompt(
        tokenizer, TARGET_PROMPT_TOKENS, args.run_id)

    cold = _post(
        args.base, content, REPRODUCTION_MAX_TOKENS,
        args.timeout_s, tokenizer, identity_key)
    cold_repeat = _post(
        args.base, content, REPRODUCTION_MAX_TOKENS,
        args.timeout_s, tokenizer, identity_key)
    ladder = []
    for max_tokens in MAX_TOKENS_LADDER:
        ladder.append({
            "max_tokens": max_tokens,
            "warm_1": _post(
                args.base, content, max_tokens, args.timeout_s, tokenizer,
                identity_key),
            "warm_2": _post(
                args.base, content, max_tokens, args.timeout_s, tokenizer,
                identity_key),
        })

    reasons = _validate_observations(cold, cold_repeat, ladder)
    report = {
        "schema": SCHEMA,
        "version": VERSION,
        "mode": args.mode,
        "qualified_diagnostic": not reasons,
        "strict_quality_non_regression_authorized": False,
        "production_promotion_authorized": False,
        "reasons": reasons,
        "source_revision": args.source_revision,
        "runtime_identity": args.runtime_identity,
        "instance": args.instance,
        "model_path": args.model_path,
        "target_prompt_tokens": TARGET_PROMPT_TOKENS,
        "reproduction_max_tokens": REPRODUCTION_MAX_TOKENS,
        "max_tokens_ladder": list(MAX_TOKENS_LADDER),
        "seed": SEED,
        "run_id_sha256": hashlib.sha256(
            args.run_id.encode("utf-8")).hexdigest(),
        "runtime_contract": {
            "sha256": contract_sha256,
            "contract": contract,
        },
        "reproduction": {
            "cold": cold,
            "warm": cold_repeat,
            "cold_warm_exact": _same_output(cold, cold_repeat),
        },
        "ladder": ladder,
        "privacy": {
            "contains_raw_requests": False,
            "contains_raw_model_outputs": False,
            "contains_token_ids": False,
            "contains_credentials": False,
        },
    }
    _atomic_write(args.out, report)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if not reasons else 1


if __name__ == "__main__":
    raise SystemExit(main())
