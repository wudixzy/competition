#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SUMMARY_FIELDS = (
    "prompt_tokens",
    "cached_tokens",
    "completion_tokens",
    "finish_reason",
    "message_sha256",
    "elapsed_s",
)


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_digest(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _request(
    value: Any,
    label: str,
    target_prompt_tokens: int,
    minimum_cached_tokens: int,
    minimum_completion_tokens: int,
    reasons: list[str],
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        reasons.append(f"{label} must be an object")
        return None
    safe = {field: value.get(field) for field in SUMMARY_FIELDS}
    if safe["prompt_tokens"] != target_prompt_tokens:
        reasons.append(
            f"{label} prompt_tokens must equal {target_prompt_tokens}")
    cached_tokens = safe["cached_tokens"]
    if (not _is_integer(cached_tokens)
            or cached_tokens < minimum_cached_tokens):
        reasons.append(
            f"{label} cached_tokens must be at least "
            f"{minimum_cached_tokens}")
    completion_tokens = safe["completion_tokens"]
    if (not _is_integer(completion_tokens)
            or completion_tokens < minimum_completion_tokens):
        reasons.append(
            f"{label} completion_tokens must be at least "
            f"{minimum_completion_tokens}")
    if not isinstance(safe["finish_reason"], str):
        reasons.append(f"{label} finish_reason must be a string")
    if not _is_digest(safe["message_sha256"]):
        reasons.append(f"{label} message_sha256 is invalid")
    elapsed_s = safe["elapsed_s"]
    if (not isinstance(elapsed_s, (int, float))
            or isinstance(elapsed_s, bool)
            or not math.isfinite(elapsed_s) or elapsed_s <= 0):
        reasons.append(f"{label} elapsed_s must be finite and positive")
    return safe


def qualify(
    source: Any,
    *,
    target_prompt_tokens: int,
    max_tokens: int,
    minimum_cached_tokens: int,
    minimum_completion_tokens: int,
    equivalence_mode: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not isinstance(source, dict):
        return {"qualified": False, "reasons": ["source must be an object"]}
    expected_contract = {
        "target_prompt_tokens": target_prompt_tokens,
        "max_tokens": max_tokens,
        "min_cached_tokens": minimum_cached_tokens,
        "min_completion_tokens": minimum_completion_tokens,
        "equivalence_mode": equivalence_mode,
    }
    for field, expected in expected_contract.items():
        if source.get(field) != expected:
            reasons.append(f"{field} must equal {expected!r}")

    first = _request(
        source.get("first"), "first", target_prompt_tokens, 0,
        minimum_completion_tokens, reasons)
    second = _request(
        source.get("second"), "second", target_prompt_tokens,
        minimum_cached_tokens, minimum_completion_tokens, reasons)
    third = None
    if equivalence_mode == "warm-repeat":
        third = _request(
            source.get("third"), "third", target_prompt_tokens,
            minimum_cached_tokens, minimum_completion_tokens, reasons)
    elif equivalence_mode != "exact":
        reasons.append("equivalence_mode must be exact or warm-repeat")

    if first is not None and first.get("cached_tokens") != 0:
        reasons.append("first cached_tokens must equal zero")
    left, right = ((first, second) if equivalence_mode == "exact"
                   else (second, third))
    if left is not None and right is not None:
        for field in (
                "completion_tokens", "finish_reason", "message_sha256"):
            if left.get(field) != right.get(field):
                reasons.append(
                    f"equivalent requests differ in {field}")

    safe_requests = {"first": first, "second": second}
    if equivalence_mode == "warm-repeat":
        safe_requests["third"] = third
    return {
        "contract": expected_contract,
        "qualified": not reasons,
        "reasons": reasons,
        "requests": safe_requests,
        "schema": "bi100-long-context-safe-gate-v1",
        "version": 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-prompt-tokens", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--min-cached-tokens", type=int, required=True)
    parser.add_argument("--min-completion-tokens", type=int, required=True)
    parser.add_argument(
        "--equivalence-mode", choices=("exact", "warm-repeat"), required=True)
    args = parser.parse_args()
    source_bytes = args.input.read_bytes()
    source = json.loads(source_bytes)
    report = qualify(
        source,
        target_prompt_tokens=args.target_prompt_tokens,
        max_tokens=args.max_tokens,
        minimum_cached_tokens=args.min_cached_tokens,
        minimum_completion_tokens=args.min_completion_tokens,
        equivalence_mode=args.equivalence_mode,
    )
    report["source_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.out)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
