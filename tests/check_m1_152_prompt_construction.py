#!/usr/bin/env python3
"""Validate M1-152 prompt lengths and partial-prefix boundaries on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import short_tp4_p90_funnel_service as contract


SCHEMA = "bi100-m1-152-tokenizer-construction-smoke-v1"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate(report: Any) -> dict[str, Any]:
    reasons: list[str] = []
    if not isinstance(report, dict):
        return {"qualified": False, "reasons": ["report must be an object"]}
    cold = report.get("cold")
    partial = report.get("partial")
    if (
        report.get("schema") != SCHEMA
        or report.get("version") != 1
        or not isinstance(cold, list)
        or len(cold) != len(contract.TARGETS)
        or not isinstance(partial, list)
        or len(partial) != len(contract.PARTIAL_TARGETS)
        or report.get("privacy") != {
            "prompts_recorded": False,
            "token_ids_recorded": False,
            "credentials_recorded": False,
        }
    ):
        reasons.append("prompt-construction report structure differs")
        return {"qualified": False, "reasons": reasons}
    for target, row in zip(contract.TARGETS, cold):
        if (
            not isinstance(row, dict)
            or row.get("target_prompt_tokens") != target
            or row.get("actual_prompt_tokens") != target
            or not _digest(row.get("prompt_sha256"))
        ):
            reasons.append(f"cold/{target}: exact prompt differs")
    for target, row in zip(contract.PARTIAL_TARGETS, partial):
        context = target - contract.PARTIAL_RESIDUAL_TOKENS
        if (
            not isinstance(row, dict)
            or row.get("target_prompt_tokens") != target
            or row.get("actual_prompt_tokens") != target
            or row.get("block_context_tokens") != context
            or row.get("cached_prefix_tokens") != context
            or row.get("residual_prefill_tokens")
            != contract.PARTIAL_RESIDUAL_TOKENS
            or not isinstance(row.get("shared_tokens_before_rounding"), int)
            or not (
                context
                <= row["shared_tokens_before_rounding"]
                < context + contract.BLOCK_SIZE
            )
            or not isinstance(row.get("primer_prompt_tokens"), int)
            or row["primer_prompt_tokens"] <= context
            or not _digest(row.get("primer_prompt_sha256"))
            or not _digest(row.get("partial_prompt_sha256"))
        ):
            reasons.append(f"partial/{target}: prefix boundary differs")
    return {"qualified": not reasons, "reasons": reasons}


def _digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def run(model_path: Path, prompt_set_id: str) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from long_context_api import build_exact_prompt, prompt_token_count
    from prefix_boundary_api import (
        build_boundary_prompts,
        common_prefix_len,
        encode_chat,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    cold = []
    for target in contract.TARGETS:
        content = build_exact_prompt(
            tokenizer,
            target,
            f"{prompt_set_id}-cold-{target}",
        )
        cold.append({
            "target_prompt_tokens": target,
            "actual_prompt_tokens": prompt_token_count(tokenizer, content),
            "prompt_sha256": _sha256_text(content),
        })

    partial = []
    for target in contract.PARTIAL_TARGETS:
        context = target - contract.PARTIAL_RESIDUAL_TOKENS
        primer_content, partial_content, shared_tokens, total_tokens = (
            build_boundary_prompts(
                tokenizer,
                context,
                contract.PARTIAL_RESIDUAL_TOKENS - 1,
                contract.BLOCK_SIZE,
                f"{prompt_set_id}-partial-{target}",
            )
        )
        primer_ids = encode_chat(tokenizer, primer_content)
        partial_ids = encode_chat(tokenizer, partial_content)
        cached = (
            common_prefix_len(primer_ids, partial_ids)
            // contract.BLOCK_SIZE
            * contract.BLOCK_SIZE
        )
        partial.append({
            "target_prompt_tokens": target,
            "actual_prompt_tokens": total_tokens,
            "block_context_tokens": context,
            "shared_tokens_before_rounding": shared_tokens,
            "cached_prefix_tokens": cached,
            "residual_prefill_tokens": len(partial_ids) - cached,
            "primer_prompt_tokens": len(primer_ids),
            "primer_prompt_sha256": _sha256_text(primer_content),
            "partial_prompt_sha256": _sha256_text(partial_content),
        })

    report = {
        "schema": SCHEMA,
        "version": 1,
        "model_path": str(model_path.resolve()),
        "prompt_set_id": prompt_set_id,
        "cold": cold,
        "partial": partial,
        "privacy": {
            "prompts_recorded": False,
            "token_ids_recorded": False,
            "credentials_recorded": False,
        },
    }
    report["evaluation"] = validate(report)
    report["qualified"] = report["evaluation"]["qualified"]
    report["reasons"] = report["evaluation"]["reasons"]
    return report


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            json.dump(
                value,
                stream,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="/root/public-storage/models/Qwen/Qwen3.6-35B-A3B",
        type=Path,
    )
    parser.add_argument(
        "--prompt-set-id",
        default="m1-152-tokenizer-smoke-v1",
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if (
        not args.model_path.is_dir()
        or not contract._valid_identifier(args.prompt_set_id)
    ):
        parser.error("prompt-construction smoke parameters are invalid")
    report = run(args.model_path, args.prompt_set_id)
    _atomic_json(args.out, report)
    print(json.dumps({
        "qualified": report["qualified"],
        "cold_count": len(report["cold"]),
        "partial_count": len(report["partial"]),
        "reasons": report["reasons"],
    }, ensure_ascii=True, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
