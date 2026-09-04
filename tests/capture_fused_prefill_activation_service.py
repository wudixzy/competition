#!/usr/bin/env python3
"""Issue one synthetic request per length for private activation capture."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from bench_fused_prefill_service import _post_stream
from long_context_api import build_exact_prompt


TARGET_BUCKETS = {
    32768: 24576,
    65536: 57344,
    131072: 122880,
}
MANIFEST_FIELDS = {
    "schema", "version", "run_id", "rank", "source_revision",
    "runtime_identity", "source_artifact_sha256", "model_identity",
    "tokenizer_identity", "instance", "captured_at_utc", "producer",
    "synthetic_prompt_attestation", "selection", "capture_topology",
    "record_count", "records", "privacy",
}
RECORD_FIELDS = {
    "bucket_min_context_tokens", "call_ordinal", "layer_index",
    "context_tokens", "query_length", "file", "sha256", "size_bytes",
    "compact_physical_blocks", "logical_blocks", "block_table",
    "head_mapping", "tensors",
}
CAPTURE_SELECTION = {
    "context_buckets": [24576, 57344, 122880],
    "full_attention_call_ordinals": [0],
}
CAPTURE_TOPOLOGY = {
    "tensor_parallel_size": 1,
    "query_heads": 16,
    "kv_heads": 2,
    "head_dim": 256,
    "gqa_ratio": 8,
    "block_size": 16,
    "query_head_order": list(range(16)),
    "kv_head_order": [0, 1],
}
CAPTURE_HEAD_MAPPING = {
    "query_head_indices": list(range(16)),
    "key_value_head_indices": [0, 1],
    "gqa_ratio": 8,
}
CAPTURE_PRIVACY = {
    "raw_activation_files_private": True,
    "raw_activation_files_may_be_committed": False,
    "contains_prompts": False,
    "contains_model_outputs": False,
    "contains_token_ids": False,
    "contains_credentials": False,
}


def _hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _hex_digest(value: object) -> bool:
    return _hex(value, 64)


def _captured_cells(
    path: Path,
    expected_run_id: str | None = None,
) -> set[tuple[int, int]]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("activation manifest is missing or corrupt") from exc
    if (
        not isinstance(value, dict)
        or set(value) != MANIFEST_FIELDS
        or value.get("schema") != "bi100-fused-prefill-activation-bank-v2"
        or value.get("version") != 2
        or value.get("rank") != 0
        or not isinstance(value.get("run_id"), str)
        or not value["run_id"]
        or (expected_run_id is not None
            and value.get("run_id") != expected_run_id)
        or not isinstance(value.get("records"), list)
        or value.get("record_count") != len(value["records"])
        or not _hex_digest(value.get("source_artifact_sha256"))
        or not _hex(value.get("source_revision"), 40)
        or not value.get("runtime_identity")
        or not isinstance(value.get("instance"), str)
        or not value["instance"]
        or not isinstance(value.get("captured_at_utc"), str)
        or not value["captured_at_utc"]
        or value.get("producer") != "baseline-pytorch-fallback"
        or value.get("synthetic_prompt_attestation")
        != "synthetic-exact-prompt-v1"
        or value.get("selection") != CAPTURE_SELECTION
        or value.get("capture_topology") != CAPTURE_TOPOLOGY
        or value.get("privacy") != CAPTURE_PRIVACY
        or not isinstance(value.get("model_identity"), dict)
        or set(value["model_identity"]) != {"name", "config_sha256"}
        or value["model_identity"].get("name") != "Qwen3.6-35B-A3B"
        or not _hex_digest(value["model_identity"].get("config_sha256"))
        or not isinstance(value.get("tokenizer_identity"), dict)
        or set(value["tokenizer_identity"]) != {"sha256"}
        or not _hex_digest(value["tokenizer_identity"].get("sha256"))
    ):
        raise RuntimeError("activation manifest contract differs")
    records = value["records"]
    cells = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != RECORD_FIELDS:
            raise RuntimeError("activation manifest record contract differs")
        bucket = record.get("bucket_min_context_tokens")
        ordinal = record.get("call_ordinal")
        context_tokens = record.get("context_tokens")
        query_length = record.get("query_length")
        logical_blocks = record.get("logical_blocks")
        compact_blocks = record.get("compact_physical_blocks")
        block_metadata = record.get("block_table")
        tensors = record.get("tensors")
        tensor_contract = {
            "query": {"shape": [query_length, 16, 256], "dtype": "torch.float16"},
            "key": {"shape": [query_length, 2, 256], "dtype": "torch.float16"},
            "value": {"shape": [query_length, 2, 256], "dtype": "torch.float16"},
            "key_cache": {
                "shape": [compact_blocks, 2, 32, 16, 8],
                "dtype": "torch.float16",
            },
            "value_cache": {
                "shape": [compact_blocks, 2, 256, 16],
                "dtype": "torch.float16",
            },
            "block_table": {
                "shape": [logical_blocks], "dtype": "torch.int32"},
        }
        if (
            not isinstance(bucket, int) or isinstance(bucket, bool)
            or not isinstance(ordinal, int) or isinstance(ordinal, bool)
            or not _hex_digest(record.get("sha256"))
            or bucket not in CAPTURE_SELECTION["context_buckets"]
            or ordinal not in CAPTURE_SELECTION[
                "full_attention_call_ordinals"]
            or not isinstance(record.get("layer_index"), int)
            or isinstance(record.get("layer_index"), bool)
            or context_tokens != bucket
            or not isinstance(query_length, int)
            or isinstance(query_length, bool)
            or not 16 < query_length <= 8192
            or context_tokens + query_length > 262144
            or logical_blocks != context_tokens // 16
            or not isinstance(compact_blocks, int)
            or isinstance(compact_blocks, bool)
            or not 0 < compact_blocks <= logical_blocks
            or not isinstance(record.get("file"), str)
            or Path(record["file"]).name != record["file"]
            or not isinstance(record.get("size_bytes"), int)
            or record["size_bytes"] <= 0
            or not isinstance(block_metadata, dict)
            or block_metadata.get("shape") != [logical_blocks]
            or not _hex_digest(block_metadata.get("sha256"))
            or block_metadata.get("logical_order")
            != "preserved_after_first_occurrence_compaction"
            or record.get("head_mapping") != CAPTURE_HEAD_MAPPING
            or tensors != tensor_contract
        ):
            raise RuntimeError("activation manifest record identity differs")
        cell = (bucket, ordinal)
        if cell in cells:
            raise RuntimeError("activation manifest contains duplicate cells")
        cells.add(cell)
    return cells


def main() -> int:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--model-path",
        default="/root/public-storage/models/Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--targets", default="32768,65536,131072")
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    targets = [int(value) for value in args.targets.split(",")]
    if (
        not targets
        or targets != sorted(set(targets))
        or any(value <= 32 for value in targets)
        or any(value + args.max_tokens > 262144 for value in targets)
    ):
        parser.error("targets must be unique increasing valid prompt lengths")
    if args.max_tokens < 1:
        parser.error("max-tokens must be positive")
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("timeout-s must be finite and positive")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True)
    started = time.monotonic()
    requests = []
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
        result = _post_stream(
            args.base, payload, args.timeout_s, tokenizer)
        if (
            result["prompt_tokens"] != target
            or result["completion_tokens"] < args.max_tokens
            or result["cached_tokens"] != 0
        ):
            raise RuntimeError(
                f"capture request contract failed for target {target}")
        expected_bucket = TARGET_BUCKETS.get(target)
        if expected_bucket is None:
            raise RuntimeError(
                f"capture target {target} has no frozen bucket mapping")
        captured_cells = _captured_cells(args.manifest, args.run_id)
        if (expected_bucket, 0) not in captured_cells:
            raise RuntimeError(
                f"activation bank missing target bucket {expected_bucket}")
        requests.append({
            "target_prompt_tokens": target,
            "elapsed_s": result["elapsed_s"],
            "ttft_s": result["ttft_s"],
            "completion_tokens": result["completion_tokens"],
            "cached_tokens": result["cached_tokens"],
            "finish_reason": result["finish_reason"],
            "first_token_sha256": result["first_token_sha256"],
            "output_sha256": result["output_sha256"],
            "captured_bucket": expected_bucket,
        })
    report = {
        "schema": "bi100-fused-prefill-activation-capture-requests-v1",
        "version": 1,
        "run_id": args.run_id,
        "targets": targets,
        "max_tokens": args.max_tokens,
        "elapsed_s": time.monotonic() - started,
        "requests": requests,
        "captured_cells": [
            list(cell) for cell in sorted(
                _captured_cells(args.manifest, args.run_id))
        ],
        "privacy": {
            "contains_prompts": False,
            "contains_model_outputs": False,
            "contains_token_ids": False,
            "contains_credentials": False,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="ascii",
    )
    print(json.dumps({
        "targets": targets,
        "elapsed_s": report["elapsed_s"],
        "qualified": True,
    }, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
