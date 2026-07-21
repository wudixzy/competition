#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from bench_fused_paged_prefill_attention import _make_case


MAX_ABS_LIMIT = 1e-3
RELATIVE_L2_LIMIT = 1e-5


class _TrackedExtension:

    def __init__(self, extension: Any):
        self.extension = extension
        self.calls: list[dict[str, Any]] = []

    def forward(self, *args: Any) -> Any:
        self.calls.append({
            "query_shape": list(args[0].shape),
            "active_blocks": int(args[5].numel()),
            "context_len": int(args[6]),
            "scale": float(args[7]),
        })
        return self.extension.forward(*args)


def _relative_l2(actual: Any, expected: Any) -> float:
    difference = (actual.float() - expected.float()).norm().item()
    denominator = expected.float().norm().item()
    if denominator == 0:
        return 0.0 if difference == 0 else math.inf
    return float(difference / denominator)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--context-len", type=int, default=4096)
    parser.add_argument("--query-len", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.context_len <= 0 or args.context_len % 16:
        parser.error("--context-len must be a positive multiple of 16")
    if not 16 < args.query_len <= 8192:
        parser.error("--query-len must be within [17, 8192]")
    if args.context_len + args.query_len > 262144:
        parser.error("context plus query exceeds 262144")

    import torch
    from vllm.attention.ops import paged_attn

    if not torch.cuda.is_available():
        raise RuntimeError("CoreX CUDA device is unavailable")
    extension = paged_attn._corex_fused_paged_prefill
    if extension is None:
        raise RuntimeError("installed corex_fused_paged_prefill is unavailable")
    extension_path = Path(extension.__file__).resolve(strict=True)
    extension_sha256 = hashlib.sha256(extension_path.read_bytes()).hexdigest()

    tensors = _make_case(
        torch, torch.device(args.device), args.context_len, args.query_len,
        seed=47)
    query, key, value, key_cache, value_cache, block_table = tensors
    padding = block_table[:7]
    padded_block_tables = torch.cat((block_table, padding)).unsqueeze(0)
    prefix_key = key[:0]
    prefix_value = value[:0]
    common = {
        "query": query,
        "key": key,
        "value": value,
        "prefix_key": prefix_key,
        "prefix_value": prefix_value,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_tables": padded_block_tables,
        "seq_index": 0,
        "block_context_len": args.context_len,
        "num_q_heads": 4,
        "num_kv_heads": 1,
        "head_dim": 256,
        "gqa_ratio": 4,
        "block_size": 16,
        "tile_sz": 512,
        "scale": 0.0625,
        "orig_dtype": torch.float16,
        "fused_request_eligible": True,
    }

    paged_attn._USE_COREX_FUSED_PAGED_PREFILL = False
    reference = paged_attn.PagedAttention._forward_prefix_segment_pytorch(
        **common)
    torch.cuda.synchronize()

    tracked = _TrackedExtension(extension)
    paged_attn._corex_fused_paged_prefill = tracked
    paged_attn._USE_COREX_FUSED_PAGED_PREFILL = True
    candidate = paged_attn.PagedAttention._forward_prefix_segment_pytorch(
        **common)
    torch.cuda.synchronize()

    difference = (candidate.float() - reference.float()).abs()
    maximum_absolute_error = float(difference.max().item())
    relative_l2 = _relative_l2(candidate, reference)
    finite = bool(torch.isfinite(candidate).all().item())
    required_blocks = args.context_len // 16
    qualified = bool(
        len(tracked.calls) == 1
        and tracked.calls[0]["active_blocks"] == required_blocks
        and finite
        and maximum_absolute_error <= MAX_ABS_LIMIT
        and relative_l2 <= RELATIVE_L2_LIMIT
    )
    result = {
        "schema": "bi100-m1-47-dispatch-parity-v1",
        "extension": {
            "path": str(extension_path),
            "sha256": extension_sha256,
            "size_bytes": extension_path.stat().st_size,
        },
        "case": {
            "context_len": args.context_len,
            "query_len": args.query_len,
            "block_table_shape": list(padded_block_tables.shape),
            "non_identity_physical_blocks": bool(
                not torch.equal(
                    block_table,
                    torch.arange(
                        required_blocks, dtype=torch.int32,
                        device=block_table.device))),
            "finite": finite,
            "maximum_absolute_error": maximum_absolute_error,
            "output_relative_l2": relative_l2,
        },
        "dispatch_calls": tracked.calls,
        "thresholds": {
            "maximum_absolute_error": MAX_ABS_LIMIT,
            "maximum_relative_l2": RELATIVE_L2_LIMIT,
        },
        "qualified": qualified,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
