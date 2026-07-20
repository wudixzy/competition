#!/usr/bin/env python3
"""Validate that a service log proves the requested KV-cache capacity."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re


MAX_SEQ_RE = re.compile(r"\bmax_seq_len=(\d+)\b")
GPU_BLOCK_RE = re.compile(r"# GPU blocks:\s*(\d+)\b")


def evaluate(log_text: str, max_model_len: int, block_size: int) -> dict:
    max_seq_values = [int(value) for value in MAX_SEQ_RE.findall(log_text)]
    gpu_block_values = [int(value) for value in GPU_BLOCK_RE.findall(log_text)]
    max_seq_len = max_seq_values[-1] if max_seq_values else None
    gpu_blocks = gpu_block_values[-1] if gpu_block_values else None
    required_gpu_blocks = math.ceil(max_model_len / block_size)
    logical_window_ok = (
        max_seq_len is not None and max_seq_len >= max_model_len)
    physical_capacity_ok = (
        gpu_blocks is not None and gpu_blocks >= required_gpu_blocks)
    return {
        "max_model_len_required": max_model_len,
        "block_size": block_size,
        "required_gpu_blocks": required_gpu_blocks,
        "observed_max_seq_len": max_seq_len,
        "observed_gpu_blocks": gpu_blocks,
        "observed_physical_tokens": (
            gpu_blocks * block_size if gpu_blocks is not None else None),
        "logical_window_ok": logical_window_ok,
        "physical_capacity_ok": physical_capacity_ok,
        "qualified": logical_window_ok and physical_capacity_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--max-model-len", type=int, default=262144)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.max_model_len <= 0:
        parser.error("--max-model-len must be positive")
    if args.block_size <= 0:
        parser.error("--block-size must be positive")

    report = evaluate(
        args.log.read_text(encoding="utf-8", errors="replace"),
        args.max_model_len,
        args.block_size,
    )
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
