#!/usr/bin/env python3
"""Measure the fixed contiguous-DMA upper bound for BI100 rank-local KV."""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Callable


SCHEMA = "bi100-cpu-kv-contiguous-transfer-v1"
VERSION = 1
BLOCK_SIZE = 16
NUM_ATTENTION_LAYERS = 10
LOCAL_NUM_KV_HEADS = 1
HEAD_SIZE = 256
DTYPE_BYTES = 2
TOKEN_COUNTS = (65_536, 131_072)
WARMUP_CYCLES = 1
MEASURED_CYCLES = 3


def bytes_per_block_per_rank() -> int:
    return (NUM_ATTENTION_LAYERS * 2 * BLOCK_SIZE * LOCAL_NUM_KV_HEADS
            * HEAD_SIZE * DTYPE_BYTES)


def bytes_for_tokens(token_count: int) -> int:
    if token_count <= 0 or token_count % BLOCK_SIZE:
        raise ValueError(
            f"token_count must be a positive multiple of {BLOCK_SIZE}")
    return token_count // BLOCK_SIZE * bytes_per_block_per_rank()


def gib_per_second(byte_count: int, elapsed_ms: float) -> float:
    if elapsed_ms <= 0 or not math.isfinite(elapsed_ms):
        raise ValueError("elapsed_ms must be finite and positive")
    return byte_count / 1024**3 / (elapsed_ms / 1000.0)


def evaluate(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    for token_count in TOKEN_COUNTS:
        case = results.get(str(token_count))
        if not isinstance(case, dict):
            reasons.append(f"missing case {token_count}")
            continue
        if case.get("exact") is not True:
            reasons.append(f"case {token_count} is not exact")
        for field in ("d2h_median_ms", "h2d_median_ms"):
            value = case.get(field)
            if (not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                    or value <= 0):
                reasons.append(f"case {token_count} has invalid {field}")
    return {
        "diagnostic_passed": not reasons,
        "qualified": False,
        "reasons": reasons or [
            "capability probe requires comparison with paged transfer evidence"
        ],
    }


def _measure(operation: Callable[[], None], synchronize: Callable[[], None]
             ) -> float:
    synchronize()
    started = time.perf_counter()
    operation()
    synchronize()
    return (time.perf_counter() - started) * 1000.0


def atomic_write(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                     dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure contiguous CPU/GPU copies at Qwen3.6 TP4 KV size")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    import torch

    if not args.device.startswith("cuda:"):
        parser.error("--device must name one CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    max_bytes = bytes_for_tokens(max(TOKEN_COUNTS))
    if max_bytes % DTYPE_BYTES:
        raise RuntimeError("KV byte count is not dtype aligned")
    elements = max_bytes // DTYPE_BYTES
    gpu = torch.empty(elements, dtype=torch.float16, device=device)
    cpu = torch.empty(elements, dtype=torch.float16, pin_memory=True)
    if not cpu.is_pinned():
        raise RuntimeError("CPU transfer buffer is not pinned")

    results: dict[str, dict[str, Any]] = {}
    synchronize = lambda: torch.cuda.synchronize(device)
    for token_count in TOKEN_COUNTS:
        byte_count = bytes_for_tokens(token_count)
        element_count = byte_count // DTYPE_BYTES
        gpu_view = gpu[:element_count]
        cpu_view = cpu[:element_count]
        probe_indices = sorted({0, element_count // 3,
                                element_count // 2, element_count - 1})
        expected = torch.arange(1, len(probe_indices) + 1,
                                dtype=torch.float16)
        gpu_view[probe_indices] = expected.to(device)
        synchronize()

        d2h = lambda: cpu_view.copy_(gpu_view, non_blocking=True)
        h2d = lambda: gpu_view.copy_(cpu_view, non_blocking=True)
        d2h()
        synchronize()
        cpu_exact = torch.equal(cpu_view[probe_indices], expected)
        gpu_view[probe_indices] = -1
        h2d()
        synchronize()
        gpu_exact = torch.equal(gpu_view[probe_indices].cpu(), expected)

        for _ in range(WARMUP_CYCLES):
            d2h()
            h2d()
        synchronize()

        trials = {"d2h_ms": [], "h2d_ms": []}
        for cycle in range(MEASURED_CYCLES):
            operations = (("d2h_ms", d2h), ("h2d_ms", h2d))
            if cycle % 2:
                operations = tuple(reversed(operations))
            for name, operation in operations:
                trials[name].append(_measure(operation, synchronize))
        d2h_ms = statistics.median(trials["d2h_ms"])
        h2d_ms = statistics.median(trials["h2d_ms"])
        results[str(token_count)] = {
            "token_count": token_count,
            "bytes_per_direction": byte_count,
            "probe_indices": probe_indices,
            "cpu_exact_after_d2h": cpu_exact,
            "gpu_exact_after_h2d": gpu_exact,
            "exact": cpu_exact and gpu_exact,
            "d2h_median_ms": d2h_ms,
            "h2d_median_ms": h2d_ms,
            "round_trip_median_ms": d2h_ms + h2d_ms,
            "d2h_gib_per_second": gib_per_second(byte_count, d2h_ms),
            "h2d_gib_per_second": gib_per_second(byte_count, h2d_ms),
            "trials": trials,
        }

    decision = evaluate(results)
    report = {
        "schema": SCHEMA,
        "version": VERSION,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "shape": {
            "num_attention_layers": NUM_ATTENTION_LAYERS,
            "block_size": BLOCK_SIZE,
            "local_num_kv_heads": LOCAL_NUM_KV_HEADS,
            "head_size": HEAD_SIZE,
            "dtype": "float16",
            "bytes_per_block_per_rank": bytes_per_block_per_rank(),
        },
        "protocol": {
            "layout": "single contiguous pinned CPU and GPU tensor",
            "token_counts": list(TOKEN_COUNTS),
            "warmup_cycles": WARMUP_CYCLES,
            "measured_cycles": MEASURED_CYCLES,
            "copy": "Tensor.copy_(non_blocking=True) with device synchronize",
        },
        "results": results,
        "decision": decision,
    }
    atomic_write(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if decision["diagnostic_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
