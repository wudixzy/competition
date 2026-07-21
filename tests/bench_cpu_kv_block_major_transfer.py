#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from bench_cpu_kv_offload_transfer import (
    BLOCK_SIZE,
    DTYPE_BYTES,
    HEAD_SIZE,
    LOCAL_NUM_KV_HEADS,
    MEASURED_CYCLES,
    NUM_ATTENTION_LAYERS,
    WARMUP_CYCLES,
    blocks_for_tokens,
    bytes_for_tokens,
    transfer_gib_per_second,
)


TOKEN_COUNTS = (65_536, 131_072)
REORDER_BLOCKS = 513


def _measure(operation: Callable[[], None], synchronize: Callable[[], None]
             ) -> float:
    synchronize()
    started = time.perf_counter()
    operation()
    synchronize()
    return (time.perf_counter() - started) * 1000.0


def _probe_blocks(num_blocks: int) -> list[int]:
    return sorted({0, num_blocks // 3, num_blocks // 2, num_blocks - 1})


def _finite_positive(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def evaluate(results: dict[str, dict[str, Any]],
             reordered_mapping_exact: bool,
             worker_d2h_before_h2d: bool) -> dict[str, Any]:
    reasons = []
    if not reordered_mapping_exact:
        reasons.append("513-block reordered mapping is not exact")
    if not worker_d2h_before_h2d:
        reasons.append("installed worker does not execute D2H before H2D")
    for token_count in TOKEN_COUNTS:
        case = results.get(str(token_count))
        if not isinstance(case, dict):
            reasons.append(f"missing case {token_count}")
            continue
        if case.get("exact") is not True:
            reasons.append(f"case {token_count} is not exact")
        for field in ("d2h_median_ms", "h2d_median_ms"):
            if not _finite_positive(case.get(field)):
                reasons.append(f"case {token_count} has invalid {field}")
    return {
        "diagnostic_passed": not reasons,
        "qualified": False,
        "reasons": reasons or [
            "production probe requires comparison with paged evidence"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure the fixed BI100 block-major CPU KV data plane")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import vllm
    from vllm.attention.ops.paged_attn import PagedAttention
    from vllm.worker.bi100_block_major_kv import (
        Bi100BlockMajorKvTransfer, STAGING_BLOCKS)
    from vllm.worker.worker import Worker

    if not args.device.startswith("cuda:"):
        parser.error("--device must name one CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    worker_source = inspect.getsource(Worker.execute_worker)
    worker_d2h_before_h2d = (
        worker_source.find(".swap_out(") >= 0
        and worker_source.find(".swap_in(") >= 0
        and worker_source.find(".swap_out(") < worker_source.find(".swap_in(")
    )

    max_blocks = blocks_for_tokens(max(TOKEN_COUNTS))
    cache_shape = PagedAttention.get_kv_cache_shape(
        max_blocks, BLOCK_SIZE, LOCAL_NUM_KV_HEADS, HEAD_SIZE)
    gpu_cache = [
        torch.empty(cache_shape, dtype=torch.float16, device=device)
        for _ in range(NUM_ATTENTION_LAYERS)
    ]
    generator = torch.Generator(device=device)
    generator.manual_seed(20260721)
    for cache in gpu_cache:
        cache.uniform_(-1.0, 1.0, generator=generator)
    expected_gpu = [cache.clone() for cache in gpu_cache]
    transfer = Bi100BlockMajorKvTransfer(gpu_cache, max_blocks)

    reorder_ids = torch.arange(REORDER_BLOCKS, dtype=torch.int64)
    cpu_destinations = (reorder_ids * 17 + 3) % REORDER_BLOCKS
    gpu_destinations = (reorder_ids * 29 + 5) % REORDER_BLOCKS
    gpu_destinations_device = gpu_destinations.to(device)
    d2h_reordered = torch.stack((reorder_ids, cpu_destinations), dim=1)
    h2d_reordered = torch.stack((cpu_destinations, gpu_destinations), dim=1)
    transfer.swap_out(d2h_reordered)
    torch.cuda.synchronize(device)
    reordered_cpu_exact = True
    for layer_index, expected in enumerate(expected_gpu):
        expected_rows = expected.view(2, max_blocks, -1)[
            :, :REORDER_BLOCKS, :].permute(1, 0, 2).cpu()
        actual_rows = transfer.cpu_cache[
            cpu_destinations, layer_index, :, :]
        reordered_cpu_exact = (
            reordered_cpu_exact and torch.equal(actual_rows, expected_rows))
    for cache in gpu_cache:
        cache.view(2, max_blocks, -1)[:, :REORDER_BLOCKS, :].fill_(-1.0)
    transfer.swap_in(h2d_reordered)
    torch.cuda.synchronize(device)
    reordered_gpu_exact = True
    for layer_index, expected in enumerate(expected_gpu):
        expected_rows = expected.view(2, max_blocks, -1)[
            :, :REORDER_BLOCKS, :]
        actual_rows = gpu_cache[layer_index].view(2, max_blocks, -1)[
            :, gpu_destinations_device, :]
        reordered_gpu_exact = (
            reordered_gpu_exact and torch.equal(actual_rows, expected_rows))
    reordered_mapping_exact = reordered_cpu_exact and reordered_gpu_exact

    for cache, expected in zip(gpu_cache, expected_gpu):
        cache.copy_(expected)
    torch.cuda.synchronize(device)

    results: dict[str, dict[str, Any]] = {}
    for token_count in TOKEN_COUNTS:
        num_blocks = blocks_for_tokens(token_count)
        ids = torch.arange(num_blocks, dtype=torch.int64)
        mapping = torch.stack((ids, ids), dim=1)
        probes = _probe_blocks(num_blocks)
        synchronize = lambda: torch.cuda.synchronize(device)
        d2h = lambda: transfer.swap_out(mapping)
        h2d = lambda: transfer.swap_in(mapping)

        d2h()
        synchronize()
        cpu_exact = True
        for layer_index, expected in enumerate(expected_gpu):
            expected_rows = expected.view(2, max_blocks, -1)[
                :, :num_blocks, :].permute(1, 0, 2).cpu()
            actual_rows = transfer.cpu_cache[:num_blocks, layer_index, :, :]
            cpu_exact = cpu_exact and torch.equal(actual_rows, expected_rows)
        for cache in gpu_cache:
            cache.view(2, max_blocks, -1)[:, :num_blocks, :].fill_(-1.0)
        h2d()
        synchronize()
        gpu_exact = all(
            torch.equal(
                gpu_cache[layer].view(2, max_blocks, -1)[:, :num_blocks, :],
                expected_gpu[layer].view(2, max_blocks, -1)[
                    :, :num_blocks, :],
            )
            for layer in range(NUM_ATTENTION_LAYERS)
        )

        for _ in range(WARMUP_CYCLES):
            d2h()
            h2d()
        synchronize()
        trials = {"d2h_ms": [], "h2d_ms": []}
        for cycle in range(MEASURED_CYCLES):
            order = (("d2h_ms", d2h), ("h2d_ms", h2d))
            if cycle % 2:
                order = tuple(reversed(order))
            for name, operation in order:
                trials[name].append(_measure(operation, synchronize))

        d2h_median_ms = statistics.median(trials["d2h_ms"])
        h2d_median_ms = statistics.median(trials["h2d_ms"])
        byte_count = bytes_for_tokens(token_count)
        results[str(token_count)] = {
            "bytes_per_direction": byte_count,
            "cpu_exact_after_d2h": cpu_exact,
            "d2h_gib_per_second": transfer_gib_per_second(
                byte_count, d2h_median_ms),
            "d2h_median_ms": d2h_median_ms,
            "exact": cpu_exact and gpu_exact,
            "gpu_exact_after_h2d": gpu_exact,
            "h2d_gib_per_second": transfer_gib_per_second(
                byte_count, h2d_median_ms),
            "h2d_median_ms": h2d_median_ms,
            "num_blocks": num_blocks,
            "probe_blocks": probes,
            "round_trip_median_ms": d2h_median_ms + h2d_median_ms,
            "token_count": token_count,
            "trials": trials,
        }

    report = {
        "decision": evaluate(
            results, reordered_mapping_exact, worker_d2h_before_h2d),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "protocol": {
            "mapping": "CPU int64 [N, 2], identity timing plus reordered gate",
            "measured_cycles": MEASURED_CYCLES,
            "staging_blocks": STAGING_BLOCKS,
            "token_counts": list(TOKEN_COUNTS),
            "warmup_cycles": WARMUP_CYCLES,
        },
        "reordered_mapping_exact": reordered_mapping_exact,
        "reordered_mapping_blocks": REORDER_BLOCKS,
        "results": results,
        "schema": "bi100-cpu-kv-block-major-transfer-v1",
        "shape": {
            "block_size": BLOCK_SIZE,
            "bytes_per_block_per_rank": (
                NUM_ATTENTION_LAYERS * 2 * BLOCK_SIZE
                * LOCAL_NUM_KV_HEADS * HEAD_SIZE * DTYPE_BYTES
            ),
            "dtype": "float16",
            "head_size": HEAD_SIZE,
            "local_num_kv_heads": LOCAL_NUM_KV_HEADS,
            "num_attention_layers": NUM_ATTENTION_LAYERS,
        },
        "torch_version": torch.__version__,
        "version": 1,
        "vllm_path": vllm.__file__,
        "vllm_version": getattr(vllm, "__version__", "unknown"),
        "worker_d2h_before_h2d": worker_d2h_before_h2d,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["decision"]["diagnostic_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
