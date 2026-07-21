#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable


NUM_ATTENTION_LAYERS = 10
BLOCK_SIZE = 16
LOCAL_NUM_KV_HEADS = 1
HEAD_SIZE = 256
DTYPE_BYTES = 2

SMOKE_TOKEN_COUNTS = (4096,)
GATE_TOKEN_COUNTS = (4096, 16384, 65536, 131072)
WARMUP_CYCLES = 1
MEASURED_CYCLES = 3

GATE_LIMITS_MS = {
    65536: {
        "d2h_median_max_ms": 2000.0,
        "h2d_median_max_ms": 2000.0,
    },
    131072: {
        "round_trip_median_max_ms": 5000.0,
    },
}


def blocks_for_tokens(token_count: int) -> int:
    if token_count <= 0 or token_count % BLOCK_SIZE:
        raise ValueError(
            f"token_count must be a positive multiple of {BLOCK_SIZE}")
    return token_count // BLOCK_SIZE


def bytes_per_block_per_rank() -> int:
    return (NUM_ATTENTION_LAYERS * 2 * BLOCK_SIZE * LOCAL_NUM_KV_HEADS
            * HEAD_SIZE * DTYPE_BYTES)


def bytes_for_tokens(token_count: int) -> int:
    return blocks_for_tokens(token_count) * bytes_per_block_per_rank()


def transfer_gib_per_second(byte_count: int, elapsed_ms: float) -> float:
    if elapsed_ms <= 0 or not math.isfinite(elapsed_ms):
        raise ValueError("elapsed_ms must be finite and positive")
    return byte_count / (1024**3) / (elapsed_ms / 1000.0)


def evaluate_gate(mode: str,
                  results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    expected = SMOKE_TOKEN_COUNTS if mode == "smoke" else GATE_TOKEN_COUNTS
    for token_count in expected:
        case = results.get(str(token_count))
        if case is None:
            reasons.append(f"missing case {token_count}")
            continue
        if not case.get("exact", False):
            reasons.append(f"case {token_count} is not exact")
        for field in ("d2h_median_ms", "h2d_median_ms"):
            value = case.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                reasons.append(f"case {token_count} has invalid {field}")
            elif value <= 0:
                reasons.append(f"case {token_count} has non-positive {field}")

    if mode == "smoke":
        return {
            "qualified": False,
            "smoke_passed": not reasons,
            "reasons": reasons or ["smoke mode cannot qualify the candidate"],
        }

    for token_count, limits in GATE_LIMITS_MS.items():
        case = results.get(str(token_count))
        if case is None:
            continue
        d2h_ms = case.get("d2h_median_ms")
        h2d_ms = case.get("h2d_median_ms")
        if not isinstance(d2h_ms, (int, float)):
            continue
        if not isinstance(h2d_ms, (int, float)):
            continue
        if d2h_ms > limits.get("d2h_median_max_ms", math.inf):
            reasons.append(
                f"case {token_count} D2H {d2h_ms:.3f} ms exceeds "
                f"{limits['d2h_median_max_ms']:.3f} ms")
        if h2d_ms > limits.get("h2d_median_max_ms", math.inf):
            reasons.append(
                f"case {token_count} H2D {h2d_ms:.3f} ms exceeds "
                f"{limits['h2d_median_max_ms']:.3f} ms")
        round_trip_ms = d2h_ms + h2d_ms
        if round_trip_ms > limits.get("round_trip_median_max_ms", math.inf):
            reasons.append(
                f"case {token_count} round trip {round_trip_ms:.3f} ms "
                f"exceeds {limits['round_trip_median_max_ms']:.3f} ms")

    return {
        "qualified": not reasons,
        "smoke_passed": None,
        "reasons": reasons,
    }


def _measure(operation: Callable[[], None], synchronize: Callable[[], None]
             ) -> float:
    synchronize()
    started = time.perf_counter()
    operation()
    synchronize()
    return (time.perf_counter() - started) * 1000.0


def _probe_blocks(num_blocks: int) -> list[int]:
    return sorted({0, num_blocks // 3, num_blocks // 2, num_blocks - 1})


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the production BI100 paged-KV swap_blocks path using "
            "fixed Qwen3.6 TP4 rank-local shapes."))
    parser.add_argument("--mode", choices=("smoke", "gate"), default="gate")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import vllm
    from vllm.attention.ops.paged_attn import PagedAttention

    if not args.device.startswith("cuda:"):
        parser.error("--device must name one CUDA device, for example cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    token_counts = (SMOKE_TOKEN_COUNTS if args.mode == "smoke"
                    else GATE_TOKEN_COUNTS)
    max_blocks = blocks_for_tokens(max(token_counts))
    expected_shape = (
        2,
        max_blocks,
        BLOCK_SIZE * LOCAL_NUM_KV_HEADS * HEAD_SIZE,
    )
    actual_shape = PagedAttention.get_kv_cache_shape(
        max_blocks, BLOCK_SIZE, LOCAL_NUM_KV_HEADS, HEAD_SIZE)
    if tuple(actual_shape) != expected_shape:
        raise RuntimeError(
            f"unexpected paged KV shape {actual_shape}, expected {expected_shape}")

    gpu_cache = [
        torch.zeros(actual_shape, dtype=torch.float16, device=device)
        for _ in range(NUM_ATTENTION_LAYERS)
    ]
    cpu_cache = [
        torch.empty(actual_shape,
                    dtype=torch.float16,
                    device="cpu",
                    pin_memory=True)
        for _ in range(NUM_ATTENTION_LAYERS)
    ]
    if not all(cache.is_pinned() for cache in cpu_cache):
        raise RuntimeError("CPU KV cache is not pinned")

    all_probe_blocks = sorted({
        block
        for token_count in token_counts
        for block in _probe_blocks(blocks_for_tokens(token_count))
    })
    expected: dict[tuple[int, int], Any] = {}
    for layer_index, layer_cache in enumerate(gpu_cache):
        for probe_index, block_index in enumerate(all_probe_blocks):
            value = float((layer_index + 1) * 16 + probe_index)
            layer_cache[:, block_index, :].fill_(value)
            expected[(layer_index, block_index)] = (
                layer_cache[:, block_index, :].cpu().clone())
    torch.cuda.synchronize(device)

    def swap_all(src: list[Any], dst: list[Any], mapping: Any) -> None:
        for layer_index in range(NUM_ATTENTION_LAYERS):
            PagedAttention.swap_blocks(
                src[layer_index], dst[layer_index], mapping)

    results: dict[str, dict[str, Any]] = {}
    for token_count in token_counts:
        num_blocks = blocks_for_tokens(token_count)
        ids = torch.arange(num_blocks, dtype=torch.int64, device="cpu")
        mapping = torch.stack((ids, ids), dim=1)
        probes = _probe_blocks(num_blocks)

        d2h = lambda: swap_all(gpu_cache, cpu_cache, mapping)
        h2d = lambda: swap_all(cpu_cache, gpu_cache, mapping)
        synchronize = lambda: torch.cuda.synchronize(device)

        d2h()
        synchronize()
        cpu_exact = all(
            torch.equal(cpu_cache[layer][:, block, :], expected[(layer, block)])
            for layer in range(NUM_ATTENTION_LAYERS)
            for block in probes
        )
        for layer in range(NUM_ATTENTION_LAYERS):
            for block in probes:
                gpu_cache[layer][:, block, :].fill_(-1.0)
        h2d()
        synchronize()
        gpu_exact = all(
            torch.equal(gpu_cache[layer][:, block, :].cpu(),
                        expected[(layer, block)])
            for layer in range(NUM_ATTENTION_LAYERS)
            for block in probes
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
            "token_count": token_count,
            "num_blocks": num_blocks,
            "bytes_per_direction": byte_count,
            "probe_blocks": probes,
            "cpu_exact_after_d2h": cpu_exact,
            "gpu_exact_after_h2d": gpu_exact,
            "exact": cpu_exact and gpu_exact,
            "d2h_median_ms": d2h_median_ms,
            "h2d_median_ms": h2d_median_ms,
            "round_trip_median_ms": d2h_median_ms + h2d_median_ms,
            "d2h_gib_per_second": transfer_gib_per_second(
                byte_count, d2h_median_ms),
            "h2d_gib_per_second": transfer_gib_per_second(
                byte_count, h2d_median_ms),
            "trials": trials,
        }

    decision = evaluate_gate(args.mode, results)
    report = {
        "schema": "bi100-cpu-kv-offload-capability-v1",
        "mode": args.mode,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "vllm_version": getattr(vllm, "__version__", "unknown"),
        "vllm_path": str(Path(vllm.__file__).resolve()),
        "shape": {
            "num_attention_layers": NUM_ATTENTION_LAYERS,
            "block_size": BLOCK_SIZE,
            "local_num_kv_heads": LOCAL_NUM_KV_HEADS,
            "head_size": HEAD_SIZE,
            "dtype": "float16",
            "bytes_per_block_per_rank": bytes_per_block_per_rank(),
        },
        "fixed_protocol": {
            "token_counts": list(token_counts),
            "warmup_cycles": WARMUP_CYCLES,
            "measured_cycles": MEASURED_CYCLES,
            "gate_limits_ms": GATE_LIMITS_MS,
            "mapping": "identity, CPU int64 [num_blocks, 2]",
            "operation_order": "alternating D2H/H2D after one warmup cycle",
        },
        "results": results,
        "decision": decision,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))

    passed = (decision["smoke_passed"] if args.mode == "smoke"
              else decision["qualified"])
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
