#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any


NUM_ATTENTION_LAYERS = 10
KV_PLANES = 2
ELEMENTS_PER_PLANE_BLOCK = 4096
GPU_BLOCKS = 1025
CPU_BLOCKS = 1536
FIXED_SEED = 20260726
EXPECTED_EXTENSION_SHA256 = (
    "7e2aafd8dc755b0ee16c3b9bb812b95548fc042bbaa840dd9db7d2c51a10474c"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fill_unique_gpu_cache(torch: Any, gpu_cache: list[Any]) -> bool:
    base = torch.arange(
        gpu_cache[0].numel(),
        dtype=torch.int64,
        device=gpu_cache[0].device,
    ).reshape(gpu_cache[0].shape)
    pattern = ((base * 37 + 113) % 997).to(dtype=torch.float16)
    for layer, tensor in enumerate(gpu_cache):
        tensor.copy_(pattern + float(layer * 1000))

    block_ids = torch.arange(
        GPU_BLOCKS,
        dtype=torch.int64,
        device=gpu_cache[0].device,
    )
    low = (block_ids % 256).to(dtype=torch.float16)
    high = (block_ids // 256).to(dtype=torch.float16)
    for layer, tensor in enumerate(gpu_cache):
        tensor[0, :, 0] = low
        tensor[0, :, 1] = high
        tensor[0, :, 2] = float(layer)
    recovered = (
        gpu_cache[0][0, :, 0].to(dtype=torch.int64)
        + 256 * gpu_cache[0][0, :, 1].to(dtype=torch.int64)
    )
    return recovered.equal(block_ids)


def _all_restored_exact(
    torch: Any,
    gpu_cache: list[Any],
    original: list[Any],
) -> bool:
    return all(
        tensor.cpu().equal(expected)
        for tensor, expected in zip(gpu_cache, original)
    )


def _finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the fixed M1-57 CacheEngine integration smoke")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--instance", required=True)
    args = parser.parse_args()

    import torch

    from vllm.block_major_kv_cache import (
        BYTES_PER_BLOCK,
        BlockMajorCpuKVCache,
        block_major_cpu_kv_enabled,
    )
    from vllm.worker.cache_engine import CacheEngine
    import vllm

    if not args.device.startswith("cuda:"):
        parser.error("--device must name one CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    extension_path = (
        Path(vllm.__file__).resolve().parent
        / "corex_block_major_kv_transfer.so"
    )
    if not extension_path.is_file():
        raise FileNotFoundError(extension_path)

    report: dict[str, Any] = {
        "schema": "bi100-m1-57-cache-engine-integration-v1",
        "experiment": "M1-57-block-major-cache-engine",
        "source_revision": args.source_revision,
        "instance": args.instance,
        "device": args.device,
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "gpu_blocks": GPU_BLOCKS,
        "cpu_blocks": CPU_BLOCKS,
        "attention_layers": NUM_ATTENTION_LAYERS,
        "bytes_per_block": BYTES_PER_BLOCK,
        "extension_sha256": _sha256(extension_path),
        "default_selector_off": block_major_cpu_kv_enabled({}) is False,
    }
    report["extension_sha_exact"] = (
        report["extension_sha256"] == EXPECTED_EXTENSION_SHA256)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        cache_shape = (
            KV_PLANES,
            GPU_BLOCKS,
            ELEMENTS_PER_PLANE_BLOCK,
        )
        gpu_cache = [
            torch.empty(
                cache_shape,
                dtype=torch.float16,
                device=device,
            )
            for _ in range(NUM_ATTENTION_LAYERS)
        ]
        report["unique_block_signatures"] = _fill_unique_gpu_cache(
            torch, gpu_cache)
        original = [tensor.cpu().clone() for tensor in gpu_cache]

        started = time.perf_counter()
        transfer = BlockMajorCpuKVCache(
            gpu_cache,
            num_cpu_blocks=CPU_BLOCKS,
            pin_memory=True,
        )
        report["allocation_elapsed_ms"] = (
            time.perf_counter() - started) * 1000.0
        report["cpu_pool_shape"] = list(transfer.cpu_pool.shape)
        report["layer_view_shapes"] = [
            list(view.shape) for view in transfer.layer_views
        ]
        report["cpu_pool_pinned"] = transfer.cpu_pool.is_pinned()
        report["cpu_null_block_zero"] = (
            transfer.cpu_pool[0].count_nonzero().item() == 0)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(FIXED_SEED)
        source_gpu = torch.randperm(
            GPU_BLOCKS, generator=generator, dtype=torch.int64)
        destination_cpu = torch.randperm(
            CPU_BLOCKS, generator=generator, dtype=torch.int64)[:GPU_BLOCKS]
        swap_out_map = torch.stack(
            (source_gpu, destination_cpu), dim=1).contiguous()
        swap_in_map = torch.stack(
            (destination_cpu, source_gpu), dim=1).contiguous()

        engine = CacheEngine.__new__(CacheEngine)
        engine._bi100_block_major_cpu_kv = transfer
        transfer_started = time.perf_counter()
        engine.swap_out(swap_out_map)
        for tensor in gpu_cache:
            tensor.fill_(-1)
        engine.swap_in(swap_in_map)
        torch.cuda.synchronize(device)
        report["round_trip_elapsed_ms"] = (
            time.perf_counter() - transfer_started) * 1000.0
        report["round_trip_byte_exact"] = _all_restored_exact(
            torch, gpu_cache, original)

        victim_gpu = int(source_gpu[0])
        preserved_cpu = int(destination_cpu[0])
        requested_cpu = int(destination_cpu[1])
        expected_victim = torch.stack([
            gpu_cache[layer][:, victim_gpu, :].cpu()
            for layer in range(NUM_ATTENTION_LAYERS)
        ])
        expected_requested = transfer.cpu_pool[requested_cpu].clone()
        engine.swap_out(torch.tensor(
            [[victim_gpu, preserved_cpu]], dtype=torch.int64))
        engine.swap_in(torch.tensor(
            [[requested_cpu, victim_gpu]], dtype=torch.int64))
        torch.cuda.synchronize(device)
        actual_victim = transfer.cpu_pool[preserved_cpu]
        actual_requested = torch.stack([
            gpu_cache[layer][:, victim_gpu, :].cpu()
            for layer in range(NUM_ATTENTION_LAYERS)
        ])
        report["same_slot_preserved_victim_exact"] = (
            actual_victim.equal(expected_victim))
        report["same_slot_promoted_request_exact"] = (
            actual_requested.equal(expected_requested))

        protected_slot = transfer.cpu_pool[2].clone()
        invalid_mapping_failed = False
        try:
            engine.swap_out(torch.tensor(
                [[GPU_BLOCKS, 2]], dtype=torch.int64))
        except ValueError as exc:
            invalid_mapping_failed = (
                "source block out of range" in str(exc))
        report["invalid_mapping_fail_fast"] = invalid_mapping_failed
        report["invalid_mapping_zero_write"] = (
            transfer.cpu_pool[2].equal(protected_slot))

        invalid_selector_failed = False
        try:
            block_major_cpu_kv_enabled({
                "BI100_BLOCK_MAJOR_CPU_KV": "true",
            })
        except RuntimeError as exc:
            invalid_selector_failed = "exactly '0' or '1'" in str(exc)
        report["invalid_selector_fail_fast"] = invalid_selector_failed

        torch.cuda.synchronize(device)
        report["gpu_memory_allocated_bytes"] = torch.cuda.memory_allocated(
            device)
        bool_gates = (
            "default_selector_off",
            "extension_sha_exact",
            "unique_block_signatures",
            "cpu_pool_pinned",
            "cpu_null_block_zero",
            "round_trip_byte_exact",
            "same_slot_preserved_victim_exact",
            "same_slot_promoted_request_exact",
            "invalid_mapping_fail_fast",
            "invalid_mapping_zero_write",
            "invalid_selector_fail_fast",
        )
        reasons = [
            f"{name} is not true"
            for name in bool_gates
            if report.get(name) is not True
        ]
        if report["cpu_pool_shape"] != [
                CPU_BLOCKS,
                NUM_ATTENTION_LAYERS,
                KV_PLANES,
                ELEMENTS_PER_PLANE_BLOCK]:
            reasons.append("CPU pool geometry changed")
        if report["layer_view_shapes"] != [[
                KV_PLANES,
                CPU_BLOCKS,
                ELEMENTS_PER_PLANE_BLOCK,
        ]] * NUM_ATTENTION_LAYERS:
            reasons.append("compatibility layer-view geometry changed")
        for name in (
                "allocation_elapsed_ms",
                "round_trip_elapsed_ms",
                "gpu_memory_allocated_bytes"):
            if not _finite_nonnegative(report.get(name)):
                reasons.append(f"{name} is not finite and non-negative")
        report["gate"] = {
            "qualified": not reasons,
            "reasons": reasons,
        }
    except Exception as exc:
        report["fatal"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        report["gate"] = {
            "qualified": False,
            "reasons": [f"fatal {type(exc).__name__}: {exc}"],
        }

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.out.write_text(rendered, encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "qualified": report["gate"]["qualified"],
        "reasons": report["gate"]["reasons"],
    }, sort_keys=True))
    return 0 if report["gate"]["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
