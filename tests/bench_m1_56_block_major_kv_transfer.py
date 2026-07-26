#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Sequence


NUM_ATTENTION_LAYERS = 10
KV_PLANES = 2
BLOCK_SIZE = 16
ELEMENTS_PER_PLANE_BLOCK = 4096
DTYPE_BYTES = 2
STAGING_BLOCKS = 512
FIXED_SEED = 20260726
REQUIRED_SPEEDUP = 4.0
BOUNDARY_TOKEN_COUNTS = (8192, 8208)
GATE_TOKEN_COUNTS = (65536, 131072)


def blocks_for_tokens(token_count: int) -> int:
    if token_count <= 0 or token_count % BLOCK_SIZE:
        raise ValueError(
            f"token_count must be a positive multiple of {BLOCK_SIZE}")
    return token_count // BLOCK_SIZE


def bytes_per_block_per_rank() -> int:
    return (NUM_ATTENTION_LAYERS * KV_PLANES
            * ELEMENTS_PER_PLANE_BLOCK * DTYPE_BYTES)


def bytes_for_tokens(token_count: int) -> int:
    return blocks_for_tokens(token_count) * bytes_per_block_per_rank()


def staging_buffer_count(pipeline: str) -> int:
    if pipeline == "single":
        return 1
    if pipeline == "double":
        return 2
    raise ValueError(f"unknown transfer pipeline: {pipeline}")


def mapping_chunks(source: Sequence[int],
                   destination: Sequence[int],
                   chunk_blocks: int = STAGING_BLOCKS,
                   ) -> list[tuple[list[int], list[int]]]:
    if len(source) != len(destination):
        raise ValueError("source and destination mappings differ in length")
    if not source:
        raise ValueError("mapping must not be empty")
    if chunk_blocks <= 0:
        raise ValueError("chunk_blocks must be positive")
    if len(set(source)) != len(source):
        raise ValueError("source mapping contains duplicates")
    if len(set(destination)) != len(destination):
        raise ValueError("destination mapping contains duplicates")
    if min(source) < 0 or min(destination) < 0:
        raise ValueError("mapping indices must be non-negative")
    return [
        (list(source[start:start + chunk_blocks]),
         list(destination[start:start + chunk_blocks]))
        for start in range(0, len(source), chunk_blocks)
    ]


def _finite_positive(value: Any) -> bool:
    return (isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def evaluate_gate(report: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    cases = report.get("cases")
    if not isinstance(cases, dict):
        return {"qualified": False, "reasons": ["missing cases"]}

    expected_tokens = (
        BOUNDARY_TOKEN_COUNTS if report.get("mode") == "smoke"
        else BOUNDARY_TOKEN_COUNTS + GATE_TOKEN_COUNTS)
    for token_count in expected_tokens:
        case = cases.get(str(token_count))
        if not isinstance(case, dict):
            reasons.append(f"missing case {token_count}")
            continue
        if case.get("d2h_exact") is not True:
            reasons.append(f"case {token_count} D2H is not byte-exact")
        if case.get("h2d_exact") is not True:
            reasons.append(f"case {token_count} H2D is not byte-exact")
        if case.get("component_same_gpu_slot_order_exact") is not True:
            reasons.append(
                f"case {token_count} component same-GPU-slot ordering failed")
        if case.get("invalid_gpu_mapping_fail_fast") is not True:
            reasons.append(
                f"case {token_count} invalid GPU mapping did not fail fast")
        if case.get("unique_block_signatures") is not True:
            reasons.append(
                f"case {token_count} block signatures are not unique")

    if report.get("mode") == "smoke":
        return {
            "qualified": False,
            "smoke_passed": not reasons,
            "reasons": reasons or [
                "smoke mode cannot qualify the production data plane"
            ],
        }

    for token_count in GATE_TOKEN_COUNTS:
        case = cases.get(str(token_count))
        if not isinstance(case, dict):
            continue
        for direction in ("d2h", "h2d"):
            speedup = case.get(f"{direction}_speedup")
            if not _finite_positive(speedup):
                reasons.append(
                    f"case {token_count} has invalid {direction} speedup")
            elif speedup < REQUIRED_SPEEDUP:
                reasons.append(
                    f"case {token_count} {direction} speedup "
                    f"{speedup:.3f}x is below {REQUIRED_SPEEDUP:.1f}x")
        components = case.get("candidate_components_ms")
        if not isinstance(components, dict):
            reasons.append(f"case {token_count} lacks component timings")
        else:
            required_components = (
                "d2h_pack", "d2h_dma", "d2h_cpu_scatter",
                "h2d_cpu_gather", "h2d_dma", "h2d_gpu_scatter",
            )
            for field in required_components:
                if not _finite_positive(components.get(field)):
                    reasons.append(
                        f"case {token_count} has invalid component {field}")
            component_groups = {
                "d2h": (
                    "d2h_pack", "d2h_dma", "d2h_cpu_scatter"),
                "h2d": (
                    "h2d_cpu_gather", "h2d_dma",
                    "h2d_gpu_scatter"),
            }
            for direction, fields in component_groups.items():
                values = [components.get(field) for field in fields]
                complete = case.get(
                    f"candidate_{direction}_median_ms")
                if (not all(_finite_positive(value) for value in values)
                        or not _finite_positive(complete)):
                    continue
                component_sum = sum(values)
                lower_bound = max(values) * 0.75
                upper_bound = component_sum * 1.5
                if not lower_bound <= complete <= upper_bound:
                    reasons.append(
                        f"case {token_count} {direction} complete time "
                        f"{complete:.3f} ms is inconsistent with components "
                        f"[{lower_bound:.3f}, {upper_bound:.3f}] ms")

    if report.get("extension_isolated_from_runtime") is not True:
        reasons.append("experimental extension is not isolated from runtime")
    return {
        "qualified": not reasons,
        "smoke_passed": None,
        "reasons": reasons,
    }


def _load_extension(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "corex_block_major_kv_transfer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _measure(operation: Callable[[], None],
             synchronize: Callable[[], None]) -> float:
    synchronize()
    started = time.perf_counter()
    operation()
    synchronize()
    return (time.perf_counter() - started) * 1000.0


def _median_trials(operation: Callable[[], None],
                   synchronize: Callable[[], None],
                   measured_cycles: int) -> tuple[float, list[float]]:
    operation()
    synchronize()
    values = [
        _measure(operation, synchronize) for _ in range(measured_cycles)
    ]
    return statistics.median(values), values


def _vendor_swap_function() -> tuple[str, Callable[..., None]]:
    import ixformer.functions as functions

    native = getattr(functions, "swap_blocks", None)
    if native is not None:
        return "ixformer.functions.swap_blocks", native
    legacy = getattr(functions, "vllm_swap_blocks", None)
    if legacy is not None:
        return "ixformer.functions.vllm_swap_blocks", legacy
    raise RuntimeError(
        "ixformer exposes neither swap_blocks nor vllm_swap_blocks")


def _case_mapping(torch: Any, num_blocks: int,
                  seed: int) -> tuple[Any, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    source = torch.randperm(
        num_blocks, generator=generator, dtype=torch.int64)
    destination = torch.randperm(
        num_blocks, generator=generator, dtype=torch.int64)
    return source, destination


def _build_candidate_chunks(torch: Any, source: Any, destination: Any,
                            device: Any) -> list[dict[str, Any]]:
    chunks = mapping_chunks(source.tolist(), destination.tolist())
    result = []
    for source_values, destination_values in chunks:
        source_cpu = torch.tensor(source_values, dtype=torch.int64)
        destination_cpu = torch.tensor(
            destination_values, dtype=torch.int64)
        result.append({
            "count": len(source_values),
            "source_cpu": source_cpu,
            "destination_cpu": destination_cpu,
            "source_gpu": source_cpu.to(device=device, dtype=torch.int32),
            "destination_gpu": destination_cpu.to(
                device=device, dtype=torch.int32),
        })
    return result


def _fill_gpu_cache(torch: Any, gpu_cache: list[Any]) -> bool:
    base = torch.arange(
        gpu_cache[0].numel(),
        dtype=torch.int32,
        device=gpu_cache[0].device,
    ).reshape(gpu_cache[0].shape)
    base.remainder_(1009)
    base_half = base.to(dtype=torch.float16)
    del base
    for layer, tensor in enumerate(gpu_cache):
        tensor.copy_(base_half)
        tensor.add_(layer * 37)
        tensor[1].add_(19)
    num_blocks = gpu_cache[0].shape[1]
    low = torch.tensor(
        [block % 256 for block in range(num_blocks)],
        dtype=torch.float16,
        device=gpu_cache[0].device,
    )
    high = torch.tensor(
        [block // 256 for block in range(num_blocks)],
        dtype=torch.float16,
        device=gpu_cache[0].device,
    )
    for layer, tensor in enumerate(gpu_cache):
        tensor[:, :, 0].copy_(low)
        tensor[:, :, 1].copy_(high)
        tensor[:, :, 2].fill_(layer)
        tensor[0, :, 3].zero_()
        tensor[1, :, 3].fill_(1)
    encoded = (
        gpu_cache[0][0, :, 0].to(dtype=torch.int32)
        + 256 * gpu_cache[0][0, :, 1].to(dtype=torch.int32))
    unique = encoded.equal(
        torch.arange(
            num_blocks, dtype=torch.int32, device=gpu_cache[0].device))
    del base_half
    return bool(unique)


def _full_d2h_exact(torch: Any, gpu_cache: list[Any],
                    baseline_cpu: list[Any], candidate_cpu: Any) -> bool:
    del torch, gpu_cache
    for layer in range(NUM_ATTENTION_LAYERS):
        candidate_layer = candidate_cpu[:, layer, :, :].permute(1, 0, 2)
        if not candidate_layer.equal(baseline_cpu[layer]):
            return False
    return True


def _full_h2d_exact(torch: Any, gpu_cache: list[Any],
                    baseline_cpu: list[Any], source: Any,
                    destination: Any) -> bool:
    source_gpu = source.to(
        device=gpu_cache[0].device, dtype=torch.int64)
    for layer in range(NUM_ATTENTION_LAYERS):
        actual = gpu_cache[layer].index_select(1, source_gpu).cpu()
        expected = baseline_cpu[layer].index_select(1, destination)
        if not actual.equal(expected):
            return False
        del actual, expected
    return True


def _run_case(torch: Any, extension: Any, device: Any, token_count: int,
              measured_cycles: int, seed: int,
              pipeline: str,
              vendor_name: str, vendor_swap: Callable[..., None],
              ) -> dict[str, Any]:
    num_blocks = blocks_for_tokens(token_count)
    cache_shape = (KV_PLANES, num_blocks, ELEMENTS_PER_PLANE_BLOCK)
    gpu_cache = [
        torch.empty(cache_shape, dtype=torch.float16, device=device)
        for _ in range(NUM_ATTENTION_LAYERS)
    ]
    unique_block_signatures = _fill_gpu_cache(torch, gpu_cache)
    if not unique_block_signatures:
        raise RuntimeError("block signatures are not unique")
    baseline_cpu = [
        torch.empty(
            cache_shape, dtype=torch.float16, device="cpu", pin_memory=True)
        for _ in range(NUM_ATTENTION_LAYERS)
    ]
    candidate_cpu = torch.empty(
        (num_blocks, NUM_ATTENTION_LAYERS, KV_PLANES,
         ELEMENTS_PER_PLANE_BLOCK),
        dtype=torch.float16,
        device="cpu",
        pin_memory=True,
    )
    buffer_count = staging_buffer_count(pipeline)
    cpu_staging = [
        torch.empty(
            (STAGING_BLOCKS, NUM_ATTENTION_LAYERS, KV_PLANES,
             ELEMENTS_PER_PLANE_BLOCK),
            dtype=torch.float16,
            device="cpu",
            pin_memory=True,
        )
        for _ in range(buffer_count)
    ]
    gpu_staging = [
        torch.empty_like(staging, device=device) for staging in cpu_staging
    ]
    transfer_events = [
        torch.cuda.Event(enable_timing=False) for _ in range(buffer_count)
    ]
    transfer_error = torch.zeros(1, dtype=torch.int32, device=device)
    if not all(tensor.is_pinned() for tensor in baseline_cpu):
        raise RuntimeError("baseline CPU cache is not pinned")
    if (not candidate_cpu.is_pinned()
            or not all(staging.is_pinned() for staging in cpu_staging)):
        raise RuntimeError("candidate CPU pool or staging is not pinned")

    source, destination = _case_mapping(torch, num_blocks, seed)
    d2h_mapping = torch.stack((source, destination), dim=1)
    h2d_mapping = torch.stack((destination, source), dim=1)
    d2h_legacy = dict(zip(source.tolist(), destination.tolist()))
    h2d_legacy = dict(zip(destination.tolist(), source.tolist()))
    d2h_chunks = _build_candidate_chunks(
        torch, source, destination, device)
    h2d_chunks = _build_candidate_chunks(
        torch, destination, source, device)
    synchronize = lambda: torch.cuda.synchronize(device)

    def clear_transfer_error() -> None:
        transfer_error.zero_()

    def check_transfer_error() -> None:
        extension.check_error(transfer_error)

    def baseline_swap_all(src: list[Any], dst: list[Any],
                          mapping: Any, legacy_mapping: dict[int, int]) -> None:
        for layer in range(NUM_ATTENTION_LAYERS):
            for kv_plane in range(KV_PLANES):
                if vendor_name.endswith("vllm_swap_blocks"):
                    vendor_swap(
                        src[layer][kv_plane], dst[layer][kv_plane],
                        legacy_mapping)
                else:
                    vendor_swap(
                        src[layer][kv_plane], dst[layer][kv_plane], mapping)

    def baseline_d2h() -> None:
        baseline_swap_all(
            gpu_cache, baseline_cpu, d2h_mapping, d2h_legacy)

    def baseline_h2d() -> None:
        baseline_swap_all(
            baseline_cpu, gpu_cache, h2d_mapping, h2d_legacy)

    def candidate_d2h() -> None:
        if pipeline == "single":
            clear_transfer_error()
            for chunk in d2h_chunks:
                count = chunk["count"]
                extension.pack(
                    gpu_cache, chunk["source_gpu"], gpu_staging[0],
                    transfer_error, count)
                cpu_staging[0][:count].copy_(
                    gpu_staging[0][:count], non_blocking=True)
                synchronize()
                extension.cpu_scatter(
                    cpu_staging[0], candidate_cpu,
                    chunk["destination_cpu"], count)
            check_transfer_error()
            return

        clear_transfer_error()
        pending: tuple[int, dict[str, Any]] | None = None
        for index, chunk in enumerate(d2h_chunks):
            slot = index % buffer_count
            count = chunk["count"]
            extension.pack(
                gpu_cache, chunk["source_gpu"], gpu_staging[slot],
                transfer_error, count)
            cpu_staging[slot][:count].copy_(
                gpu_staging[slot][:count], non_blocking=True)
            transfer_events[slot].record()
            if pending is not None:
                pending_slot, pending_chunk = pending
                transfer_events[pending_slot].synchronize()
                extension.cpu_scatter(
                    cpu_staging[pending_slot], candidate_cpu,
                    pending_chunk["destination_cpu"],
                    pending_chunk["count"])
            pending = (slot, chunk)
        if pending is not None:
            pending_slot, pending_chunk = pending
            transfer_events[pending_slot].synchronize()
            extension.cpu_scatter(
                cpu_staging[pending_slot], candidate_cpu,
                pending_chunk["destination_cpu"], pending_chunk["count"])
        check_transfer_error()

    def candidate_h2d() -> None:
        clear_transfer_error()
        for index, chunk in enumerate(h2d_chunks):
            slot = index % buffer_count
            count = chunk["count"]
            if pipeline == "double" and index >= buffer_count:
                transfer_events[slot].synchronize()
            extension.cpu_gather(
                candidate_cpu, chunk["source_cpu"],
                cpu_staging[slot], count)
            gpu_staging[slot][:count].copy_(
                cpu_staging[slot][:count], non_blocking=True)
            extension.scatter(
                gpu_staging[slot], chunk["destination_gpu"],
                gpu_cache, transfer_error, count)
            if pipeline == "double":
                transfer_events[slot].record()
            else:
                synchronize()
        synchronize()
        check_transfer_error()

    def pack_component() -> None:
        clear_transfer_error()
        for chunk in d2h_chunks:
            extension.pack(
                gpu_cache, chunk["source_gpu"],
                gpu_staging[0], transfer_error, chunk["count"])
            synchronize()
        check_transfer_error()

    def d2h_dma_component() -> None:
        for chunk in d2h_chunks:
            count = chunk["count"]
            cpu_staging[0][:count].copy_(
                gpu_staging[0][:count], non_blocking=True)
            synchronize()

    def cpu_scatter_component() -> None:
        for chunk in d2h_chunks:
            extension.cpu_scatter(
                cpu_staging[0], candidate_cpu,
                chunk["destination_cpu"], chunk["count"])

    def cpu_gather_component() -> None:
        for chunk in h2d_chunks:
            extension.cpu_gather(
                candidate_cpu, chunk["source_cpu"],
                cpu_staging[0], chunk["count"])

    def h2d_dma_component() -> None:
        for chunk in h2d_chunks:
            count = chunk["count"]
            gpu_staging[0][:count].copy_(
                cpu_staging[0][:count], non_blocking=True)
            synchronize()

    def scatter_component() -> None:
        clear_transfer_error()
        for chunk in h2d_chunks:
            extension.scatter(
                gpu_staging[0], chunk["destination_gpu"],
                gpu_cache, transfer_error, chunk["count"])
            synchronize()
        check_transfer_error()

    baseline_d2h_ms, baseline_d2h_trials = _median_trials(
        baseline_d2h, synchronize, measured_cycles)
    baseline_h2d_ms, baseline_h2d_trials = _median_trials(
        baseline_h2d, synchronize, measured_cycles)
    candidate_d2h_ms, candidate_d2h_trials = _median_trials(
        candidate_d2h, synchronize, measured_cycles)
    candidate_h2d_ms, candidate_h2d_trials = _median_trials(
        candidate_h2d, synchronize, measured_cycles)

    component_operations = {
        "d2h_pack": pack_component,
        "d2h_dma": d2h_dma_component,
        "d2h_cpu_scatter": cpu_scatter_component,
        "h2d_cpu_gather": cpu_gather_component,
        "h2d_dma": h2d_dma_component,
        "h2d_gpu_scatter": scatter_component,
    }
    component_ms = {
        name: _median_trials(
            operation, synchronize, measured_cycles)[0]
        for name, operation in component_operations.items()
    }

    baseline_d2h()
    candidate_d2h()
    synchronize()
    d2h_exact = _full_d2h_exact(
        torch, gpu_cache, baseline_cpu, candidate_cpu)

    for layer in gpu_cache:
        layer.fill_(-1)
    candidate_h2d()
    synchronize()
    h2d_exact = _full_h2d_exact(
        torch, gpu_cache, baseline_cpu, source, destination)

    victim_gpu = int(source[0])
    requested_gpu = int(source[1])
    preserved_cpu = int(destination[0])
    requested_cpu = int(destination[1])
    expected_victim = candidate_cpu[preserved_cpu].clone()
    expected_requested = candidate_cpu[requested_cpu].clone()
    same_slot_d2h = _build_candidate_chunks(
        torch,
        torch.tensor([victim_gpu], dtype=torch.int64),
        torch.tensor([preserved_cpu], dtype=torch.int64),
        device,
    )
    same_slot_h2d = _build_candidate_chunks(
        torch,
        torch.tensor([requested_cpu], dtype=torch.int64),
        torch.tensor([victim_gpu], dtype=torch.int64),
        device,
    )
    d2h_chunk = same_slot_d2h[0]
    clear_transfer_error()
    extension.pack(
        gpu_cache, d2h_chunk["source_gpu"], gpu_staging[0],
        transfer_error, 1)
    cpu_staging[0][:1].copy_(
        gpu_staging[0][:1], non_blocking=True)
    synchronize()
    check_transfer_error()
    extension.cpu_scatter(
        cpu_staging[0], candidate_cpu,
        d2h_chunk["destination_cpu"], 1)
    h2d_chunk = same_slot_h2d[0]
    extension.cpu_gather(
        candidate_cpu, h2d_chunk["source_cpu"], cpu_staging[0], 1)
    gpu_staging[0][:1].copy_(
        cpu_staging[0][:1], non_blocking=True)
    clear_transfer_error()
    extension.scatter(
        gpu_staging[0], h2d_chunk["destination_gpu"], gpu_cache,
        transfer_error, 1)
    synchronize()
    check_transfer_error()
    preserved_exact = candidate_cpu[preserved_cpu].equal(expected_victim)
    promoted = torch.stack([
        gpu_cache[layer][:, victim_gpu, :]
        for layer in range(NUM_ATTENTION_LAYERS)
    ]).cpu()
    requested_exact = promoted.equal(expected_requested)
    same_slot_exact = preserved_exact and requested_exact

    invalid_gpu_mapping_fail_fast = False
    invalid_ids = torch.tensor(
        [num_blocks, -1], dtype=torch.int32, device=device)
    clear_transfer_error()
    extension.pack(
        gpu_cache, invalid_ids, gpu_staging[0], transfer_error, 2)
    synchronize()
    try:
        check_transfer_error()
    except RuntimeError as exc:
        invalid_gpu_mapping_fail_fast = (
            "out-of-range id" in str(exc))

    byte_count = bytes_for_tokens(token_count)
    return {
        "token_count": token_count,
        "block_count": num_blocks,
        "bytes_per_direction": byte_count,
        "mapping_seed": seed,
        "pipeline": pipeline,
        "staging_buffer_count": buffer_count,
        "mapping_source_unique": source.unique().numel() == num_blocks,
        "mapping_destination_unique": (
            destination.unique().numel() == num_blocks),
        "chunk_blocks": STAGING_BLOCKS,
        "chunk_count": len(d2h_chunks),
        "final_chunk_blocks": d2h_chunks[-1]["count"],
        "baseline_d2h_median_ms": baseline_d2h_ms,
        "baseline_h2d_median_ms": baseline_h2d_ms,
        "candidate_d2h_median_ms": candidate_d2h_ms,
        "candidate_h2d_median_ms": candidate_h2d_ms,
        "d2h_speedup": baseline_d2h_ms / candidate_d2h_ms,
        "h2d_speedup": baseline_h2d_ms / candidate_h2d_ms,
        "baseline_d2h_trials_ms": baseline_d2h_trials,
        "baseline_h2d_trials_ms": baseline_h2d_trials,
        "candidate_d2h_trials_ms": candidate_d2h_trials,
        "candidate_h2d_trials_ms": candidate_h2d_trials,
        "candidate_components_ms": component_ms,
        "d2h_exact": d2h_exact,
        "h2d_exact": h2d_exact,
        "unique_block_signatures": unique_block_signatures,
        "invalid_gpu_mapping_fail_fast": invalid_gpu_mapping_fail_fast,
        "component_same_gpu_slot_order_exact": same_slot_exact,
        "same_slot_details": {
            "scope": "component_call_order_only",
            "victim_gpu_block": victim_gpu,
            "requested_source_gpu_block": requested_gpu,
            "preserved_cpu_slot": preserved_cpu,
            "requested_cpu_slot": requested_cpu,
            "preserved_victim_exact": preserved_exact,
            "promoted_request_exact": requested_exact,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure a fixed BI100 block-major pack/DMA/scatter data plane "
            "against the production layer-wise ixformer swap path."))
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "gate"), default="gate")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument(
        "--pipeline", choices=("single", "double"), default="single")
    args = parser.parse_args()

    import torch

    if not args.device.startswith("cuda:"):
        parser.error("--device must name one CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if not args.extension.is_file():
        raise FileNotFoundError(args.extension)

    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    extension = _load_extension(args.extension)
    vendor_name, vendor_swap = _vendor_swap_function()
    extension_sha256 = hashlib.sha256(
        args.extension.read_bytes()).hexdigest()
    token_counts = (
        BOUNDARY_TOKEN_COUNTS if args.mode == "smoke"
        else BOUNDARY_TOKEN_COUNTS + GATE_TOKEN_COUNTS)
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "M1-56-block-major-kv-data-plane",
        "mode": args.mode,
        "source_revision": args.source_revision,
        "instance": args.instance,
        "device": args.device,
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "vendor_swap": vendor_name,
        "extension_sha256": extension_sha256,
        "fixed_seed": FIXED_SEED,
        "attention_layers": NUM_ATTENTION_LAYERS,
        "kv_planes": KV_PLANES,
        "elements_per_plane_block": ELEMENTS_PER_PLANE_BLOCK,
        "block_size": BLOCK_SIZE,
        "staging_blocks": STAGING_BLOCKS,
        "pipeline": args.pipeline,
        "staging_buffer_count": staging_buffer_count(args.pipeline),
        "staging_bytes": (
            staging_buffer_count(args.pipeline) * STAGING_BLOCKS
            * bytes_per_block_per_rank()),
        "required_speedup": REQUIRED_SPEEDUP,
        "extension_isolated_from_runtime": True,
        "cases": {},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        for index, token_count in enumerate(token_counts):
            measured_cycles = (
                3 if token_count in GATE_TOKEN_COUNTS else 1)
            report["cases"][str(token_count)] = _run_case(
                torch=torch,
                extension=extension,
                device=device,
                token_count=token_count,
                measured_cycles=measured_cycles,
                seed=FIXED_SEED + index,
                pipeline=args.pipeline,
                vendor_name=vendor_name,
                vendor_swap=vendor_swap,
            )
        report["gate"] = evaluate_gate(report)
    except Exception as exc:
        report["fatal"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        report["gate"] = {
            "qualified": False,
            "reasons": [f"fatal {type(exc).__name__}: {exc}"],
        }
        args.out.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n")
        raise

    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "qualified": report["gate"]["qualified"],
        "reasons": report["gate"]["reasons"],
    }, sort_keys=True))
    return 0 if (
        report["gate"]["qualified"]
        or report["gate"].get("smoke_passed") is True
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
