"""Fixed block-major CPU KV transfer path for BI100/CoreX."""

from __future__ import annotations

import importlib
import os
from typing import Any, Iterable, Mapping, Sequence, Tuple


TRANSFER_LAYOUT_ENV = "BI100_CPU_KV_TRANSFER_LAYOUT"
PAGED_LAYOUT = "paged"
BLOCK_MAJOR_LAYOUT = "block_major"
STAGING_BLOCKS = 512
EXPECTED_LAYERS = 10
EXPECTED_BLOCK_PAYLOAD_ELEMENTS = 16 * 1 * 256


def transfer_layout_from_env(
    environ: Mapping[str, str] | None = None,
) -> str:
    source = os.environ if environ is None else environ
    value = source.get(TRANSFER_LAYOUT_ENV, PAGED_LAYOUT)
    if value not in (PAGED_LAYOUT, BLOCK_MAJOR_LAYOUT):
        raise RuntimeError(
            f"{TRANSFER_LAYOUT_ENV} must be 'paged' or 'block_major', "
            f"got {value!r}")
    return value


def chunk_ranges(count: int, chunk_size: int = STAGING_BLOCKS
                 ) -> Tuple[Tuple[int, int], ...]:
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("count must be a non-negative integer")
    if (not isinstance(chunk_size, int) or isinstance(chunk_size, bool)
            or chunk_size <= 0):
        raise ValueError("chunk_size must be a positive integer")
    return tuple(
        (start, min(start + chunk_size, count))
        for start in range(0, count, chunk_size)
    )


def validate_mapping_pairs(
    pairs: Iterable[Sequence[int]],
    source_limit: int,
    destination_limit: int,
) -> Tuple[Tuple[int, int], ...]:
    if source_limit <= 0 or destination_limit <= 0:
        raise ValueError("mapping limits must be positive")
    normalized = []
    sources = set()
    destinations = set()
    for index, pair in enumerate(pairs):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"mapping row {index} must contain two integers")
        source, destination = pair
        if (not isinstance(source, int) or isinstance(source, bool)
                or not isinstance(destination, int)
                or isinstance(destination, bool)):
            raise TypeError(f"mapping row {index} must contain integers")
        if source < 0 or source >= source_limit:
            raise ValueError(
                f"mapping source {source} is outside [0, {source_limit})")
        if destination < 0 or destination >= destination_limit:
            raise ValueError(
                "mapping destination "
                f"{destination} is outside [0, {destination_limit})")
        if source in sources:
            raise ValueError(f"duplicate mapping source {source}")
        if destination in destinations:
            raise ValueError(f"duplicate mapping destination {destination}")
        sources.add(source)
        destinations.add(destination)
        normalized.append((source, destination))
    return tuple(normalized)


class Bi100BlockMajorKvTransfer:
    """Pack/scatter layer-major GPU KV through one block-major CPU tier."""

    def __init__(self, gpu_cache: Sequence[Any], num_cpu_blocks: int) -> None:
        import torch

        if not isinstance(num_cpu_blocks, int) or num_cpu_blocks <= 0:
            raise ValueError("num_cpu_blocks must be positive")
        if len(gpu_cache) != EXPECTED_LAYERS:
            raise RuntimeError(
                f"block-major transfer requires {EXPECTED_LAYERS} attention "
                f"layers, got {len(gpu_cache)}")
        first = gpu_cache[0]
        if (not first.is_cuda or first.dtype != torch.float16
                or not first.is_contiguous() or first.dim() < 3
                or first.shape[0] != 2 or first.shape[1] <= 0):
            raise RuntimeError(
                "block-major transfer requires contiguous CUDA FP16 cache "
                "with shape [2, blocks, ...]")
        device = first.device
        num_gpu_blocks = int(first.shape[1])
        payload_elements = first[0, 0].numel()
        if payload_elements != EXPECTED_BLOCK_PAYLOAD_ELEMENTS:
            raise RuntimeError(
                "block-major transfer requires block_size=16, one local KV "
                f"head, and head_size=256; payload={payload_elements}")
        for layer_index, layer_cache in enumerate(gpu_cache):
            if (layer_cache.device != device or layer_cache.dtype != first.dtype
                    or not layer_cache.is_contiguous()
                    or tuple(layer_cache.shape) != tuple(first.shape)):
                raise RuntimeError(
                    f"GPU KV layer {layer_index} does not match layer 0")

        try:
            extension = importlib.import_module(
                "vllm.corex_block_major_kv_transfer")
        except Exception as exc:
            raise RuntimeError(
                "block-major transfer extension is unavailable") from exc
        if not callable(getattr(extension, "pack", None)):
            raise RuntimeError("block-major extension has no pack function")
        if not callable(getattr(extension, "scatter", None)):
            raise RuntimeError("block-major extension has no scatter function")

        self._torch = torch
        self._extension = extension
        self.gpu_cache = list(gpu_cache)
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks
        self.num_layers = len(gpu_cache)
        self.payload_elements = payload_elements
        self.device = device
        self.chunk_blocks = min(STAGING_BLOCKS, num_cpu_blocks,
                                num_gpu_blocks)
        stage_shape = (
            self.chunk_blocks,
            self.num_layers,
            2,
            self.payload_elements,
        )
        self.cpu_cache = torch.zeros(
            (num_cpu_blocks, self.num_layers, 2, self.payload_elements),
            dtype=first.dtype,
            device="cpu",
            pin_memory=True,
        )
        self.cpu_staging = torch.empty(
            stage_shape,
            dtype=first.dtype,
            device="cpu",
            pin_memory=True,
        )
        self.gpu_staging = torch.empty(
            stage_shape,
            dtype=first.dtype,
            device=device,
        )
        if not self.cpu_cache.is_pinned() or not self.cpu_staging.is_pinned():
            raise RuntimeError("block-major CPU KV tensors must be pinned")

    def _mapping_pairs(self, mapping: Any, *, swap_in: bool
                       ) -> Tuple[Tuple[int, int], ...]:
        torch = self._torch
        if not isinstance(mapping, torch.Tensor):
            raise TypeError("block-major mapping must be a torch.Tensor")
        if mapping.device.type != "cpu" or mapping.dtype != torch.int64:
            raise ValueError("block-major mapping must be a CPU int64 tensor")
        if mapping.dim() != 2 or mapping.shape[1] != 2:
            raise ValueError("block-major mapping must have shape [N, 2]")
        if not mapping.is_contiguous():
            raise ValueError("block-major mapping must be contiguous")
        source_limit = self.num_cpu_blocks if swap_in else self.num_gpu_blocks
        destination_limit = (
            self.num_gpu_blocks if swap_in else self.num_cpu_blocks)
        return validate_mapping_pairs(
            mapping.tolist(), source_limit, destination_limit)

    def swap_out(self, mapping: Any) -> None:
        pairs = self._mapping_pairs(mapping, swap_in=False)
        torch = self._torch
        for start, end in chunk_ranges(len(pairs), self.chunk_blocks):
            count = end - start
            chunk = pairs[start:end]
            gpu_ids_cpu = torch.tensor(
                [pair[0] for pair in chunk], dtype=torch.int64)
            cpu_ids = torch.tensor(
                [pair[1] for pair in chunk], dtype=torch.int64)
            gpu_ids = gpu_ids_cpu.to(self.device)
            gpu_stage = self.gpu_staging[:count]
            cpu_stage = self.cpu_staging[:count]
            self._extension.pack(self.gpu_cache, gpu_ids, gpu_stage)
            cpu_stage.copy_(gpu_stage, non_blocking=True)
            torch.cuda.current_stream(self.device).synchronize()
            self.cpu_cache.index_copy_(0, cpu_ids, cpu_stage)

    def swap_in(self, mapping: Any) -> None:
        pairs = self._mapping_pairs(mapping, swap_in=True)
        torch = self._torch
        for start, end in chunk_ranges(len(pairs), self.chunk_blocks):
            count = end - start
            chunk = pairs[start:end]
            cpu_ids = torch.tensor(
                [pair[0] for pair in chunk], dtype=torch.int64)
            gpu_ids_cpu = torch.tensor(
                [pair[1] for pair in chunk], dtype=torch.int64)
            cpu_stage = self.cpu_staging[:count]
            gpu_stage = self.gpu_staging[:count]
            torch.index_select(self.cpu_cache, 0, cpu_ids, out=cpu_stage)
            gpu_stage.copy_(cpu_stage, non_blocking=True)
            gpu_ids = gpu_ids_cpu.to(self.device)
            self._extension.scatter(gpu_stage, self.gpu_cache, gpu_ids)
            # One staging buffer is reused across deterministic chunks. Wait
            # until this chunk no longer reads it before the next CPU gather.
            torch.cuda.current_stream(self.device).synchronize()
