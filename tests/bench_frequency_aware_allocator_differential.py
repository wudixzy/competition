#!/usr/bin/env python3
"""Compare M1-41 with a full-scan oracle in the real vLLM allocator."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

from vllm.core.block.block_table import BlockTable
from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.core.block.prefix_caching_block import PrefixCachingBlockAllocator
from vllm.core.evictor_v2 import BlockMetaData, Evictor
from vllm.utils import Device


class FullScanFrequencyOracle(Evictor):
    def __init__(self):
        self.free_table: dict[int, BlockMetaData] = {}
        self.frequency_by_hash: dict[bytes, int] = {}

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self) -> tuple[int, bytes]:
        if not self.free_table:
            raise ValueError("No usable cache memory left")
        block_id = min(self.free_table, key=lambda value: (
            self.frequency_by_hash[self.free_table[value].content_hash],
            self.free_table[value].last_accessed,
            -self.free_table[value].num_hashed_tokens,
            value,
        ))
        block = self.free_table.pop(block_id)
        return block_id, block.content_hash

    def add(self, block_id: int, content_hash: bytes,
            num_hashed_tokens: int, last_accessed: float):
        self.frequency_by_hash[content_hash] = (
            self.frequency_by_hash.get(content_hash, 0) + 1)
        self.free_table[block_id] = BlockMetaData(
            content_hash, num_hashed_tokens, last_accessed)

    def update(self, block_id: int, last_accessed: float):
        self.free_table[block_id].last_accessed = last_accessed

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        self.free_table.pop(block_id)

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


def load_candidate(path: Path):
    spec = importlib.util.spec_from_file_location(
        "m1_41_experimental_evictor", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load candidate evictor: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.FrequencyAwareEvictor


def generate_workload(seed: int, requests: int, max_tokens: int
                      ) -> list[tuple[list[int], list[int]]]:
    rng = random.Random(seed)
    histories: list[list[int]] = []
    workload = []
    next_token = 1000
    for _ in range(requests):
        reusable = [
            history for history in histories
            if len(history) <= max_tokens - 3
        ]
        if reusable and rng.random() < 0.72:
            base = list(rng.choice(reusable))
            extension = [
                next_token + offset
                for offset in range(rng.randint(1, 3))
            ]
            next_token += len(extension)
            prompt = (base + extension)[:max_tokens]
        else:
            length = rng.randint(1, min(9, max_tokens))
            prompt = [next_token + offset for offset in range(length)]
            next_token += length
        room = max_tokens - len(prompt)
        completion = [
            next_token + offset
            for offset in range(rng.randint(0, min(5, room)))
        ]
        next_token += len(completion)
        workload.append((prompt, completion))
        histories.append(prompt + completion)
        histories = histories[-16:]
    return workload


def make_allocator(capacity: int, block_size: int, policy: Evictor):
    allocator = CpuGpuBlockAllocator.create(
        allocator_type="prefix_caching",
        num_gpu_blocks=capacity,
        num_cpu_blocks=1,
        block_size=block_size,
    )
    gpu_allocator = allocator._allocators[Device.GPU]
    if not isinstance(gpu_allocator, PrefixCachingBlockAllocator):
        raise AssertionError("unexpected GPU allocator type")
    gpu_allocator.evictor = policy
    return allocator, gpu_allocator


def execute_request(allocator, prompt: list[int], completion: list[int],
                    timestamp: float, block_size: int) -> int:
    table = BlockTable(
        block_size=block_size,
        block_allocator=allocator,
    )
    table.allocate(prompt)
    strict_blocks = (len(prompt) - 1) // block_size
    hits = 0
    for block in table.blocks[:strict_blocks]:
        if not block.computed:
            break
        hits += 1
    allocator.mark_blocks_as_computed([])
    for token in completion:
        table.append_token_ids([token])
        allocator.mark_blocks_as_computed([])
    allocator.mark_blocks_as_accessed(table.physical_block_ids, timestamp)
    table.free()
    return hits


def run_case(candidate_type, seed: int, requests: int, capacity: int,
             block_size: int) -> dict:
    candidate = candidate_type()
    oracle = FullScanFrequencyOracle()
    candidate_allocator, candidate_gpu = make_allocator(
        capacity, block_size, candidate)
    oracle_allocator, oracle_gpu = make_allocator(
        capacity, block_size, oracle)
    candidate_hits = 0
    oracle_hits = 0
    workload = generate_workload(seed, requests, capacity * block_size)
    for index, (prompt, completion) in enumerate(workload, 1):
        candidate_hits += execute_request(
            candidate_allocator, prompt, completion,
            float(index), block_size)
        oracle_hits += execute_request(
            oracle_allocator, prompt, completion,
            float(index), block_size)
        if candidate_hits != oracle_hits:
            raise AssertionError(f"request {index}: hit count diverged")
        if candidate_gpu._cached_blocks != oracle_gpu._cached_blocks:
            raise AssertionError(f"request {index}: cache map diverged")
        if candidate.frequency_by_hash != oracle.frequency_by_hash:
            raise AssertionError(f"request {index}: frequency map diverged")
        if len(candidate._heap) > 2 * candidate.num_blocks + 1:
            raise AssertionError(f"request {index}: heap bound exceeded")
        if any(
            not isinstance(value, bytes) or len(value) != 32
            for value in candidate.frequency_by_hash
        ):
            raise AssertionError("allocator emitted a non-SHA content key")
    return {
        "seed": seed,
        "requests": requests,
        "capacity_blocks": capacity,
        "block_size": block_size,
        "hit_blocks": candidate_hits,
        "final_cached_blocks": len(candidate_gpu._cached_blocks),
        "frequency_entries": len(candidate.frequency_by_hash),
        "final_heap_entries": len(candidate._heap),
        "ok": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate_type = load_candidate(args.candidate)
    cases = [
        run_case(candidate_type, 20260721, 200, 12, 4),
        run_case(candidate_type, 7, 200, 7, 4),
        run_case(candidate_type, 99, 200, 16, 8),
    ]
    report = {
        "experiment": "M1-41-real-allocator-differential",
        "content_key": "sha256-bytes",
        "cases": cases,
        "requests": sum(case["requests"] for case in cases),
        "ok": all(case["ok"] for case in cases),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
