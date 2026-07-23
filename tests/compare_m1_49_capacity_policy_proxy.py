#!/usr/bin/env python3
"""Compare fixed cache policies at legacy and estimated M1-49 capacities.

This is a synthetic workload diagnostic. It cannot qualify a runtime policy
or replace the required complete same-session 881-request trace.
"""

from __future__ import annotations

import argparse
import heapq
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "tests/bench_frequency_aware_content_evictor.py"
SPEC = importlib.util.spec_from_file_location(
    "m1_49_capacity_proxy_workload", BENCH_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load fixed proxy workload: {BENCH_PATH}")
BENCH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCH
SPEC.loader.exec_module(BENCH)

LEGACY_CAPACITY_BLOCKS = 16_878
M1_49_ESTIMATED_CAPACITY_BLOCKS = 67_512
HISTORICAL_LRU_HITS = 1_429
HISTORICAL_LRU_EVICTIONS = 1_418_464


class HeapLruOracle:
    """Efficiently reproduce vLLM's LRU victim ordering for this proxy."""

    def __init__(self) -> None:
        self.free_table: dict[int, Any] = {}
        self.frequency_by_hash: dict[bytes, int] = {}
        self._heap: list[tuple[float, int, int, int]] = []
        self._generations: dict[int, int] = {}
        self._next_generation = 0

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)

    def _heap_key(self, block_id: int) -> tuple[float, int, int, int]:
        block = self.free_table[block_id]
        return (
            block.last_accessed,
            -block.num_hashed_tokens,
            block_id,
            self._generations[block_id],
        )

    def _compact_if_needed(self) -> None:
        if len(self._heap) <= 2 * len(self.free_table) + 1:
            return
        self._heap = [
            self._heap_key(block_id) for block_id in self.free_table
        ]
        heapq.heapify(self._heap)

    def _push(self, block_id: int) -> None:
        self._next_generation += 1
        self._generations[block_id] = self._next_generation
        heapq.heappush(self._heap, self._heap_key(block_id))
        self._compact_if_needed()

    def add(self, block_id: int, content_hash: bytes,
            num_hashed_tokens: int, last_accessed: float) -> None:
        self.free_table[block_id] = BENCH.evictor_v2.BlockMetaData(
            content_hash, num_hashed_tokens, last_accessed)
        self._push(block_id)

    def update(self, block_id: int, last_accessed: float) -> None:
        self.free_table[block_id].last_accessed = last_accessed
        self._push(block_id)

    def remove(self, block_id: int) -> None:
        if block_id not in self.free_table:
            raise ValueError("attempted to remove a non-resident LRU block")
        self.free_table.pop(block_id)
        self._generations.pop(block_id, None)
        self._compact_if_needed()

    def evict(self) -> tuple[int, bytes]:
        while self._heap:
            _, _, block_id, generation = heapq.heappop(self._heap)
            if self._generations.get(block_id) != generation:
                continue
            block = self.free_table.pop(block_id)
            self._generations.pop(block_id)
            self._compact_if_needed()
            return block_id, block.content_hash
        raise ValueError("no usable LRU cache block")


def _workload(capacity_blocks: int) -> list[tuple[int, int]]:
    previous = BENCH.CAPACITY_BLOCKS
    try:
        BENCH.CAPACITY_BLOCKS = capacity_blocks
        return BENCH.workload()
    finally:
        BENCH.CAPACITY_BLOCKS = previous


def simulate(policy_factory: Callable[[], Any],
             capacity_blocks: int) -> dict[str, Any]:
    requests = _workload(capacity_blocks)
    policy = policy_factory()
    free_ids = list(range(capacity_blocks - 1, -1, -1))
    content_to_block: dict[bytes, int] = {}
    block_to_content: dict[int, bytes] = {}
    hits = 0
    evictions = 0
    total_accesses = 0
    peak_heap_entries = 0

    for request_index, (blocks, reusable_blocks) in enumerate(requests):
        active = []
        for depth, content_hash in enumerate(BENCH._content_hashes(
                request_index, blocks, reusable_blocks)):
            total_accesses += 1
            block_id = content_to_block.pop(content_hash, None)
            if block_id is not None:
                hits += 1
                block_to_content.pop(block_id)
                policy.remove(block_id)
            elif free_ids:
                block_id = free_ids.pop()
            else:
                block_id, evicted_hash = policy.evict()
                evictions += 1
                if content_to_block.pop(evicted_hash) != block_id:
                    raise AssertionError("content mapping diverged")
                if block_to_content.pop(block_id) != evicted_hash:
                    raise AssertionError("block mapping diverged")
            active.append((block_id, content_hash, depth))

        timestamp = float(request_index + 1)
        for block_id, content_hash, depth in active:
            policy.add(
                block_id, content_hash, (depth + 1) * BENCH.BLOCK_SIZE,
                timestamp)
            content_to_block[content_hash] = block_id
            block_to_content[block_id] = content_hash

        peak_heap_entries = max(
            peak_heap_entries, len(getattr(policy, "_heap", ())))
        if len(content_to_block) != policy.num_blocks:
            raise AssertionError("live cache size diverged")
        if len(getattr(policy, "_heap", ())) > 2 * policy.num_blocks + 1:
            raise AssertionError("heap bound exceeded")

    if hits + evictions + len(content_to_block) != total_accesses:
        raise AssertionError("cache lifecycle accounting diverged")
    return {
        "hits": hits,
        "hit_rate": hits / total_accesses,
        "evictions": evictions,
        "final_blocks": len(content_to_block),
        "total_accesses": total_accesses,
        "final_cache_sha256": BENCH._cache_digest(content_to_block),
        "peak_heap_entries": peak_heap_entries,
        "final_heap_entries": len(getattr(policy, "_heap", ())),
        "frequency_entries": len(
            getattr(policy, "frequency_by_hash", {})),
    }


def _percentage_point_gain(candidate: dict[str, Any],
                           control: dict[str, Any]) -> float:
    if candidate["total_accesses"] != control["total_accesses"]:
        raise ValueError("policy proxy access counts differ")
    return 100.0 * (candidate["hit_rate"] - control["hit_rate"])


def build_report() -> dict[str, Any]:
    reports = {}
    for capacity in (
            LEGACY_CAPACITY_BLOCKS, M1_49_ESTIMATED_CAPACITY_BLOCKS):
        lru = simulate(HeapLruOracle, capacity)
        frequency = simulate(BENCH.FrequencyAwareEvictor, capacity)
        reports[str(capacity)] = {
            "lru": lru,
            "frequency": frequency,
            "frequency_hit_gain_percentage_points": (
                _percentage_point_gain(frequency, lru)),
        }

    legacy_lru = reports[str(LEGACY_CAPACITY_BLOCKS)]["lru"]
    if (
        legacy_lru["hits"] != HISTORICAL_LRU_HITS
        or legacy_lru["evictions"] != HISTORICAL_LRU_EVICTIONS
    ):
        raise AssertionError("heap LRU oracle does not match frozen evidence")

    candidate = reports[str(M1_49_ESTIMATED_CAPACITY_BLOCKS)]
    return {
        "schema": "bi100-m1-49-capacity-policy-proxy-v1",
        "version": 1,
        "scope": "synthetic-capacity-direction-only",
        "qualification_trace": False,
        "promotion_authorized": False,
        "proxy": {
            "requests": BENCH.REQUESTS,
            "prompt_tokens": BENCH.REQUESTS * BENCH.MEAN_PROMPT_TOKENS,
            "block_size": BENCH.BLOCK_SIZE,
            "legacy_capacity_blocks": LEGACY_CAPACITY_BLOCKS,
            "candidate_capacity_blocks_estimate": (
                M1_49_ESTIMATED_CAPACITY_BLOCKS),
            "candidate_capacity_is_measured": False,
        },
        "validation": {
            "legacy_lru_counts_match_frozen_evidence": True,
            "policy_access_counts_match": all(
                row["lru"]["total_accesses"]
                == row["frequency"]["total_accesses"]
                for row in reports.values()),
        },
        "reports": reports,
        "diagnostic": {
            "candidate_lru_capacity_hit_gain_percentage_points": (
                _percentage_point_gain(
                    candidate["lru"],
                    reports[str(LEGACY_CAPACITY_BLOCKS)]["lru"],
                )),
            "candidate_frequency_hit_gain_over_lru_percentage_points": (
                candidate["frequency_hit_gain_percentage_points"]),
            "candidate_frequency_extra_hit_blocks": (
                candidate["frequency"]["hits"] - candidate["lru"]["hits"]),
            "candidate_frequency_fewer_evictions": (
                candidate["lru"]["evictions"]
                - candidate["frequency"]["evictions"]),
            "capacity_pressure_remains": (
                candidate["frequency"]["evictions"] > 0),
            "frequency_direction_remains_plausible": (
                candidate["frequency_hit_gain_percentage_points"] >= 5.0),
        },
        "limitations": [
            "capacity is an estimate until the M1-49 TP4 startup gate runs",
            "proxy block identities are synthetic bucket-shaped content",
            "GDN state availability is not modeled",
            "no TTFT, throughput, score, or promotion claim is permitted",
            "the complete same-session privacy-safe 881 trace remains required",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
