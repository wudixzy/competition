#!/usr/bin/env python3
"""Fixed CPU and memory gate for M1-41 content-frequency eviction."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "vllm" / "core" / "evictor_v2.py"
SPEC = importlib.util.spec_from_file_location(
    "m1_41_bench_evictor_v2", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load evictor module: {MODULE_PATH}")
evictor_v2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evictor_v2
SPEC.loader.exec_module(evictor_v2)
FrequencyAwareEvictor = evictor_v2.FrequencyAwareEvictor


REQUESTS = 881
MEAN_PROMPT_TOKENS = 25_654
MEAN_GENERATED_TOKENS = 429
BLOCK_SIZE = 16
CAPACITY_BLOCKS = 16_878
REPEATS = 3

# Frozen M1-29 LRU evidence from the identical request/block lifecycle. The
# bytes-key candidate is compared with this conservative integer-key baseline;
# re-running the O(N) LRU path would add about 256 seconds without changing the
# decision margin.
HISTORICAL_LRU_TOTAL_MS = 256_372.0
HISTORICAL_LRU_MEAN_REQUEST_MS = 291.001
HISTORICAL_LRU_P90_REQUEST_MS = 1_170.245


@dataclass(frozen=True)
class Bucket:
    count: int
    low: int
    high: int
    reusable_rate: float


BUCKETS = (
    Bucket(95, 256, 2_047, 0.33),
    Bucket(1, 2_048, 4_095, 0.01),
    Bucket(219, 4_096, 8_191, 0.89),
    Bucket(222, 8_192, 16_383, 0.58),
    Bucket(115, 16_384, 32_767, 0.60),
    Bucket(110, 32_768, 65_535, 0.66),
    Bucket(114, 65_536, 131_071, 0.69),
    Bucket(5, 131_072, 234_436, 0.23),
)


def _initial_lengths() -> tuple[list[int], list[int], list[int], list[float]]:
    values = []
    lows = []
    highs = []
    rates = []
    for bucket in BUCKETS:
        for index in range(bucket.count):
            fraction = (index + 0.5) / bucket.count
            values.append(
                bucket.low + int(fraction * (bucket.high - bucket.low)))
            lows.append(bucket.low)
            highs.append(bucket.high)
            rates.append(bucket.reusable_rate)
    if len(values) != REQUESTS:
        raise AssertionError("bucket counts must total 881")
    return values, lows, highs, rates


def _fit_total(values: list[int], lows: list[int], highs: list[int],
               target: int) -> list[int]:
    current = sum(values)
    if current == target:
        return values
    anchors = lows if current > target else highs
    available = sum(
        abs(value - anchor) for value, anchor in zip(values, anchors))
    needed = abs(current - target)
    if needed > available:
        raise ValueError("target prompt-token total is outside bucket bounds")
    keep = (available - needed) / available
    fitted = [
        anchor + int((value - anchor) * keep)
        for value, anchor in zip(values, anchors)
    ]
    residual = target - sum(fitted)
    step = 1 if residual > 0 else -1
    remaining = abs(residual)
    while remaining:
        progressed = False
        for index, value in enumerate(fitted):
            candidate = value + step
            if lows[index] <= candidate <= highs[index]:
                fitted[index] = candidate
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise AssertionError("cannot distribute fitted-token residual")
    return fitted


def workload() -> list[tuple[int, int]]:
    values, lows, highs, rates = _initial_lengths()
    values = _fit_total(
        values, lows, highs, REQUESTS * MEAN_PROMPT_TOKENS)
    requests = []
    for tokens, rate in zip(values, rates):
        blocks = math.ceil(tokens / BLOCK_SIZE)
        if blocks > CAPACITY_BLOCKS:
            raise AssertionError("a proxy request exceeds cache capacity")
        requests.append((blocks, round(blocks * rate)))
    if sum(values) != REQUESTS * MEAN_PROMPT_TOKENS:
        raise AssertionError("proxy mean prompt length drifted")
    return requests


def _content_key(value: int) -> bytes:
    return value.to_bytes(32, "big", signed=False)


def _content_hashes(request_index: int, blocks: int,
                    reusable_blocks: int):
    family = (request_index * 17) % 64
    for depth in range(blocks):
        if depth < reusable_blocks:
            value = (family << 32) | depth
        else:
            value = (1 << 63) | (request_index << 20) | depth
        yield _content_key(value)
    generated_blocks = math.ceil(MEAN_GENERATED_TOKENS / BLOCK_SIZE)
    for offset in range(generated_blocks):
        value = (
            (3 << 62) | (request_index << 20) | (blocks + offset))
        yield _content_key(value)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _cache_digest(content_to_block: dict[bytes, int]) -> str:
    digest = hashlib.sha256()
    for content_hash in sorted(content_to_block):
        digest.update(content_hash)
        digest.update(content_to_block[content_hash].to_bytes(
            4, "little", signed=False))
    return digest.hexdigest()


def simulate(requests: list[tuple[int, int]]) -> dict:
    policy = FrequencyAwareEvictor()
    free_ids = list(range(CAPACITY_BLOCKS - 1, -1, -1))
    content_to_block: dict[bytes, int] = {}
    block_to_content: dict[int, bytes] = {}
    hits = 0
    evictions = 0
    request_ms = []
    peak_heap_entries = 0
    for request_index, (blocks, reusable_blocks) in enumerate(requests):
        started = time.perf_counter_ns()
        active = []
        for depth, content_hash in enumerate(_content_hashes(
                request_index, blocks, reusable_blocks)):
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
                    raise AssertionError("evictor content mapping diverged")
                if block_to_content.pop(block_id) != evicted_hash:
                    raise AssertionError("evictor block mapping diverged")
            active.append((block_id, content_hash, depth))
        timestamp = float(request_index + 1)
        for block_id, content_hash, depth in active:
            policy.add(
                block_id, content_hash, (depth + 1) * BLOCK_SIZE, timestamp)
            content_to_block[content_hash] = block_id
            block_to_content[block_id] = content_hash
        request_ms.append(
            (time.perf_counter_ns() - started) / 1.0e6)
        peak_heap_entries = max(peak_heap_entries, len(policy._heap))
        if len(content_to_block) != policy.num_blocks:
            raise AssertionError("live cache size diverged")
        if len(policy._heap) > 2 * policy.num_blocks + 1:
            raise AssertionError("heap bound exceeded")
    return {
        "hits": hits,
        "evictions": evictions,
        "final_blocks": len(content_to_block),
        "final_cache_sha256": _cache_digest(content_to_block),
        "total_ms": sum(request_ms),
        "mean_request_ms": statistics.mean(request_ms),
        "p90_request_ms": _percentile(request_ms, 0.90),
        "max_request_ms": max(request_ms),
        "peak_heap_entries": peak_heap_entries,
        "final_heap_entries": len(policy._heap),
        "frequency_entries": len(policy.frequency_by_hash),
    }


def proxy_metadata() -> dict:
    return {
        "requests": REQUESTS,
        "prompt_tokens": REQUESTS * MEAN_PROMPT_TOKENS,
        "mean_prompt_tokens": MEAN_PROMPT_TOKENS,
        "generated_tokens_per_request": MEAN_GENERATED_TOKENS,
        "block_size": BLOCK_SIZE,
        "capacity_blocks": CAPACITY_BLOCKS,
        "content_key": "32-byte-logical-prefix",
        "qualification_trace": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--memory-child", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    requests = workload()
    if args.memory_child:
        run = simulate(requests)
        print(json.dumps({
            "proxy": proxy_metadata(),
            "run": run,
            "peak_rss_mib": resource.getrusage(
                resource.RUSAGE_SELF).ru_maxrss / 1024,
        }))
        return 0
    if args.output is None:
        parser.error("--output is required")

    runs = []
    for _ in range(REPEATS):
        gc.collect()
        runs.append(simulate(requests))
    memory_process = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--memory-child"],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    memory_report = json.loads(memory_process.stdout)
    memory_run = memory_report["run"]
    candidate_total = statistics.median(
        run["total_ms"] for run in runs)
    candidate_mean = statistics.median(
        run["mean_request_ms"] for run in runs)
    candidate_p90 = statistics.median(
        run["p90_request_ms"] for run in runs)
    deterministic = all(
        (run["hits"], run["evictions"], run["final_cache_sha256"])
        == (runs[0]["hits"], runs[0]["evictions"],
            runs[0]["final_cache_sha256"])
        for run in runs
    )
    heap_bound = 2 * CAPACITY_BLOCKS + 1024
    gates = {
        "deterministic": deterministic,
        "mean_added_cpu_at_most_3ms": (
            candidate_mean - HISTORICAL_LRU_MEAN_REQUEST_MS <= 3.0),
        "p90_added_cpu_at_most_10ms": (
            candidate_p90 - HISTORICAL_LRU_P90_REQUEST_MS <= 10.0),
        "process_peak_rss_at_most_256mib": (
            memory_report["peak_rss_mib"] <= 256.0),
        "heap_entries_bounded": (
            memory_run["peak_heap_entries"] <= heap_bound),
    }
    report = {
        "experiment": "M1-41-content-frequency-evictor-feasibility",
        "proxy": proxy_metadata(),
        "historical_lru": {
            "source": "M1_29_FREQUENCY_AWARE_EVICTOR_FEASIBILITY_20260718.md",
            "total_ms": HISTORICAL_LRU_TOTAL_MS,
            "mean_request_ms": HISTORICAL_LRU_MEAN_REQUEST_MS,
            "p90_request_ms": HISTORICAL_LRU_P90_REQUEST_MS,
        },
        "runs": runs,
        "memory_run": memory_run,
        "summary": {
            "candidate_total_median_ms": candidate_total,
            "candidate_mean_request_ms": candidate_mean,
            "candidate_p90_request_ms": candidate_p90,
            "mean_added_ms_per_request": (
                candidate_mean - HISTORICAL_LRU_MEAN_REQUEST_MS),
            "p90_added_ms_per_request": (
                candidate_p90 - HISTORICAL_LRU_P90_REQUEST_MS),
            "candidate_peak_process_rss_mib": (
                memory_report["peak_rss_mib"]),
            "heap_entry_bound": heap_bound,
            "deterministic": deterministic,
        },
        "gate": {
            "checks": gates,
            "continuation": all(gates.values()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["gate"]["continuation"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
