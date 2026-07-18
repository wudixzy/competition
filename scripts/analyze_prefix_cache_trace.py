#!/usr/bin/env python3
"""Offline simulator for BI100 prefix-cache traces."""

import argparse
import base64
import hashlib
import json
import math
import os
import re
import string
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any, Dict, Tuple

MARKER = "[BI100_CACHE_TRACE] "
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

try:
    from qwen3_6_scripts.gdn_prefix import (  # type: ignore
        GdnPrefixStatePolicy,
        capture_points_for_step,
        final_capture_key,
        key_at_strict_boundary,
        keys_from_block_hashes,
        strict_prefix_block_count,
    )
except (ModuleNotFoundError, ImportError, OSError):
    # Fallback implementations for environments where the script is imported
    # directly and qwen3_6_scripts is unavailable.
    from collections import OrderedDict

    GdnPrefixKey = Tuple[int, bytes]

    def strict_prefix_block_count(token_count: int, block_size: int) -> int:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if token_count <= 1:
            return 0
        return (token_count - 1) // block_size

    def key_at_strict_boundary(block_hashes: Sequence[bytes], token_count: int,
                               block_size: int) -> GdnPrefixKey | None:
        block_count = min(len(block_hashes),
                          strict_prefix_block_count(token_count, block_size))
        if block_count <= 0:
            return None
        return (block_count, block_hashes[block_count - 1])

    def final_capture_key(block_hashes: Sequence[bytes], prompt_tokens: int,
                         block_size: int, restore_mode: str,
                         replay_alignment: int) -> GdnPrefixKey | None:
        if restore_mode != "direct":
            raise ValueError(f"unknown GDN restore mode: {restore_mode}")
        return key_at_strict_boundary(block_hashes, prompt_tokens, block_size)

    def capture_points_for_step(targets: Iterable[GdnPrefixKey],
                               physical_context_tokens: int,
                               logical_end_tokens: int,
                               block_size: int) -> Tuple[Tuple[int, GdnPrefixKey], ...]:
        if physical_context_tokens < 0 or logical_end_tokens < 0:
            raise ValueError("token positions must be non-negative")
        selected = {}
        for key in targets:
            boundary_tokens = key[0] * block_size
            if physical_context_tokens < boundary_tokens <= logical_end_tokens:
                selected[boundary_tokens - physical_context_tokens] = key
        points = tuple(sorted(selected.items()))
        if len(points) > 2:
            raise ValueError("at most two GDN capture points are allowed per step")
        return points

    def keys_from_block_hashes(block_hashes: Sequence[bytes]) -> list[GdnPrefixKey]:
        return [
            (idx + 1, digest) for idx, digest in enumerate(block_hashes)
            if len(digest) == 32
        ]

    def _validate_digest(digest: bytes) -> None:
        if not isinstance(digest, (bytes, bytearray)) or len(digest) != 32:
            raise ValueError("GDN prefix digest must be exactly 32 bytes")

    class GdnPrefixStatePolicy:
        def __init__(self, policy: str) -> None:
            if policy not in {"fine32", "admission64", "off"}:
                raise ValueError(f"unknown GDN cache policy: {policy}")
            self.policy = policy
            self.capacity = {"fine32": 32, "admission64": 64, "off": 0}[policy]
            self._resident: OrderedDict[GdnPrefixKey, None] = OrderedDict()

        def __len__(self) -> int:
            return len(self._resident)

        def select_restore(self, live_prefix_keys: Sequence[GdnPrefixKey],
                          max_blocks: int) -> GdnPrefixKey | None:
            if self.capacity == 0 or max_blocks <= 0:
                return None
            best = None
            for key in live_prefix_keys[:max_blocks]:
                if key in self._resident:
                    best = key
            if best is not None:
                self._resident.move_to_end(best)
            return best

        def repeated_branch_candidate(self, live_prefix_keys: Sequence[GdnPrefixKey],
                                    max_blocks: int) -> GdnPrefixKey | None:
            if self.policy != "admission64" or max_blocks <= 0:
                return None
            candidate = live_prefix_keys[min(len(live_prefix_keys), max_blocks) - 1]
            if candidate in self._resident:
                return None
            return candidate

        def admit(self, keys: Iterable[GdnPrefixKey]) -> Tuple[GdnPrefixKey, ...]:
            if self.capacity == 0:
                return ()
            evicted: list[GdnPrefixKey] = []
            for key in keys:
                _validate_digest(key[1])
                if key in self._resident:
                    self._resident.move_to_end(key)
                else:
                    self._resident[key] = None
                while len(self._resident) > self.capacity:
                    evicted_key, _ = self._resident.popitem(last=False)
                    evicted.append(evicted_key)
            return tuple(evicted)


def _trace_payload(line: str, path: str) -> Dict[str, Any] | None:
    if MARKER not in line:
        return None
    if line.lstrip().startswith("{"):
        try:
            wrapped = json.loads(line)
        except json.JSONDecodeError:
            wrapped = None
        if isinstance(wrapped, dict) and isinstance(wrapped.get("log"), str):
            line = wrapped["log"]
    if MARKER not in line:
        return None
    payload = ANSI_ESCAPE.sub("", line.split(MARKER, 1)[1]).strip()
    try:
        record, end = json.JSONDecoder().raw_decode(payload)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid trace JSON in {path}") from exc
    if payload[end:].strip():
        raise ValueError(f"unexpected content after trace JSON in {path}")
    if not isinstance(record, dict):
        raise ValueError(f"trace record in {path} must be an object")
    return record


def _hex16(value: Any, field: str) -> str:
    if (not isinstance(value, str) or len(value) != 16 or
            any(char not in string.hexdigits for char in value)):
        raise ValueError(f"{field} must be 16 hex characters")
    return value.lower()


def _file_provenance(path: str) -> Dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"path": os.fspath(path), "bytes": size, "sha256": digest.hexdigest()}


def _baseline_metrics(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("baseline metrics must be a JSON object")
    run_id = data.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("baseline metrics run_id must be non-empty")
    trace_session = _hex16(data.get("trace_session_sha256"), "baseline trace_session_sha256")
    cache_tps = data.get("cache_tps")
    weighted_score = data.get("weighted_score")
    for name, value in (("cache_tps", cache_tps), ("weighted_score", weighted_score)):
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value)):
            raise ValueError(f"baseline metrics {name} must be finite")
    if cache_tps < 0:
        raise ValueError("baseline Cache TPS must be non-negative")
    if weighted_score <= 0:
        raise ValueError("baseline weighted score must be positive")
    return {
        "run_id": run_id.strip(),
        "trace_session_sha256": trace_session,
        "cache_tps": float(cache_tps),
        "weighted_score": float(weighted_score),
        "source": _file_provenance(path),
    }


def _decode_block_hashes(record: Dict[str, Any], path: str) -> None:
    raw = record.get("block_hashes")
    full_blocks = record["full_blocks"]
    try:
        raw_bytes = base64.b64decode(raw, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid base64 block_hashes") from exc
    if not isinstance(raw, str):
        raise ValueError("block_hashes must be a base64 string")
    expected = full_blocks * 32
    if len(raw_bytes) != expected:
        raise ValueError("block_hashes length does not match full_blocks")
    record["_hashes"] = [
        raw_bytes[i:i + 32]
        for i in range(0, len(raw_bytes), 32)
    ]


def read(paths: Sequence[str]) -> list[Dict[str, Any]]:
    records = []
    request_ids = set()
    for path in paths:
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                record = _trace_payload(line, path)
                if record is None:
                    continue
                if (record.get("version") != 4
                        or record.get("hash_encoding") != "sha256_base64"):
                    raise ValueError("unsupported trace version or encoding")
                record["trace_session_sha256"] = _hex16(
                    record.get("trace_session_sha256"),
                    "trace_session_sha256",
                )
                ordinal = record.get("ordinal")
                if (not isinstance(ordinal, int) or isinstance(ordinal, bool)
                        or ordinal <= 0):
                    raise ValueError("ordinal must be a positive integer")
                request_id = _hex16(record.get("request_id_sha256"), "request_id_sha256")
                if request_id in request_ids:
                    raise ValueError("duplicate request_id_sha256 in trace")
                request_ids.add(request_id)
                block_size = record.get("block_size")
                capacity = record.get("capacity_blocks")
                prompt_allocated = record.get("prompt_allocated_blocks")
                allocated = record.get("allocated_blocks")
                full_blocks = record.get("full_blocks")
                prompt_tokens = record.get("prompt_tokens")
                total_tokens = record.get("total_tokens")

                if not isinstance(block_size, int) or block_size <= 0:
                    raise ValueError("block_size must be positive")
                if not isinstance(capacity, int) or capacity <= 0:
                    raise ValueError("capacity_blocks must be positive")
                if (not isinstance(prompt_allocated, int)
                        or prompt_allocated <= 0):
                    raise ValueError("prompt_allocated_blocks must be positive")
                if not isinstance(allocated, int) or allocated <= 0:
                    raise ValueError("allocated_blocks must be positive")
                if not isinstance(full_blocks, int) or full_blocks < 0:
                    raise ValueError("full_blocks must be non-negative")
                if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
                    raise ValueError("prompt_tokens must be positive")
                if not isinstance(total_tokens, int) or total_tokens < prompt_tokens:
                    raise ValueError("total_tokens must be at least prompt_tokens")

                expected_prompt_allocated = (
                    prompt_tokens + block_size - 1) // block_size
                expected_allocated = (total_tokens + block_size - 1) // block_size
                if prompt_allocated != expected_prompt_allocated:
                    raise ValueError("prompt_allocated_blocks is inconsistent with prompt_tokens")
                if allocated != expected_allocated:
                    raise ValueError("allocated_blocks is inconsistent with total_tokens")
                if allocated > capacity:
                    raise ValueError("request has more blocks than capacity")
                if full_blocks != total_tokens // block_size:
                    raise ValueError("full_blocks is inconsistent with total_tokens")

                prompt_full_blocks = prompt_tokens // block_size
                if prompt_full_blocks > full_blocks:
                    raise ValueError("final block hashes do not cover the prompt")

                _decode_block_hashes(record, path)
                if len(record["_hashes"]) != full_blocks:
                    raise ValueError("block_hashes length does not match full_blocks")

                record["_prompt_full_blocks"] = prompt_full_blocks
                records.append(record)

    if not records:
        raise ValueError("no BI100_CACHE_TRACE records found")

    sessions = {record["trace_session_sha256"] for record in records}
    if len(sessions) != 1:
        raise ValueError("trace contains multiple runtime sessions")

    ordinals = [record["ordinal"] for record in records]
    if ordinals != list(range(1, len(records) + 1)):
        raise ValueError("trace ordinals must be contiguous and ordered from 1")

    return records


def _evict(cache, last: Dict[bytes, Tuple[int, int]], frequency: Dict[bytes, int],
          candidate: bool) -> None:
    if not cache:
        raise ValueError("request cannot fit in cache")
    if candidate:
        victim = min(cache, key=lambda block: (
            frequency[block], last.get(block, (-1, -1))[0],
            -last.get(block, (-1, -1))[1], block))
    else:
        victim = min(cache, key=lambda block: (
            last.get(block, (-1, -1))[0],
            -last.get(block, (-1, -1))[1], block))
    cache.remove(victim)
    last.pop(victim, None)


def _simulate(records: Sequence[Dict[str, Any]], capacity: int,
              candidate: bool, policy: str,
              gdn_chunk_tokens: int = 8192) -> Dict[str, Any]:
    if policy not in {"off", "fine32", "admission64"}:
        raise ValueError("policy must be one of off, fine32, admission64")
    if gdn_chunk_tokens <= 0:
        raise ValueError("gdn_chunk_tokens must be positive")
    cache = set()
    last = {}
    frequency = defaultdict(int)
    hits = total = hit_tokens = gap_blocks = 0
    avoided_tokens = 0
    gdn_policy = GdnPrefixStatePolicy(policy)

    for tick, record in enumerate(records, 1):
        hashes = record["_hashes"]
        block_size = record["block_size"]
        prompt_tokens = record["prompt_tokens"]
        strict_blocks = strict_prefix_block_count(prompt_tokens, block_size)
        prompt_full_blocks = record["_prompt_full_blocks"]
        prompt_hashes = hashes[:prompt_full_blocks]

        hit = 0
        saw_gap = False
        active_blocks = 0

        def allocate_slot() -> None:
            nonlocal active_blocks
            if len(cache) + active_blocks >= capacity:
                _evict(cache, last, frequency, candidate)
            active_blocks += 1

        for depth, block in enumerate(prompt_hashes):
            was_cached = block in cache
            if was_cached:
                cache.remove(block)
                active_blocks += 1
            else:
                allocate_slot()
            frequency[block] += 1
            if depth >= strict_blocks:
                continue
            if not saw_gap and was_cached:
                hit += 1
            else:
                saw_gap = True
                if was_cached:
                    gap_blocks += 1

        hits += hit
        total += strict_blocks
        hit_tokens += hit * block_size

        # A recurrent state is usable only while the exact contiguous KV
        # prefix is also resident. State-only hits cannot restore attention KV.
        live_keys = keys_from_block_hashes(prompt_hashes[:hit])
        restore_key = gdn_policy.select_restore(live_keys, len(live_keys))
        if restore_key is not None:
            avoided_tokens += restore_key[0] * block_size

        capture_targets: list[Tuple[int, bytes]] = []
        if policy == "fine32":
            restore_tokens = (restore_key[0] * block_size
                              if restore_key is not None else 0)
            logical_end = min(prompt_tokens,
                              restore_tokens + gdn_chunk_tokens)
            while logical_end > restore_tokens:
                step_key = key_at_strict_boundary(
                    hashes, logical_end, block_size)
                if step_key is not None:
                    capture_targets.append(step_key)
                if logical_end >= prompt_tokens:
                    break
                logical_end = min(prompt_tokens,
                                  logical_end + gdn_chunk_tokens)
        elif policy == "admission64":
            branch_key = gdn_policy.repeated_branch_candidate(
                live_keys, len(live_keys))
            if branch_key is not None:
                capture_targets.append(branch_key)
            final_key = final_capture_key(
                hashes, prompt_tokens, block_size, "direct", gdn_chunk_tokens)
            if final_key is not None:
                capture_targets.append(final_key)
        gdn_policy.admit(dict.fromkeys(capture_targets))

        while active_blocks < record["prompt_allocated_blocks"]:
            allocate_slot()
        if active_blocks != record["prompt_allocated_blocks"]:
            raise ValueError("prompt allocation replay diverged")

        mutable_tail = prompt_tokens % block_size != 0
        for block in hashes[len(prompt_hashes):]:
            if not mutable_tail:
                allocate_slot()
                mutable_tail = True
            frequency[block] += 1
            if block in cache:
                cache.remove(block)
            mutable_tail = False

        if record["total_tokens"] % block_size != 0:
            if not mutable_tail:
                allocate_slot()
                mutable_tail = True
        elif mutable_tail:
            raise ValueError("final mutable-tail replay diverged")
        if active_blocks != record["allocated_blocks"]:
            raise ValueError("final allocation replay diverged")

        cache.update(hashes)
        if len(cache) > capacity:
            raise ValueError("free replay exceeded cache capacity")
        for depth, block in enumerate(hashes):
            last[block] = (tick, depth)

    total_prompt_tokens = sum(record["prompt_tokens"] for record in records)
    return {
        "policy": policy,
        "hit_blocks": hits,
        "total_blocks": total,
        "hit_tokens": hit_tokens,
        "gap_blocks": gap_blocks,
        "raw_kv_contiguous_hit_tokens": hit_tokens,
        "raw_kv_contiguous_hit_block_rate": hits / total if total else 0.0,
        "raw_kv_hit_token_rate": (hit_tokens / total_prompt_tokens
                                   if total_prompt_tokens else 0.0),
        "usable_gdn_state_avoided_tokens": avoided_tokens,
        "usable_gdn_state_avoided_token_rate": (avoided_tokens /
                                                total_prompt_tokens
                                                if total_prompt_tokens else 0.0),
        # Compatibility aliases now mean the actually usable intersection,
        # rather than double-counting raw KV and recurrent-state hits.
        "combined_hit_tokens": avoided_tokens,
        "combined_hit_token_rate": (avoided_tokens / total_prompt_tokens
                                    if total_prompt_tokens else 0.0),
        "final_cache": cache,
        "gdn_policy_cache_size": len(gdn_policy),
    }


def simulate(records: Sequence[Dict[str, Any]], capacity: int,
             candidate: bool = False, policy: str = "off",
             gdn_chunk_tokens: int = 8192) -> Dict[str, Any]:
    if isinstance(candidate, str) and policy == "off":
        policy = candidate
        candidate = False
    result = _simulate(records, capacity, candidate, policy,
                       gdn_chunk_tokens)
    del result["final_cache"]
    return result


def _metrics(raw: Dict[str, Any], total_prompt_tokens: int) -> Dict[str, Any]:
    return {
        "policy": raw["policy"],
        "hit_tokens": raw["hit_tokens"],
        "hit_blocks": raw["hit_blocks"],
        "total_blocks": raw["total_blocks"],
        "gap_blocks": raw["gap_blocks"],
        "raw_kv_contiguous_hit_tokens": raw["raw_kv_contiguous_hit_tokens"],
        "raw_kv_contiguous_hit_blocks": raw["hit_blocks"],
        "raw_kv_contiguous_hit_block_rate": raw["raw_kv_contiguous_hit_block_rate"],
        "raw_kv_contiguous_hit_token_rate": raw["raw_kv_hit_token_rate"],
        "usable_gdn_state_avoided_tokens": raw["usable_gdn_state_avoided_tokens"],
        "usable_gdn_state_avoided_token_rate": raw["usable_gdn_state_avoided_token_rate"],
        "combined_hit_token_rate": raw["combined_hit_token_rate"],
        "combined_hit_tokens": raw["combined_hit_tokens"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--capacity-blocks", type=int)
    parser.add_argument("--expected-requests", type=int, default=881)
    parser.add_argument("--expected-block-size", type=int, default=16)
    parser.add_argument("--gdn-chunk-tokens", type=int, default=8192)
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline-cache-tps", type=float)
    parser.add_argument("--baseline-weighted-score", type=float)
    parser.add_argument("--baseline-metrics")
    parser.add_argument("--cache-coefficient", type=float, default=0.56)
    args = parser.parse_args(argv)

    records = read(args.logs)

    if (args.baseline_metrics is not None
            and (args.baseline_cache_tps is not None
                 or args.baseline_weighted_score is not None)):
        raise ValueError("baseline metrics file cannot be combined with manual baselines")

    baseline = (_baseline_metrics(args.baseline_metrics)
                if args.baseline_metrics is not None else None)
    if (baseline is not None
            and baseline["trace_session_sha256"] != records[0]["trace_session_sha256"]):
        raise ValueError("baseline metrics trace session does not match logs")

    baseline_cache_tps = baseline["cache_tps"] if baseline is not None else args.baseline_cache_tps
    baseline_weighted_score = (baseline["weighted_score"] if baseline is not None
                              else args.baseline_weighted_score)

    if args.expected_requests <= 0:
        raise ValueError("expected request count must be positive")
    if len(records) != args.expected_requests:
        raise ValueError(f"expected {args.expected_requests} requests, found {len(records)}")

    if args.expected_block_size <= 0:
        raise ValueError("expected block size must be positive")
    block_sizes = {record["block_size"] for record in records}
    if block_sizes != {args.expected_block_size}:
        raise ValueError(
            f"trace block sizes {sorted(block_sizes)} do not match {args.expected_block_size}")

    capacities = {record["capacity_blocks"] for record in records}
    if len(capacities) != 1:
        raise ValueError("capacity field is inconsistent")
    if args.capacity_blocks is None:
        capacity = capacities.pop()
    else:
        capacity = args.capacity_blocks
    if capacity <= 0:
        raise ValueError("capacity must be positive")

    total_prompt_tokens = sum(record["prompt_tokens"] for record in records)
    policy_results = {}
    for policy in ("off", "fine32", "admission64"):
        policy_results[policy] = _simulate(
            records, capacity, False, policy, args.gdn_chunk_tokens)
    # The current submission policy is fine32; admission64 is the candidate.
    vllm_lru = _metrics(policy_results["fine32"], total_prompt_tokens)
    frequency_aware = _metrics(policy_results["admission64"], total_prompt_tokens)

    report = {
        "requests": len(records),
        "prompt_tokens": total_prompt_tokens,
        "generated_tokens": sum(record["total_tokens"] - record["prompt_tokens"] for record in records),
        "prompt_full_blocks": sum(record["_prompt_full_blocks"] for record in records),
        "final_full_blocks": sum(record["full_blocks"] for record in records),
        "strict_eligible_blocks": sum(
            strict_prefix_block_count(record["prompt_tokens"], record["block_size"])
            for record in records
        ),
        "capacity_blocks": capacity,
        "trace_version": 4,
        "trace_session_sha256": records[0]["trace_session_sha256"],
        "trace_ordinals": {
            "first": records[0]["ordinal"],
            "last": records[-1]["ordinal"],
            "contiguous": True,
        },
        "source_logs": [_file_provenance(path) for path in args.logs],
        "policy_metrics": {
            policy: _metrics(result, total_prompt_tokens)
            for policy, result in policy_results.items()
        },
        "vllm_lru": vllm_lru,
        "frequency_aware": frequency_aware,
        "delta_hit_block_rate_percentage_points": (
            100 * (frequency_aware["raw_kv_contiguous_hit_block_rate"]
                    - vllm_lru["raw_kv_contiguous_hit_block_rate"])
        ),
        "delta_hit_token_rate_percentage_points": (
            100 * (frequency_aware["raw_kv_contiguous_hit_token_rate"]
                    - vllm_lru["raw_kv_contiguous_hit_token_rate"])
        ),
    }

    if ((baseline_cache_tps is None) != (baseline_weighted_score is None)):
        raise ValueError("baseline Cache TPS and weighted score must be provided together")

    if baseline_cache_tps is not None:
        if baseline_cache_tps < 0:
            raise ValueError("baseline Cache TPS must be non-negative")
        if baseline_weighted_score is None or baseline_weighted_score <= 0:
            raise ValueError("baseline weighted score must be positive")
        if args.cache_coefficient < 0:
            raise ValueError("cache coefficient must be non-negative")
        if frequency_aware["combined_hit_token_rate"] == 0:
            raise ValueError("baseline LRU token hit rate is zero")

        upper = baseline_cache_tps * (
            frequency_aware["combined_hit_token_rate"]
            / (vllm_lru["combined_hit_token_rate"] or math.inf)
        )
        delta_contribution = (upper - baseline_cache_tps) * args.cache_coefficient
        candidate_score = baseline_weighted_score + delta_contribution
        score_gain = candidate_score / baseline_weighted_score - 1.0

        report["weighted_cache_tps_upper_bound"] = {
            "baseline_cache_tps": baseline_cache_tps,
            "baseline_weighted_score": baseline_weighted_score,
            "baseline_metrics": baseline,
            "manual_baseline_is_non_qualifying": baseline is None,
            "coefficient": args.cache_coefficient,
            "baseline_contribution": baseline_cache_tps * args.cache_coefficient,
            "candidate_contribution": upper * args.cache_coefficient,
            "candidate_cache_tps_upper_bound": upper,
            "delta_contribution": delta_contribution,
            "candidate_weighted_score_upper_bound": candidate_score,
            "weighted_score_gain_fraction": score_gain,
        }

        gates = {
            "candidate_token_hit_rate_above_50pct": (
                frequency_aware["raw_kv_contiguous_hit_token_rate"] > 0.50
                or frequency_aware["combined_hit_token_rate"] > 0.50
            ),
            "token_hit_rate_gain_at_least_5pp": (
                frequency_aware["usable_gdn_state_avoided_token_rate"]
                - vllm_lru["usable_gdn_state_avoided_token_rate"] >= 0.05
            ),
            "weighted_score_gain_at_least_5pct": score_gain >= 0.05,
            "complete_unique_trace": len(records) == args.expected_requests,
            "single_runtime_session": True,
            "contiguous_ordered_trace": True,
            "baseline_metrics_file_provided": baseline is not None,
            "baseline_matches_trace_session": baseline is not None,
            "sequential_allocator_lifecycle_replay": True,
            "strict_contiguous_prefix_accounting": True,
            "policy_coverage": {"off": True, "fine32": True,
                                 "admission64": True},
        }
        report["qualification"] = {
            "ok": all(isinstance(v, bool) and v for v in gates.values()
                       if isinstance(v, bool)),
            "gates": gates,
            "projection_is_upper_bound": True,
        }

    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
