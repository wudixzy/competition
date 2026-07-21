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
from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any, Dict, Tuple

MARKER = "[BI100_CACHE_TRACE] "
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

# M1-44 fixed-shape medians at 131,072 tokens / 8,192 blocks.  The
# projection intentionally charges the measured one-way cost per content
# block, rather than scaling an aggregate cache rate.
M1_44_D2H_MS_PER_BLOCK = 944.999061524868 / 8192.0
M1_44_H2D_MS_PER_BLOCK = 1065.4232949018478 / 8192.0


def _percentile(values: Sequence[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)

try:
    from qwen3_6_scripts.gdn_prefix import (  # type: ignore
        GdnPrefixStatePolicy,
        capture_points_for_step,
        final_capture_key,
        key_at_strict_boundary,
        keys_from_block_hashes,
        gdn_restore_alignment,
        strict_prefix_block_count,
    )
except (ModuleNotFoundError, ImportError, OSError):
    # Fallback implementations for environments where the script is imported
    # directly and qwen3_6_scripts is unavailable.
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
        if restore_mode == "direct":
            return key_at_strict_boundary(
                block_hashes, prompt_tokens, block_size)
        if restore_mode not in {"chunk64", "aligned"}:
            raise ValueError(f"unknown GDN restore mode: {restore_mode}")
        if (replay_alignment <= 0 or replay_alignment % block_size != 0
                or prompt_tokens <= 1):
            return None
        boundary_tokens = (
            (prompt_tokens - 1) // replay_alignment * replay_alignment)
        block_count = min(
            len(block_hashes), boundary_tokens // block_size)
        if block_count <= 0:
            return None
        return (block_count, block_hashes[block_count - 1])

    def gdn_restore_alignment(restore_mode: str, block_size: int,
                              scheduler_chunk_tokens: int) -> int:
        if restore_mode == "direct":
            return block_size
        if restore_mode == "chunk64":
            alignment = 64
        elif restore_mode == "aligned":
            alignment = scheduler_chunk_tokens
        else:
            raise ValueError(f"unknown GDN restore mode: {restore_mode}")
        if alignment <= 0 or alignment % block_size != 0:
            raise ValueError("restore alignment must be divisible by block size")
        return alignment

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
    output_tps_p10 = data.get("output_tps_p10")
    success_rate = data.get("success_rate")
    for name, value in (("cache_tps", cache_tps),
                        ("weighted_score", weighted_score),
                        ("output_tps_p10", output_tps_p10),
                        ("success_rate", success_rate)):
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value)):
            raise ValueError(f"baseline metrics {name} must be finite")
    if cache_tps < 0:
        raise ValueError("baseline Cache TPS must be non-negative")
    if weighted_score <= 0:
        raise ValueError("baseline weighted score must be positive")
    if output_tps_p10 < 0:
        raise ValueError("baseline Output TPS P10 must be non-negative")
    if not 0 <= success_rate <= 1:
        raise ValueError("baseline success rate must be between zero and one")
    return {
        "run_id": run_id.strip(),
        "trace_session_sha256": trace_session,
        "cache_tps": float(cache_tps),
        "weighted_score": float(weighted_score),
        "output_tps_p10": float(output_tps_p10),
        "success_rate": float(success_rate),
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

                for field in ("ttft_s", "request_latency_s",
                              "time_in_queue_s", "observed_input_tps",
                              "observed_output_tps"):
                    value = record.get(field)
                    if value is not None and (
                            not isinstance(value, (int, float))
                            or isinstance(value, bool)
                            or not math.isfinite(value) or value < 0):
                        raise ValueError(f"{field} must be finite and non-negative")
                observed_cached = record.get(
                    "observed_effective_cached_tokens")
                if observed_cached is not None and (
                        not isinstance(observed_cached, int)
                        or isinstance(observed_cached, bool)
                        or not 0 <= observed_cached <= prompt_tokens):
                    raise ValueError(
                        "observed_effective_cached_tokens is inconsistent")
                qualification_trace = record.get("qualification_trace")
                if (qualification_trace is not None
                        and not isinstance(qualification_trace, bool)):
                    raise ValueError("qualification_trace must be boolean")

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
          candidate: bool) -> bytes:
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
    return victim


def _qualification_trace(records: Sequence[Dict[str, Any]],
                         explicitly_declared: bool = False) -> bool:
    """Only an explicitly declared, complete 881-request trace can qualify."""
    if not explicitly_declared:
        return False
    if len(records) != 881:
        return False
    if [record["ordinal"] for record in records] != list(range(1, 882)):
        return False
    return True


def _simulate(records: Sequence[Dict[str, Any]], capacity: int,
              candidate: bool, policy: str,
              gdn_chunk_tokens: int = 8192,
              restore_mode: str = "direct",
              cpu_capacity: int = 0,
              h2d_ms_per_block: float = M1_44_H2D_MS_PER_BLOCK,
              d2h_ms_per_block: float = M1_44_D2H_MS_PER_BLOCK) -> Dict[str, Any]:
    if policy not in {"off", "fine32", "admission64"}:
        raise ValueError("policy must be one of off, fine32, admission64")
    if gdn_chunk_tokens <= 0:
        raise ValueError("gdn_chunk_tokens must be positive")
    if cpu_capacity < 0:
        raise ValueError("cpu_capacity must be non-negative")
    for name, value in (("h2d_ms_per_block", h2d_ms_per_block),
                        ("d2h_ms_per_block", d2h_ms_per_block)):
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and non-negative")
    cache = set()
    last = {}
    frequency = defaultdict(int)
    cpu_cache: OrderedDict[bytes, None] = OrderedDict()
    hits = total = hit_tokens = gap_blocks = 0
    cpu_hits = h2d_blocks = d2h_blocks = d2h_skipped_blocks = 0
    avoided_tokens = 0
    request_results: list[Dict[str, Any]] = []
    gdn_policy = GdnPrefixStatePolicy(policy)

    for tick, record in enumerate(records, 1):
        hashes = record["_hashes"]
        block_size = record["block_size"]
        prompt_tokens = record["prompt_tokens"]
        strict_blocks = strict_prefix_block_count(prompt_tokens, block_size)
        prompt_full_blocks = record["_prompt_full_blocks"]
        prompt_hashes = hashes[:prompt_full_blocks]

        hit = 0
        cpu_hit = 0
        request_h2d = 0
        request_d2h = 0
        effective_hit = 0
        saw_gap = False
        active_blocks = 0
        h2d_claimed = False
        ready_cpu = set(cpu_cache)
        step_new_cpu: set[bytes] = set()
        deferred_d2h: list[bytes] = []
        deferred_hashes: set[bytes] = set()

        def demote(victim: bytes) -> None:
            nonlocal request_d2h, d2h_blocks
            if cpu_capacity <= 0:
                return
            if victim in cpu_cache:
                cpu_cache.move_to_end(victim)
                return
            if victim in deferred_hashes:
                return
            if len(cpu_cache) < cpu_capacity:
                cpu_cache[victim] = None
                step_new_cpu.add(victim)
                request_d2h += 1
                d2h_blocks += 1
                return
            deferred_d2h.append(victim)
            deferred_hashes.add(victim)

        def resolve_cpu_step(has_h2d: bool) -> None:
            nonlocal request_d2h, d2h_blocks, d2h_skipped_blocks
            if has_h2d:
                d2h_skipped_blocks += len(deferred_d2h)
            else:
                for victim in deferred_d2h:
                    cpu_victim = next(
                        (item for item in cpu_cache
                         if item not in step_new_cpu), None)
                    if cpu_victim is None:
                        d2h_skipped_blocks += 1
                        continue
                    del cpu_cache[cpu_victim]
                    cpu_cache[victim] = None
                    step_new_cpu.add(victim)
                    request_d2h += 1
                    d2h_blocks += 1
            deferred_d2h.clear()
            deferred_hashes.clear()
            step_new_cpu.clear()

        def allocate_slot() -> None:
            nonlocal active_blocks
            if len(cache) + active_blocks >= capacity:
                victim = _evict(cache, last, frequency, candidate)
                demote(victim)
            active_blocks += 1

        for depth, block in enumerate(prompt_hashes):
            was_cached = block in cache
            # A D2H destination becomes readable only after this scheduler
            # step executes, so prompt lookup is limited to the start snapshot.
            was_cpu_cached = not was_cached and block in ready_cpu
            if was_cached:
                cache.remove(block)
                active_blocks += 1
            elif was_cpu_cached:
                h2d_claimed = True
                request_h2d += 1
                h2d_blocks += 1
                cpu_cache.move_to_end(block)
                allocate_slot()
            else:
                allocate_slot()
            frequency[block] += 1
            if was_cpu_cached:
                cpu_hit += 1
            if depth >= strict_blocks:
                continue
            source_hit = was_cached or was_cpu_cached
            if not saw_gap and source_hit:
                effective_hit += 1
                if was_cached:
                    hit += 1
            else:
                saw_gap = True
                if was_cached:
                    gap_blocks += 1

        while active_blocks < record["prompt_allocated_blocks"]:
            allocate_slot()
        if active_blocks != record["prompt_allocated_blocks"]:
            raise ValueError("prompt allocation replay diverged")
        resolve_cpu_step(h2d_claimed)
        prompt_d2h = request_d2h

        hits += hit
        cpu_hits += cpu_hit
        total += strict_blocks
        hit_tokens += hit * block_size

        # A recurrent state is usable only while the exact contiguous KV
        # prefix is also resident. State-only hits cannot restore attention KV.
        live_keys = keys_from_block_hashes(prompt_hashes[:effective_hit])
        restore_alignment = gdn_restore_alignment(
            restore_mode, block_size, gdn_chunk_tokens)
        if restore_mode != "direct":
            live_keys = [
                key for key in live_keys
                if key[0] * block_size % restore_alignment == 0
            ]
        restore_key = gdn_policy.select_restore(live_keys, len(live_keys))
        request_avoided_tokens = (
            restore_key[0] * block_size if restore_key is not None else 0)
        avoided_tokens += request_avoided_tokens
        residual_prefill_tokens = max(
            0, prompt_tokens - request_avoided_tokens)
        request_result: Dict[str, Any] = {
            "ordinal": record["ordinal"],
            "prompt_tokens": prompt_tokens,
            "raw_kv_contiguous_hit_tokens": hit * block_size,
            "raw_kv_contiguous_hit_blocks": hit,
            "cpu_hit_blocks": cpu_hit,
            "h2d_blocks": request_h2d,
            "prefill_d2h_blocks": prompt_d2h,
            "effective_hit_blocks": (
                restore_key[0] if restore_key is not None else 0),
            "effective_hit_tokens": request_avoided_tokens,
            "usable_gdn_state_avoided_tokens": request_avoided_tokens,
            "residual_prefill_tokens": residual_prefill_tokens,
        }

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
                hashes, prompt_tokens, block_size, restore_mode,
                restore_alignment)
            if final_key is not None:
                capture_targets.append(final_key)
        gdn_policy.admit(dict.fromkeys(capture_targets))

        mutable_tail = prompt_tokens % block_size != 0
        for block in hashes[len(prompt_hashes):]:
            if not mutable_tail:
                allocate_slot()
                resolve_cpu_step(False)
                mutable_tail = True
            frequency[block] += 1
            if block in cache:
                cache.remove(block)
            mutable_tail = False

        if record["total_tokens"] % block_size != 0:
            if not mutable_tail:
                allocate_slot()
                resolve_cpu_step(False)
                mutable_tail = True
        elif mutable_tail:
            raise ValueError("final mutable-tail replay diverged")
        if active_blocks != record["allocated_blocks"]:
            raise ValueError("final allocation replay diverged")
        if deferred_d2h or deferred_hashes or step_new_cpu:
            raise ValueError("CPU transfer replay did not drain")

        decode_d2h = request_d2h - prompt_d2h
        request_result.update({
            "d2h_blocks": request_d2h,
            "decode_d2h_blocks": decode_d2h,
        })
        ttft_s = record.get("ttft_s")
        request_latency_s = record.get("request_latency_s")
        observed_cached = record.get("observed_effective_cached_tokens")
        if (isinstance(ttft_s, (int, float))
                and isinstance(request_latency_s, (int, float))
                and isinstance(observed_cached, int)):
            queue_s = float(record.get("time_in_queue_s", 0.0))
            observed_residual = max(1, prompt_tokens - observed_cached)
            prefill_s = max(0.0, float(ttft_s) - queue_s)
            projected_prefill_s = (
                prefill_s * residual_prefill_tokens / observed_residual)
            h2d_s = request_h2d * h2d_ms_per_block / 1000.0
            prefill_d2h_s = prompt_d2h * d2h_ms_per_block / 1000.0
            decode_d2h_s = decode_d2h * d2h_ms_per_block / 1000.0
            projected_ttft_s = (
                queue_s + projected_prefill_s + h2d_s + prefill_d2h_s)
            baseline_decode_s = max(
                0.0, float(request_latency_s) - float(ttft_s))
            projected_latency_s = (
                projected_ttft_s + baseline_decode_s + decode_d2h_s)
            request_result.update({
                "observed_effective_cached_tokens": observed_cached,
                "observed_residual_prefill_tokens": observed_residual,
                "baseline_prefill_s": prefill_s,
                "projected_prefill_s": projected_prefill_s,
                "h2d_transfer_s": h2d_s,
                "prefill_d2h_transfer_s": prefill_d2h_s,
                "decode_d2h_transfer_s": decode_d2h_s,
                "d2h_transfer_s": prefill_d2h_s + decode_d2h_s,
                "observed_ttft_s": float(ttft_s),
                "projected_ttft_s": projected_ttft_s,
                "observed_request_latency_s": float(request_latency_s),
                "projected_request_latency_s": projected_latency_s,
            })
        request_results.append(request_result)

        cache.update(hashes)
        if len(cache) > capacity:
            raise ValueError("free replay exceeded cache capacity")
        for depth, block in enumerate(hashes):
            last[block] = (tick, depth)

    total_prompt_tokens = sum(record["prompt_tokens"] for record in records)
    return {
        "policy": policy,
        "gdn_restore_mode": restore_mode,
        "kv_eviction_policy": (
            "frequency_aware_m1_29" if candidate else "lru"),
        "hit_blocks": hits,
        "total_blocks": total,
        "hit_tokens": hit_tokens,
        "gap_blocks": gap_blocks,
        "cpu_capacity_blocks": cpu_capacity,
        "cpu_hit_blocks": cpu_hits,
        "h2d_blocks": h2d_blocks,
        "d2h_blocks": d2h_blocks,
        "d2h_skipped_blocks": d2h_skipped_blocks,
        "effective_hit_blocks": sum(
            item["effective_hit_blocks"] for item in request_results),
        "effective_hit_tokens": avoided_tokens,
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
        "residual_prefill_tokens": sum(
            item["residual_prefill_tokens"] for item in request_results),
        "request_results": request_results,
        "final_cache": cache,
        "final_cpu_cache": cpu_cache,
        "gdn_policy_cache_size": len(gdn_policy),
    }


def simulate(records: Sequence[Dict[str, Any]], capacity: int,
             candidate: bool = False, policy: str = "off",
             gdn_chunk_tokens: int = 8192,
             restore_mode: str = "direct", cpu_capacity: int = 0,
             h2d_ms_per_block: float = M1_44_H2D_MS_PER_BLOCK,
             d2h_ms_per_block: float = M1_44_D2H_MS_PER_BLOCK) -> Dict[str, Any]:
    if isinstance(candidate, str) and policy == "off":
        policy = candidate
        candidate = False
    result = _simulate(records, capacity, candidate, policy,
                       gdn_chunk_tokens, restore_mode, cpu_capacity,
                       h2d_ms_per_block, d2h_ms_per_block)
    del result["final_cache"]
    del result["final_cpu_cache"]
    return result


def _metrics(raw: Dict[str, Any], total_prompt_tokens: int) -> Dict[str, Any]:
    request_results = raw["request_results"]
    projected_ttfts = [
        item["projected_ttft_s"] for item in request_results
        if "projected_ttft_s" in item
    ]
    projected_latencies = [
        item["projected_request_latency_s"] for item in request_results
        if "projected_request_latency_s" in item
    ]
    timing_complete = (
        len(projected_ttfts) == len(request_results)
        and len(projected_latencies) == len(request_results))
    metrics = {
        "policy": raw["policy"],
        "gdn_restore_mode": raw["gdn_restore_mode"],
        "kv_eviction_policy": raw["kv_eviction_policy"],
        "hit_tokens": raw["hit_tokens"],
        "hit_blocks": raw["hit_blocks"],
        "total_blocks": raw["total_blocks"],
        "gap_blocks": raw["gap_blocks"],
        "cpu_capacity_blocks": raw["cpu_capacity_blocks"],
        "cpu_hit_blocks": raw["cpu_hit_blocks"],
        "cpu_hit_block_rate": (
            raw["cpu_hit_blocks"] / raw["total_blocks"]
            if raw["total_blocks"] else 0.0),
        "h2d_blocks": raw["h2d_blocks"],
        "d2h_blocks": raw["d2h_blocks"],
        "d2h_skipped_blocks": raw["d2h_skipped_blocks"],
        "effective_hit_blocks": raw["effective_hit_blocks"],
        "effective_hit_tokens": raw["effective_hit_tokens"],
        "effective_hit_block_rate": (
            raw["effective_hit_blocks"] / raw["total_blocks"]
            if raw["total_blocks"] else 0.0),
        "raw_kv_contiguous_hit_tokens": raw["raw_kv_contiguous_hit_tokens"],
        "raw_kv_contiguous_hit_blocks": raw["hit_blocks"],
        "raw_kv_contiguous_hit_block_rate": raw["raw_kv_contiguous_hit_block_rate"],
        "raw_kv_contiguous_hit_token_rate": raw["raw_kv_hit_token_rate"],
        "usable_gdn_state_avoided_tokens": raw["usable_gdn_state_avoided_tokens"],
        "usable_gdn_state_avoided_token_rate": raw["usable_gdn_state_avoided_token_rate"],
        "combined_hit_token_rate": raw["combined_hit_token_rate"],
        "combined_hit_tokens": raw["combined_hit_tokens"],
        "residual_prefill_tokens": raw["residual_prefill_tokens"],
        "residual_prefill_token_rate": (
            raw["residual_prefill_tokens"] / total_prompt_tokens
            if total_prompt_tokens else 0.0),
        "residual_prefill_tokens_p90": _percentile([
            item["residual_prefill_tokens"] for item in request_results
        ], 90),
        "per_request_timing_projection_complete": timing_complete,
        "projected_ttft_p90_s": (
            _percentile(projected_ttfts, 90) if timing_complete else None),
        "projected_sequential_wall_s": (
            sum(projected_latencies) if timing_complete else None),
    }
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--capacity-blocks", type=int)
    parser.add_argument("--cpu-capacity-blocks", "--cpu-tier-capacity-blocks",
                        dest="cpu_capacity_blocks", type=int, default=0)
    parser.add_argument("--expected-requests", type=int, default=881)
    parser.add_argument(
        "--qualification-trace",
        action="store_true",
        help=("Explicitly declare that the input is the complete official "
              "881-request trace. Structural checks still apply."))
    parser.add_argument("--expected-block-size", type=int, default=16)
    parser.add_argument("--gdn-chunk-tokens", type=int, default=8192)
    parser.add_argument(
        "--gdn-restore-mode",
        choices=("direct", "chunk64", "aligned"),
        default="direct")
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline-cache-tps", type=float)
    parser.add_argument("--baseline-weighted-score", type=float)
    parser.add_argument("--baseline-metrics")
    parser.add_argument(
        "--h2d-ms-per-block", type=float,
        default=M1_44_H2D_MS_PER_BLOCK)
    parser.add_argument(
        "--d2h-ms-per-block", type=float,
        default=M1_44_D2H_MS_PER_BLOCK)
    args = parser.parse_args(argv)

    records = read(args.logs)

    if (args.baseline_cache_tps is not None
            or args.baseline_weighted_score is not None):
        raise ValueError(
            "aggregate hit-rate scaling is disabled; provide --baseline-metrics "
            "and per-request timing fields in the trace")

    baseline = (_baseline_metrics(args.baseline_metrics)
                if args.baseline_metrics is not None else None)
    if (baseline is not None
            and baseline["trace_session_sha256"] != records[0]["trace_session_sha256"]):
        raise ValueError("baseline metrics trace session does not match logs")

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
    if args.cpu_capacity_blocks < 0:
        raise ValueError("cpu capacity must be non-negative")
    for name, value in (("h2d-ms-per-block", args.h2d_ms_per_block),
                        ("d2h-ms-per-block", args.d2h_ms_per_block)):
        if (not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and non-negative")

    total_prompt_tokens = sum(record["prompt_tokens"] for record in records)
    control_policy_results = {}
    policy_results = {}
    for policy in ("off", "fine32", "admission64"):
        restore_mode = (
            args.gdn_restore_mode if policy == "admission64" else "direct")
        control_policy_results[policy] = _simulate(
            records, capacity, False, policy, args.gdn_chunk_tokens,
            restore_mode, 0, args.h2d_ms_per_block, args.d2h_ms_per_block)
        policy_results[policy] = _simulate(
            records, capacity, False, policy, args.gdn_chunk_tokens,
            restore_mode, args.cpu_capacity_blocks,
            args.h2d_ms_per_block, args.d2h_ms_per_block)
    policy_results["admission64_m1_29"] = _simulate(
        records, capacity, True, "admission64", args.gdn_chunk_tokens,
        args.gdn_restore_mode, args.cpu_capacity_blocks,
        args.h2d_ms_per_block, args.d2h_ms_per_block)
    # The current submission policy is fine32; admission64 is the candidate.
    vllm_lru = _metrics(
        control_policy_results["fine32"], total_prompt_tokens)
    admission64_control = _metrics(
        control_policy_results["admission64"], total_prompt_tokens)
    admission64 = _metrics(
        policy_results["admission64"], total_prompt_tokens)
    frequency_aware = _metrics(
        policy_results["admission64_m1_29"], total_prompt_tokens)

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
        "cpu_capacity_blocks": args.cpu_capacity_blocks,
        "h2d_ms_per_block": args.h2d_ms_per_block,
        "d2h_ms_per_block": args.d2h_ms_per_block,
        "trace_version": 4,
        "qualification_trace": _qualification_trace(
            records, explicitly_declared=args.qualification_trace),
        "candidate_gdn_restore_mode": args.gdn_restore_mode,
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
        "control_policy_metrics": {
            policy: _metrics(result, total_prompt_tokens)
            for policy, result in control_policy_results.items()
        },
        "vllm_lru": vllm_lru,
        "admission64_control": admission64_control,
        "admission64": admission64,
        "frequency_aware": frequency_aware,
        "cpu_tier_admission64_effective_hit_gain_percentage_points": (
            100 * (admission64["usable_gdn_state_avoided_token_rate"]
                   - admission64_control[
                       "usable_gdn_state_avoided_token_rate"])),
        "delta_hit_block_rate_percentage_points": (
            100 * (frequency_aware["raw_kv_contiguous_hit_block_rate"]
                    - vllm_lru["raw_kv_contiguous_hit_block_rate"])
        ),
        "delta_hit_token_rate_percentage_points": (
            100 * (frequency_aware["raw_kv_contiguous_hit_token_rate"]
                    - vllm_lru["raw_kv_contiguous_hit_token_rate"])
        ),
    }

    if baseline is not None:
        qualifications = {}
        for name in ("admission64", "admission64_m1_29"):
            candidate_metrics = report["policy_metrics"][name]
            wall_s = candidate_metrics["projected_sequential_wall_s"]
            timing_complete = wall_s is not None and wall_s > 0
            projected_input_tps = (
                total_prompt_tokens / wall_s if timing_complete else None)
            projected_cache_tps = (
                candidate_metrics["usable_gdn_state_avoided_tokens"] / wall_s
                if timing_complete else None)
            projected_score = (
                baseline["output_tps_p10"] * 16.796
                + projected_input_tps * 2.799
                + projected_cache_tps * 0.56
                if timing_complete else None)
            score_gain = (
                projected_score / baseline["weighted_score"] - 1.0
                if projected_score is not None else None)
            gates = {
                "qualification_trace": report["qualification_trace"],
                "per_request_timing_complete": timing_complete,
                "effective_hit_rate_at_least_50pct": (
                    candidate_metrics["usable_gdn_state_avoided_token_rate"]
                    >= 0.50),
                "effective_hit_rate_gain_at_least_5pp": (
                    candidate_metrics["usable_gdn_state_avoided_token_rate"]
                    - vllm_lru["usable_gdn_state_avoided_token_rate"] >= 0.05),
                "weighted_score_gain_at_least_5pct": (
                    score_gain is not None and score_gain >= 0.05),
                "output_tps_p10_at_least_20": (
                    baseline["output_tps_p10"] >= 20.0),
                "success_rate_at_least_99pct": (
                    baseline["success_rate"] >= 0.99),
                "projected_ttft_p90_at_most_5s": (
                    candidate_metrics["projected_ttft_p90_s"] is not None
                    and candidate_metrics["projected_ttft_p90_s"] <= 5.0),
                "projected_weighted_score_at_least_8000": (
                    projected_score is not None and projected_score >= 8000.0),
            }
            qualifications[name] = {
                "ok": report["qualification_trace"] and all(gates.values()),
                "gates": gates,
                "qualification_trace": report["qualification_trace"],
                "projection_model": "per_request_residual_prefill",
                "projected_input_tps": projected_input_tps,
                "projected_cache_tps": projected_cache_tps,
                "projected_weighted_score": projected_score,
                "weighted_score_gain_fraction": score_gain,
            }
        report["qualification"] = qualifications

    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
