#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


EVENT_RE = re.compile(
    r"(?:VllmWorkerProcess pid=(?P<pid>\d+).*?)?"
    r"\[BI100_PROFILE_EVENT\]\s+(?P<payload>\{.*\})")
SCHEMA = "bi100-m1-48-prefill-path-profile-v2"
VERSION = 2
EVENT_SCHEMA = "bi100-profile-event-v1"
EVENT_VERSION = 1
EVENT_KEYS = {
    "schema",
    "version",
    "tp_rank",
    "forward_index",
    "metadata",
    "event_count",
    "model_forward_event_count",
    "regions",
    "counters",
    "host_model_start_to_flush_ms",
    "host_gap_since_previous_flush_ms",
}
SERVICE_SCHEMA = "bi100-m1-48-prefill-service-v1"
SERVICE_VERSION = 1
MAX_PROFILE_OVERHEAD_FRACTION = 0.15
MAX_MODEL_RANK_SPREAD_FRACTION = 0.10

EXCLUSIVE_MODEL_REGIONS = (
    "model.embed",
    "layer.input_norm",
    "layer.gdn",
    "layer.full_attn",
    "layer.post_attn_norm",
    "layer.moe",
    "model.final_norm",
)
FULL_ATTN_SUBREGIONS = (
    "full_attn.project_qgkv",
    "full_attn.norm_rope",
    "full_attn.attention",
    "full_attn.gate",
    "full_attn.output_proj",
)
PER_FORWARD_REGION_COUNTS = {
    "model.forward": 1,
    "model.embed": 1,
    "layer.input_norm": 40,
    "layer.gdn": 30,
    "layer.full_attn": 10,
    "layer.post_attn_norm": 40,
    "layer.moe": 40,
    "model.final_norm": 1,
    "full_attn.project_qgkv": 10,
    "full_attn.norm_rope": 10,
    "full_attn.attention": 10,
    "full_attn.gate": 10,
    "full_attn.output_proj": 10,
    "moe.router": 40,
    "moe.routed": 40,
    "moe.shared": 40,
    "moe.combine": 40,
    "moe.all_reduce": 40,
    "xformers.kv_write": 10,
}
CONDITIONAL_REGIONS = {
    "xformers.dense_prefill",
    "xformers.paged_prefill",
    "paged_attn.prefix_pytorch",
    "gdn_prefix.restore",
    "gdn_prefix.save",
}
ALLOWED_REGIONS = set(PER_FORWARD_REGION_COUNTS) | CONDITIONAL_REGIONS


def _finite(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def parse_log(path: Path) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EVENT_RE.search(line)
        if match is None:
            continue
        process = match.group("pid") or "driver"
        payload = json.loads(match.group("payload"))
        if (not isinstance(payload, dict)
                or payload.get("schema") != EVENT_SCHEMA
                or payload.get("version") != EVENT_VERSION):
            raise ValueError(f"invalid profile event schema for {process}")
        if set(payload) != EVENT_KEYS:
            raise ValueError(f"invalid profile event fields for {process}")
        records[process].append(payload)
    if not records:
        raise ValueError(f"no BI100_PROFILE_EVENT records found in {path}")
    for process, process_records in records.items():
        process_records.sort(key=lambda item: item["forward_index"])
        indices = [item.get("forward_index") for item in process_records]
        expected = list(range(len(indices)))
        if indices != expected:
            raise ValueError(
                f"non-contiguous forward indices for {process}: {indices}")
    return dict(records)


def bind_tp_ranks(
    process_records: dict[str, list[dict[str, Any]]],
    expected_processes: int,
) -> dict[int, list[dict[str, Any]]]:
    by_rank: dict[int, list[dict[str, Any]]] = {}
    for process, records in process_records.items():
        ranks = {record.get("tp_rank") for record in records}
        if len(ranks) != 1:
            raise ValueError(f"TP rank drift for process {process}: {ranks}")
        rank = next(iter(ranks))
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise ValueError(f"invalid TP rank for process {process}: {rank!r}")
        if rank in by_rank:
            raise ValueError(f"duplicate TP rank {rank}")
        by_rank[rank] = records
    expected_ranks = list(range(expected_processes))
    if sorted(by_rank) != expected_ranks:
        raise ValueError(
            f"TP ranks must equal {expected_ranks}, got {sorted(by_rank)}")
    return by_rank


def split_requests(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    requests: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    saw_decode = False
    for record in records:
        metadata = record.get("metadata") or {}
        expected_metadata_keys = {
            "phase",
            "prefill_tokens",
            "decode_tokens",
            "context_len",
            "gdn_restore",
            "gdn_capture_points",
            "gdn_evict_keys",
        }
        if set(metadata) != expected_metadata_keys:
            raise ValueError(
                "profile metadata fields differ: "
                f"expected {sorted(expected_metadata_keys)}, "
                f"got {sorted(metadata)}")
        phase = metadata.get("phase")
        if phase not in {"prefill", "decode"}:
            raise ValueError(f"unknown profile phase: {phase!r}")
        if phase == "prefill" and current:
            previous = current[-1].get("metadata") or {}
            previous_end = (
                int(previous.get("context_len") or 0)
                + int(previous.get("prefill_tokens") or 0))
            context_len = int(metadata.get("context_len") or 0)
            if saw_decode or context_len < previous_end:
                requests.append(current)
                current = []
                saw_decode = False
        if phase == "decode":
            saw_decode = True
        current.append(record)
    if current:
        requests.append(current)
    return requests


def _prefill_tokens(records: list[dict[str, Any]]) -> int:
    return sum(
        int((record.get("metadata") or {}).get("prefill_tokens") or 0)
        for record in records)


def _aggregate_regions(records: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for record in records:
        for name, stats in (record.get("regions") or {}).items():
            value = stats.get("total_ms") if isinstance(stats, dict) else None
            if not _finite(value) or value < 0:
                raise ValueError(f"invalid region duration for {name}: {value!r}")
            totals[name] += float(value)
    return dict(totals)


def _region_counts(record: dict[str, Any]) -> dict[str, int]:
    counts = {}
    regions = record.get("regions") or {}
    if not isinstance(regions, dict):
        raise ValueError("profile regions must be an object")
    for name, stats in regions.items():
        value = stats.get("count") if isinstance(stats, dict) else None
        if (not isinstance(value, int) or isinstance(value, bool)
                or value <= 0):
            raise ValueError(f"invalid region count for {name}: {value!r}")
        counts[name] = value
    event_count = record.get("event_count")
    if event_count != sum(counts.values()):
        raise ValueError(
            f"event_count {event_count!r} does not equal region count sum "
            f"{sum(counts.values())}")
    unknown = sorted(set(counts) - ALLOWED_REGIONS)
    if unknown:
        raise ValueError(f"unexpected profile regions: {unknown}")
    return counts


def _mean_rank_regions(
    rank_records: dict[int, list[dict[str, Any]]],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    rank_totals = {
        rank: _aggregate_regions(records)
        for rank, records in rank_records.items()
    }
    names = sorted({name for totals in rank_totals.values() for name in totals})
    means = {}
    spread = {}
    for name in names:
        values = [totals.get(name, 0.0) for totals in rank_totals.values()]
        means[name] = statistics.mean(values)
        spread[name] = {
            "min_ms": min(values),
            "max_ms": max(values),
            "mean_ms": means[name],
        }
    return means, spread


def _counter_key(rows: Any) -> str:
    if not isinstance(rows, list):
        raise ValueError("profile counters must be a list")
    return json.dumps(rows, sort_keys=True, separators=(",", ":"))


def _aligned_forwards(
    rank_records: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    counts = {len(records) for records in rank_records.values()}
    if len(counts) != 1:
        raise ValueError(f"forward-count mismatch by rank: {counts}")
    rows = []
    for offset in range(next(iter(counts))):
        selected = {
            rank: records[offset]
            for rank, records in rank_records.items()
        }
        indices = {record.get("forward_index") for record in selected.values()}
        if len(indices) != 1:
            raise ValueError(f"forward index mismatch at offset {offset}: {indices}")
        metadata_values = {
            json.dumps(record.get("metadata") or {}, sort_keys=True)
            for record in selected.values()
        }
        if len(metadata_values) != 1:
            raise ValueError(f"forward metadata mismatch at offset {offset}")
        counter_values = {
            _counter_key(record.get("counters") or [])
            for record in selected.values()
        }
        if len(counter_values) != 1:
            raise ValueError(f"forward counter mismatch at offset {offset}")
        if any(record.get("model_forward_event_count") != 1
               for record in selected.values()):
            raise ValueError(
                f"model.forward event count must equal one at offset {offset}")
        region_counts = {
            json.dumps(_region_counts(record), sort_keys=True,
                       separators=(",", ":"))
            for record in selected.values()
        }
        if len(region_counts) != 1:
            raise ValueError(f"region count mismatch at offset {offset}")
        regions, _ = _mean_rank_regions({
            rank: [record] for rank, record in selected.items()
        })
        model_by_rank = {
            rank: float(record["regions"]["model.forward"]["total_ms"])
            for rank, record in selected.items()
        }
        model_mean = statistics.mean(model_by_rank.values())
        if model_mean <= 0:
            raise ValueError(
                f"model.forward duration must be positive at offset {offset}")
        model_spread = (
            max(model_by_rank.values()) - min(model_by_rank.values())
        ) / model_mean
        host_model_by_rank = {
            rank: record.get("host_model_start_to_flush_ms")
            for rank, record in selected.items()
        }
        if any(not _finite(value) or value <= 0
               for value in host_model_by_rank.values()):
            raise ValueError(f"invalid host model duration at offset {offset}")
        host_gap_by_rank = {
            rank: record.get("host_gap_since_previous_flush_ms")
            for rank, record in selected.items()
        }
        finite_gaps = [value for value in host_gap_by_rank.values()
                       if _finite(value)]
        if finite_gaps and len(finite_gaps) != len(host_gap_by_rank):
            raise ValueError(f"partial host gap data at offset {offset}")
        rows.append({
            "offset": offset,
            "forward_index": next(iter(indices)),
            "metadata": next(iter(selected.values())).get("metadata") or {},
            "regions_ms_per_rank_mean": regions,
            "model_forward_ms_by_rank": model_by_rank,
            "model_rank_spread_fraction": model_spread,
            "counters": json.loads(next(iter(counter_values))),
            "host_model_start_to_flush_ms_by_rank": host_model_by_rank,
            "host_gap_since_previous_flush_ms_by_rank": host_gap_by_rank,
        })
    return rows


def _worker_request_spans_ms(
    rank_records: dict[int, list[dict[str, Any]]],
) -> dict[int, float]:
    spans = {}
    for rank, records in rank_records.items():
        if not records:
            raise ValueError(f"rank {rank} has no selected prefill forwards")
        total = 0.0
        for offset, record in enumerate(records):
            duration = record.get("host_model_start_to_flush_ms")
            if not _finite(duration) or duration <= 0:
                raise ValueError(f"rank {rank} has an invalid host duration")
            total += float(duration)
            if offset:
                gap = record.get("host_gap_since_previous_flush_ms")
                if not _finite(gap) or gap < 0:
                    raise ValueError(f"rank {rank} has an invalid host gap")
                total += float(gap)
        spans[rank] = total
    return spans


def _expected_chunk_geometry(total: int, chunk_size: int) -> list[tuple[int, int]]:
    if total <= 0 or chunk_size <= 0:
        raise ValueError("total and chunk size must be positive")
    geometry = []
    context = 0
    while context < total:
        tokens = min(chunk_size, total - context)
        geometry.append((context, tokens))
        context += tokens
    return geometry


def _strict_segments(
    context_len: int,
    query_len: int,
    block_size: int,
) -> list[tuple[int, int]]:
    strict_prefix_len = ((context_len + query_len - 1) // block_size) * block_size
    split = strict_prefix_len - context_len
    if 0 < split < query_len:
        return [(split, context_len),
                (query_len - split, strict_prefix_len)]
    return [(query_len, context_len)]


def _expected_dispatch_rows(
    context_len: int,
    query_len: int,
    block_size: int,
) -> list[dict[str, Any]]:
    rows = []
    for segment_len, segment_context in _strict_segments(
            context_len, query_len, block_size):
        rows.append({
            "block_size": block_size,
            "context_len": segment_context,
            "count": 10,
            "head_dim": 256,
            "kv_heads": 1,
            "name": "paged_attn.prefix_dispatch",
            "path": "pytorch",
            "query_heads": 8,
            "query_len": segment_len,
            "request_query_len": query_len,
        })
    return sorted(rows, key=lambda row: json.dumps(row, sort_keys=True))


def _validate_forward_contract(
    records: list[dict[str, Any]],
    rank: int,
    expected_geometry: list[tuple[int, int]],
    block_size: int,
    reasons: list[str],
) -> None:
    if len(records) != len(expected_geometry):
        reasons.append(
            f"rank {rank} must have {len(expected_geometry)} prefill forwards, "
            f"got {len(records)}")
    for offset, expected in enumerate(expected_geometry):
        if offset >= len(records):
            break
        record = records[offset]
        metadata = record.get("metadata") or {}
        geometry = (metadata.get("context_len"), metadata.get("prefill_tokens"))
        if geometry != expected:
            reasons.append(
                f"rank {rank} prefill geometry at offset {offset} must be "
                f"{expected}, got {geometry}")
        if metadata.get("decode_tokens") != 0:
            reasons.append(
                f"rank {rank} prefill carries decode tokens at offset {offset}")
        if metadata.get("gdn_restore") is not False:
            reasons.append(
                f"rank {rank} cold prefill restored GDN state at offset {offset}")
        expected_capture_count = 1 if offset == len(expected_geometry) - 1 else 0
        if metadata.get("gdn_capture_points") != expected_capture_count:
            reasons.append(
                f"rank {rank} cold prefill capture count at offset {offset} "
                f"must be {expected_capture_count}")
        if metadata.get("gdn_evict_keys") != 0:
            reasons.append(
                f"rank {rank} cold prefill evicted GDN state at offset {offset}")

        counts = _region_counts(record)
        expected_counts = dict(PER_FORWARD_REGION_COUNTS)
        segment_count = len(_strict_segments(
            expected[0], expected[1], block_size))
        expected_counts["xformers.paged_prefill"] = 10
        expected_counts["paged_attn.prefix_pytorch"] = 10 * segment_count
        restore_count = 1 if metadata.get("gdn_restore") else 0
        capture_count = metadata.get("gdn_capture_points")
        if not isinstance(capture_count, int) or isinstance(capture_count, bool):
            reasons.append(
                f"rank {rank} has invalid capture count at offset {offset}")
            capture_count = 0
        if restore_count:
            expected_counts["gdn_prefix.restore"] = restore_count
        if capture_count:
            expected_counts["gdn_prefix.save"] = capture_count
        if counts != expected_counts:
            reasons.append(
                f"rank {rank} region counts differ at offset {offset}")

        observed_counters = sorted(
            record.get("counters") or [],
            key=lambda row: json.dumps(row, sort_keys=True),
        )
        expected_counters = _expected_dispatch_rows(
            expected[0], expected[1], block_size)
        if observed_counters != expected_counters:
            reasons.append(
                f"rank {rank} paged dispatch differs at offset {offset}")


def _load_service_measurement(
    path: Path,
    expected_mode: str,
    expected_prefill_tokens: int,
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(report, dict)
            or report.get("schema") != SERVICE_SCHEMA
            or report.get("version") != SERVICE_VERSION):
        raise ValueError(f"invalid M1-48 service report: {path}")
    if report.get("mode") != expected_mode:
        raise ValueError(
            f"service report mode must be {expected_mode}: {path}")
    if report.get("qualified_measurement") is not True or report.get("reasons"):
        raise ValueError(f"service report is not qualified: {path}")
    protocol = report.get("protocol") or {}
    expected_protocol = {
        "stream": True,
        "max_tokens": 1,
        "min_tokens": 1,
        "temperature": 0,
        "seed": 20260722,
        "thinking": False,
        "target_prompt_tokens": expected_prefill_tokens,
        "max_model_len": 262144,
    }
    if protocol != expected_protocol:
        raise ValueError(f"service protocol mismatch: {path}")
    request = report.get("request") or {}
    if (request.get("prompt_tokens") != expected_prefill_tokens
            or request.get("cached_tokens") != 0
            or request.get("completion_tokens") != 1):
        raise ValueError(f"service request contract mismatch: {path}")
    for field in ("ttft_s", "elapsed_s"):
        if not _finite(request.get(field)) or request[field] <= 0:
            raise ValueError(f"invalid service request {field}: {path}")
    return report


def summarize(
    log_path: Path,
    expected_prefill_tokens: int,
    expected_processes: int = 4,
    profile_service: Path | None = None,
    control_service: Path | None = None,
    expected_chunk_size: int = 8192,
    block_size: int = 16,
) -> dict[str, Any]:
    if expected_prefill_tokens <= 0:
        raise ValueError("expected_prefill_tokens must be positive")
    if expected_processes != 4:
        raise ValueError("M1-48 requires exact TP4 evidence")
    if profile_service is None or control_service is None:
        raise ValueError("control and profile service reports are required")

    control = _load_service_measurement(
        control_service, "control", expected_prefill_tokens)
    profile = _load_service_measurement(
        profile_service, "profile", expected_prefill_tokens)
    if control.get("run_id") != profile.get("run_id"):
        raise ValueError("control and profile run IDs differ")
    control_request = control["request"]
    profile_request = profile["request"]
    if control_request.get("output_sha256") != profile_request.get("output_sha256"):
        raise ValueError("control and profile output digests differ")

    parsed_processes = parse_log(log_path)
    if len(parsed_processes) != expected_processes:
        raise ValueError(
            f"expected {expected_processes} profile processes, found "
            f"{len(parsed_processes)}")
    parsed = bind_tp_ranks(parsed_processes, expected_processes)
    grouped = {rank: split_requests(records)
               for rank, records in parsed.items()}
    selected: dict[int, list[dict[str, Any]]] = {}
    selected_indices = set()
    for rank, requests in grouped.items():
        if len(requests) != 1:
            raise ValueError(
                f"rank {rank} must contain exactly one profiled request, "
                f"got {len(requests)}")
        matches = [index for index, request in enumerate(requests)
                   if _prefill_tokens(request) == expected_prefill_tokens]
        if len(matches) != 1:
            raise ValueError(
                f"rank {rank} has {len(matches)} requests with exactly "
                f"{expected_prefill_tokens} prefill tokens")
        selected_indices.add(matches[0])
        selected[rank] = [
            record for record in requests[matches[0]]
            if (record.get("metadata") or {}).get("phase") == "prefill"
        ]
    if len(selected_indices) != 1:
        raise ValueError(
            f"selected request index differs by rank: {selected_indices}")
    if selected_indices != {0}:
        raise ValueError(
            f"profiled request must be the first request: {selected_indices}")

    expected_geometry = _expected_chunk_geometry(
        expected_prefill_tokens, expected_chunk_size)
    reasons = []
    for rank, records in selected.items():
        _validate_forward_contract(
            records, rank, expected_geometry, block_size, reasons)

    forwards = _aligned_forwards(selected)
    regions, rank_spread = _mean_rank_regions(selected)
    rank_totals = {
        rank: _aggregate_regions(records)
        for rank, records in selected.items()
    }
    worker_spans = _worker_request_spans_ms(selected)
    worker_critical_ms = max(worker_spans.values())
    model_mean_ms = regions.get("model.forward", 0.0)
    model_by_rank = {
        rank: totals.get("model.forward", 0.0)
        for rank, totals in rank_totals.items()
    }
    model_critical_ms = max(model_by_rank.values())
    layer_full_ms = regions.get("layer.full_attn", 0.0)
    attention_ms = regions.get("full_attn.attention", 0.0)
    kv_write_ms = regions.get("xformers.kv_write", 0.0)
    dense_prefill_ms = regions.get("xformers.dense_prefill", 0.0)
    paged_prefill_ms = regions.get("xformers.paged_prefill", 0.0)
    paged_segment_ms = regions.get("paged_attn.prefix_pytorch", 0.0)

    exclusive_coverage_by_rank = {}
    full_coverage_by_rank = {}
    for rank, totals in rank_totals.items():
        model = totals.get("model.forward", 0.0)
        full = totals.get("layer.full_attn", 0.0)
        exclusive = sum(totals.get(name, 0.0)
                        for name in EXCLUSIVE_MODEL_REGIONS)
        full_subregions = sum(totals.get(name, 0.0)
                              for name in FULL_ATTN_SUBREGIONS)
        exclusive_coverage_by_rank[rank] = exclusive / model if model > 0 else None
        full_coverage_by_rank[rank] = full_subregions / full if full > 0 else None
        if (exclusive_coverage_by_rank[rank] is None
                or not 0.97 <= exclusive_coverage_by_rank[rank] <= 1.03):
            reasons.append(
                f"rank {rank} exclusive regions do not close model.forward")
        if (full_coverage_by_rank[rank] is None
                or not 0.97 <= full_coverage_by_rank[rank] <= 1.03):
            reasons.append(
                f"rank {rank} full-attention regions do not close layer.full_attn")

    if model_mean_ms <= 0 or layer_full_ms <= 0 or attention_ms <= 0:
        reasons.append("required model/full-attention timing is missing")
    if paged_prefill_ms <= 0 or paged_segment_ms <= 0:
        reasons.append("required paged-prefill timing is missing")
    if paged_segment_ms > paged_prefill_ms * 1.02:
        reasons.append("paged segments exceed the inclusive paged-prefill region")
    if paged_prefill_ms > attention_ms * 1.02:
        reasons.append("paged prefill exceeds the inclusive attention region")

    profile_ttft_s = float(profile_request["ttft_s"])
    control_ttft_s = float(control_request["ttft_s"])
    overhead_fraction = profile_ttft_s / control_ttft_s - 1.0
    if abs(overhead_fraction) > MAX_PROFILE_OVERHEAD_FRACTION:
        reasons.append(
            "profile TTFT perturbation exceeds the fixed 15% bound")
    if worker_critical_ms > profile_ttft_s * 1000 * 1.05:
        reasons.append("worker critical span exceeds profiled TTFT")
    if model_critical_ms > worker_critical_ms * 1.05:
        reasons.append("model critical time exceeds worker critical span")
    model_spread = (
        (max(model_by_rank.values()) - min(model_by_rank.values()))
        / statistics.mean(model_by_rank.values())
        if statistics.mean(model_by_rank.values()) > 0 else math.inf)
    if model_spread > MAX_MODEL_RANK_SPREAD_FRACTION:
        reasons.append("model.forward TP-rank spread exceeds 10%")
    max_forward_model_spread = max(
        row["model_rank_spread_fraction"] for row in forwards)
    if max_forward_model_spread > MAX_MODEL_RANK_SPREAD_FRACTION:
        reasons.append("per-forward model TP-rank spread exceeds 10%")

    region_critical_upper_ms = {
        name: values["max_ms"] for name, values in rank_spread.items()
    }
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "qualified_profile": not reasons,
        "reasons": reasons,
        "source": {
            "log": str(log_path),
            "control_service": str(control_service),
            "profile_service": str(profile_service),
        },
        "request": {
            "group_index": next(iter(selected_indices)),
            "prefill_tokens": expected_prefill_tokens,
            "expected_chunk_size": expected_chunk_size,
            "block_size": block_size,
            "forward_count": len(forwards),
            "tp_ranks": sorted(selected),
            "control_ttft_s": control_ttft_s,
            "profile_ttft_s": profile_ttft_s,
            "profile_overhead_fraction": overhead_fraction,
            "profile_overhead_limit_fraction": MAX_PROFILE_OVERHEAD_FRACTION,
            "worker_span_ms_by_rank": worker_spans,
            "worker_critical_path_ms": worker_critical_ms,
            "model_forward_ms_by_rank": model_by_rank,
            "model_critical_path_ms": model_critical_ms,
            "model_rank_spread_fraction": model_spread,
            "max_forward_model_rank_spread_fraction": (
                max_forward_model_spread),
            "model_share_of_profiled_ttft": (
                model_critical_ms / (profile_ttft_s * 1000)),
            "worker_share_of_profiled_ttft": (
                worker_critical_ms / (profile_ttft_s * 1000)),
            "control_output_sha256": control_request["output_sha256"],
        },
        "coverage": {
            "exclusive_model_regions_by_rank": exclusive_coverage_by_rank,
            "full_attention_subregions_by_rank": full_coverage_by_rank,
        },
        "full_attention": {
            "inclusive_ms_per_rank_mean": layer_full_ms,
            "subregions_ms_per_rank_mean": {
                name: regions.get(name, 0.0)
                for name in FULL_ATTN_SUBREGIONS
            },
            "kv_write_ms_per_rank_mean": kv_write_ms,
            "dense_prefill_ms_per_rank_mean": dense_prefill_ms,
            "paged_prefill_ms_per_rank_mean": paged_prefill_ms,
            "paged_segment_ms_per_rank_mean": paged_segment_ms,
            "attention_unattributed_ms_per_rank_mean": (
                attention_ms - kv_write_ms - dense_prefill_ms
                - paged_prefill_ms),
            "paged_unattributed_ms_per_rank_mean": (
                paged_prefill_ms - paged_segment_ms),
            "paged_share_of_model_work": (
                paged_segment_ms / model_mean_ms
                if model_mean_ms > 0 else None),
            "paged_critical_upper_share_of_control_ttft": (
                region_critical_upper_ms.get(
                    "paged_attn.prefix_pytorch", 0.0)
                / (control_ttft_s * 1000)),
        },
        "regions_ms_per_rank_mean": regions,
        "regions_critical_path_upper_bound_ms": region_critical_upper_ms,
        "rank_spread": rank_spread,
        "forwards": forwards,
    }


def _write_atomic(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--expected-prefill-tokens", type=int, required=True)
    parser.add_argument("--expected-processes", type=int, default=4)
    parser.add_argument("--expected-chunk-size", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--control-service", type=Path, required=True)
    parser.add_argument("--profile-service", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(
        args.log,
        args.expected_prefill_tokens,
        args.expected_processes,
        args.profile_service,
        args.control_service,
        args.expected_chunk_size,
        args.block_size,
    )
    _write_atomic(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qualified_profile"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
