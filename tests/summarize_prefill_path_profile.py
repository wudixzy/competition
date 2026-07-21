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
SCHEMA = "bi100-m1-48-prefill-path-profile-v1"
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
        records[process].append(payload)
    if not records:
        raise ValueError(f"no BI100_PROFILE_EVENT records found in {path}")
    for process, process_records in records.items():
        process_records.sort(key=lambda item: item["forward_index"])
        indices = [item["forward_index"] for item in process_records]
        expected = list(range(indices[0], indices[0] + len(indices)))
        if indices != expected:
            raise ValueError(
                f"non-contiguous forward indices for {process}: {indices}")
    return dict(records)


def split_requests(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    requests: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    saw_decode = False
    for record in records:
        metadata = record.get("metadata") or {}
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


def _mean_rank_regions(
    process_records: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    rank_totals = {
        process: _aggregate_regions(records)
        for process, records in process_records.items()
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
    process_records: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    counts = {len(records) for records in process_records.values()}
    if len(counts) != 1:
        raise ValueError(f"forward-count mismatch by process: {counts}")
    rows = []
    for offset in range(next(iter(counts))):
        selected = {
            process: records[offset]
            for process, records in process_records.items()
        }
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
        regions, _ = _mean_rank_regions({
            process: [record] for process, record in selected.items()
        })
        host_model = [record.get("host_model_to_flush_ms")
                      for record in selected.values()]
        if any(not _finite(value) or value < 0 for value in host_model):
            raise ValueError(f"invalid host model duration at offset {offset}")
        host_gaps = [record.get("host_gap_since_previous_flush_ms")
                     for record in selected.values()]
        finite_gaps = [float(value) for value in host_gaps if _finite(value)]
        rows.append({
            "offset": offset,
            "metadata": next(iter(selected.values())).get("metadata") or {},
            "regions_ms_per_rank": regions,
            "counters": json.loads(next(iter(counter_values))),
            "host_model_to_flush_ms_mean": statistics.mean(host_model),
            "host_gap_since_previous_flush_ms_mean": (
                statistics.mean(finite_gaps) if finite_gaps else None),
        })
    return rows


def _load_client_elapsed(path: Path | None) -> float | None:
    if path is None:
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    first = report.get("first") if isinstance(report, dict) else None
    elapsed = first.get("elapsed_s") if isinstance(first, dict) else None
    if not _finite(elapsed) or elapsed <= 0:
        raise ValueError("client summary first.elapsed_s is invalid")
    return float(elapsed)


def summarize(
    log_path: Path,
    expected_prefill_tokens: int,
    expected_processes: int = 4,
    client_summary: Path | None = None,
    candidate_core_speedup: float | None = None,
) -> dict[str, Any]:
    if expected_prefill_tokens <= 0:
        raise ValueError("expected_prefill_tokens must be positive")
    if (candidate_core_speedup is not None
            and (not _finite(candidate_core_speedup)
                 or candidate_core_speedup <= 1)):
        raise ValueError("candidate_core_speedup must be finite and above one")

    parsed = parse_log(log_path)
    if len(parsed) != expected_processes:
        raise ValueError(
            f"expected {expected_processes} profile processes, found {len(parsed)}")
    grouped = {process: split_requests(records)
               for process, records in parsed.items()}
    selected: dict[str, list[dict[str, Any]]] = {}
    selected_indices = set()
    for process, requests in grouped.items():
        matches = [index for index, request in enumerate(requests)
                   if _prefill_tokens(request) == expected_prefill_tokens]
        if len(matches) != 1:
            raise ValueError(
                f"{process} has {len(matches)} requests with exactly "
                f"{expected_prefill_tokens} prefill tokens")
        selected_indices.add(matches[0])
        selected[process] = requests[matches[0]]
    if len(selected_indices) != 1:
        raise ValueError(f"selected request index differs by process: {selected_indices}")

    regions, rank_spread = _mean_rank_regions(selected)
    forwards = _aligned_forwards(selected)
    model_ms = regions.get("model.forward", 0.0)
    layer_full_ms = regions.get("layer.full_attn", 0.0)
    attention_ms = regions.get("full_attn.attention", 0.0)
    paged_ms = regions.get("paged_attn.prefix_pytorch", 0.0)
    exclusive_ms = sum(regions.get(name, 0.0)
                       for name in EXCLUSIVE_MODEL_REGIONS)
    full_subregion_ms = sum(regions.get(name, 0.0)
                            for name in FULL_ATTN_SUBREGIONS)
    client_elapsed_s = _load_client_elapsed(client_summary)

    reasons = []
    if model_ms <= 0:
        reasons.append("model.forward duration is missing")
    if layer_full_ms <= 0:
        reasons.append("layer.full_attn duration is missing")
    if attention_ms <= 0:
        reasons.append("full_attn.attention duration is missing")
    if paged_ms <= 0:
        reasons.append("paged_attn.prefix_pytorch duration is missing")
    if paged_ms > attention_ms * 1.02:
        reasons.append("paged attention exceeds its inclusive attention region")
    exclusive_coverage = exclusive_ms / model_ms if model_ms > 0 else None
    full_subregion_coverage = (
        full_subregion_ms / layer_full_ms if layer_full_ms > 0 else None)
    if (exclusive_coverage is None
            or not 0.97 <= exclusive_coverage <= 1.03):
        reasons.append("exclusive model regions do not close model.forward")
    if (full_subregion_coverage is None
            or not 0.97 <= full_subregion_coverage <= 1.03):
        reasons.append("full-attention subregions do not close layer.full_attn")
    service_model_coverage = (
        model_ms / 1000 / client_elapsed_s
        if client_elapsed_s is not None else None)
    if (service_model_coverage is not None
            and not 0.90 <= service_model_coverage <= 1.05):
        reasons.append("model.forward does not close the client cold latency")

    paged_share = paged_ms / model_ms if model_ms > 0 else None
    projected_improvement = None
    if paged_share is not None and candidate_core_speedup is not None:
        projected_improvement = (
            paged_share * (1.0 - 1.0 / candidate_core_speedup))

    return {
        "schema": SCHEMA,
        "version": 1,
        "qualified_profile": not reasons,
        "reasons": reasons,
        "source": {
            "log": str(log_path),
            "client_summary": (
                str(client_summary) if client_summary is not None else None),
        },
        "request": {
            "group_index": next(iter(selected_indices)),
            "prefill_tokens": expected_prefill_tokens,
            "processes": sorted(selected),
            "forward_count": len(forwards),
            "client_elapsed_s": client_elapsed_s,
            "model_forward_ms_per_rank": model_ms,
            "service_model_coverage": service_model_coverage,
        },
        "coverage": {
            "exclusive_model_regions": exclusive_coverage,
            "full_attention_subregions": full_subregion_coverage,
        },
        "full_attention": {
            "inclusive_ms_per_rank": layer_full_ms,
            "subregions_ms_per_rank": {
                name: regions.get(name, 0.0)
                for name in FULL_ATTN_SUBREGIONS
            },
            "paged_prefix_ms_per_rank": paged_ms,
            "paged_share_of_model_forward": paged_share,
            "attention_nonpaged_ms_per_rank": max(0.0, attention_ms - paged_ms),
            "outside_attention_ms_per_rank": max(0.0, layer_full_ms - attention_ms),
            "candidate_core_speedup": candidate_core_speedup,
            "amdahl_projected_service_improvement": projected_improvement,
        },
        "regions_ms_per_rank": regions,
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
    parser.add_argument("--client-summary", type=Path)
    parser.add_argument("--candidate-core-speedup", type=float)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(
        args.log,
        args.expected_prefill_tokens,
        args.expected_processes,
        args.client_summary,
        args.candidate_core_speedup,
    )
    _write_atomic(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qualified_profile"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
