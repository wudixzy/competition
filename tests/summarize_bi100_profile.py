#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


PROFILE_RE = re.compile(
    r"(?:VllmWorkerProcess pid=(?P<pid>\d+).*?)?"
    r"\[BI100_PROFILE\]\s+(?P<name>\S+)\s+(?P<ms>[0-9.]+)\s+ms")


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def parse_profile(path: Path, layers: int) -> dict[str, Any]:
    records: dict[str, list[tuple[int, str, float]]] = defaultdict(list)
    input_norm_count: dict[str, int] = defaultdict(int)

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = PROFILE_RE.search(line)
        if match is None:
            continue
        process = match.group("pid") or "driver"
        name = match.group("name")
        if name == "layer.input_norm":
            input_norm_count[process] += 1
        count = input_norm_count[process]
        if count == 0:
            raise ValueError(
                f"profile record before first layer.input_norm for {process}: {name}")
        forward_index = (count - 1) // layers
        records[process].append((forward_index, name, float(match.group("ms"))))

    if not records:
        raise ValueError(f"no BI100 profile records found in {path}")
    return {"records": records, "input_norm_count": input_norm_count}


def summarize(path: Path, layers: int, skip_prefill: int) -> dict[str, Any]:
    parsed = parse_profile(path, layers)
    records = parsed["records"]
    per_process: dict[str, Any] = {}
    decode_by_name: dict[str, list[float]] = defaultdict(list)

    for process, process_records in sorted(records.items()):
        by_forward: dict[int, dict[str, float]] = defaultdict(
            lambda: defaultdict(float))
        counts: dict[int, dict[str, int]] = defaultdict(
            lambda: defaultdict(int))
        for forward_index, name, milliseconds in process_records:
            by_forward[forward_index][name] += milliseconds
            counts[forward_index][name] += 1

        expected_forwards = parsed["input_norm_count"][process] // layers
        complete_forwards = [
            index for index in sorted(by_forward)
            if counts[index].get("layer.input_norm") == layers
        ]
        if len(complete_forwards) != expected_forwards:
            raise ValueError(
                f"incomplete profile for {process}: expected {expected_forwards} "
                f"forwards, found {len(complete_forwards)}")
        decode_forwards = complete_forwards[skip_prefill:]
        if not decode_forwards:
            raise ValueError(f"no decode forwards found for {process}")
        for index in decode_forwards:
            for name, milliseconds in by_forward[index].items():
                decode_by_name[name].append(milliseconds)

        per_process[process] = {
            "complete_forwards": len(complete_forwards),
            "prefill_forwards": complete_forwards[:skip_prefill],
            "decode_forwards": len(decode_forwards),
            "records": len(process_records),
        }

    region_summary = {}
    for name, values in sorted(decode_by_name.items()):
        region_summary[name] = {
            "samples": len(values),
            "mean_ms_per_token_per_rank": statistics.mean(values),
            "p50_ms_per_token_per_rank": percentile(values, 50),
            "p90_ms_per_token_per_rank": percentile(values, 90),
            "min_ms_per_token_per_rank": min(values),
            "max_ms_per_token_per_rank": max(values),
        }

    tracked_totals = []
    sample_count = min(len(values) for values in decode_by_name.values())
    ordered_names = sorted(decode_by_name)
    for index in range(sample_count):
        tracked_totals.append(sum(decode_by_name[name][index]
                                  for name in ordered_names))
    tracked_mean = statistics.mean(tracked_totals)
    for stats in region_summary.values():
        stats["share_of_tracked_mean"] = (
            stats["mean_ms_per_token_per_rank"] / tracked_mean)

    return {
        "log": str(path),
        "layers": layers,
        "skip_prefill_forwards": skip_prefill,
        "processes": per_process,
        "tracked_regions": ordered_names,
        "tracked_total": {
            "samples": len(tracked_totals),
            "mean_ms_per_token_per_rank": tracked_mean,
            "p50_ms_per_token_per_rank": percentile(tracked_totals, 50),
            "p90_ms_per_token_per_rank": percentile(tracked_totals, 90),
        },
        "regions": region_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize filtered BI100 TP decode profile logs.")
    parser.add_argument("log", type=Path)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--skip-prefill-forwards", type=int, default=1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = summarize(args.log, args.layers, args.skip_prefill_forwards)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
