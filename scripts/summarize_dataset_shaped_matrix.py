#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence


REQUEST_RE = re.compile(
    r"^(4096|7800|16000)_pair([1-3])_(cold|warm)\.json$")


def percentile(values: Sequence[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def load_requests(directory: Path) -> list[dict[str, Any]]:
    requests = []
    for path in sorted((directory / "requests").glob("*.json")):
        match = REQUEST_RE.match(path.name)
        if match is None:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        timing = data.get("timing") or {}
        target, pair, phase = match.groups()
        requests.append({
            "path": path.name,
            "target": int(target),
            "pair": int(pair),
            "phase": phase,
            "ok": bool(timing.get("ok")),
            "prompt_tokens": int(timing.get("prompt_tokens") or 0),
            "cached_tokens": int(timing.get("cached_tokens") or 0),
            "completion_tokens": int(timing.get("completion_tokens") or 0),
            "ttft_s": float(timing.get("ttft_s") or 0.0),
            "latency_s": float(timing.get("latency_s") or 0.0),
            "output_tps": float(timing.get("output_tps_decode") or 0.0),
        })
    return requests


def summarize(directory: Path) -> dict[str, Any]:
    requests = load_requests(directory)
    expected = {
        (target, pair, phase)
        for target in (4096, 7800, 16000)
        for pair in (1, 2, 3)
        for phase in ("cold", "warm")
    }
    observed = {
        (item["target"], item["pair"], item["phase"])
        for item in requests
    }
    cold = [item for item in requests if item["phase"] == "cold"]
    warm = [item for item in requests if item["phase"] == "warm"]
    for item in requests:
        ttft_s = item["ttft_s"]
        item["input_tps"] = (
            item["prompt_tokens"] / ttft_s if ttft_s > 0 else 0.0)
        item["cache_tps"] = (
            item["cached_tokens"] / ttft_s if ttft_s > 0 else 0.0)

    output_tps_p10 = percentile(
        [item["output_tps"] for item in requests if item["ok"]], 10)
    cold_ttft = sum(item["ttft_s"] for item in cold)
    warm_ttft = sum(item["ttft_s"] for item in warm)
    input_tps_aggregate = (
        sum(item["prompt_tokens"] for item in cold) / cold_ttft
        if cold_ttft > 0 else 0.0)
    cache_tps_aggregate = (
        sum(item["cached_tokens"] for item in warm) / warm_ttft
        if warm_ttft > 0 else 0.0)
    prompt_tokens = sum(item["prompt_tokens"] for item in requests)
    cached_tokens = sum(item["cached_tokens"] for item in requests)
    weighted = (
        output_tps_p10 * 16.796
        + input_tps_aggregate * 2.799
        + cache_tps_aggregate * 0.56)

    by_length = {}
    for target in (4096, 7800, 16000):
        target_cold = [item for item in cold if item["target"] == target]
        target_warm = [item for item in warm if item["target"] == target]
        by_length[str(target)] = {
            "cold_ttft_p90_s": percentile(
                [item["ttft_s"] for item in target_cold], 90),
            "warm_ttft_p90_s": percentile(
                [item["ttft_s"] for item in target_warm], 90),
            "cold_input_tps_p10": percentile(
                [item["input_tps"] for item in target_cold], 10),
            "warm_cache_tps_p10": percentile(
                [item["cache_tps"] for item in target_warm], 10),
            "output_tps_p10": percentile(
                [item["output_tps"] for item in target_cold + target_warm], 10),
            "warm_cached_tokens": [
                item["cached_tokens"] for item in target_warm],
        }

    return {
        "scope": {"requests": len(requests), "expected_requests": 18},
        "validation": {
            "complete_matrix": observed == expected,
            "success_rate": (
                sum(item["ok"] for item in requests) / len(requests)
                if requests else 0.0),
            "cold_cached_zero": all(
                item["cached_tokens"] == 0 for item in cold),
            "token_count_match": all(
                item["prompt_tokens"] == item["target"] for item in requests),
        },
        "aggregate": {
            "output_tps_p10": output_tps_p10,
            "input_tps_aggregate": input_tps_aggregate,
            "cache_tps_aggregate": cache_tps_aggregate,
            "ttft_p90_all_s": percentile(
                [item["ttft_s"] for item in requests], 90),
            "ttft_p90_cold_s": percentile(
                [item["ttft_s"] for item in cold], 90),
            "ttft_p90_warm_s": percentile(
                [item["ttft_s"] for item in warm], 90),
            "cache_hit_rate": (
                cached_tokens / prompt_tokens if prompt_tokens else 0.0),
            "weighted_score": weighted,
        },
        "by_length": by_length,
        "requests": requests,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.directory)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["validation"]["complete_matrix"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
