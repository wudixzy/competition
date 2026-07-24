#!/usr/bin/env python3
"""Qualify a privacy-redacted replay of the frozen selected dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence


SOURCE_SCHEMA = "bi100-selected-dataset-replay-v1"
SCHEMA = "bi100-selected-dataset-qualification-v1"
VERSION = 1
EXPECTED_DATASET_SHA256 = (
    "dac6afc77621b51dbc09cfa046c008a1e51a779bb771edcb27cb6a686f8884c8"
)
EXPECTED_CONVERSATION_TURNS = (4, 4, 3, 2)
EXPECTED_TURNS = sum(EXPECTED_CONVERSATION_TURNS)
TURN_FIELDS = (
    "conversation",
    "turn",
    "request_message_count",
    "ok",
    "error_kind",
    "finish_reason",
    "prompt_tokens",
    "cached_tokens",
    "uncached_prompt_tokens",
    "completion_tokens",
    "ttft_s",
    "latency_s",
    "output_tps_decode",
    "content_sha256",
)
AGGREGATE_FIELDS = (
    "wall_s",
    "ttft_p90_s",
    "output_tps_p10",
    "cache_hit_rate",
    "input_tps_residual_proxy",
    "cache_tps_proxy",
    "weighted_score_proxy",
    "prompt_tokens",
    "cached_tokens",
    "uncached_prompt_tokens",
    "completion_tokens",
)


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _percentile(values: Sequence[float], percent: float) -> float:
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


def _close(observed: Any, expected: float) -> bool:
    return _finite(observed) and math.isclose(
        float(observed), expected, rel_tol=1e-9, abs_tol=1e-9)


def qualify(source: Any) -> dict[str, Any]:
    reasons: list[str] = []
    if not isinstance(source, dict):
        return {
            "schema": SCHEMA,
            "version": VERSION,
            "qualified": False,
            "reasons": ["source must be an object"],
        }
    if source.get("schema") != SOURCE_SCHEMA:
        reasons.append("source schema is invalid")
    label = source.get("label")
    if not isinstance(label, str) or not label:
        reasons.append("label must be a nonempty string")

    dataset = source.get("dataset")
    if not isinstance(dataset, dict):
        dataset = {}
        reasons.append("dataset contract is missing")
    expected_dataset = {
        "path_name": "chat_dataset_v0.json",
        "sha256": EXPECTED_DATASET_SHA256,
        "conversation_count": len(EXPECTED_CONVERSATION_TURNS),
        "turn_count": EXPECTED_TURNS,
    }
    if dataset != expected_dataset:
        reasons.append("dataset contract differs from the frozen selection")

    privacy = source.get("privacy")
    expected_privacy = {
        "contains_raw_messages": False,
        "contains_raw_model_output": False,
    }
    if privacy != expected_privacy:
        reasons.append("privacy declaration is invalid")

    validation = source.get("validation")
    expected_validation = {
        "complete_replay": True,
        "success_rate": 1.0,
        "all_successful": True,
    }
    if validation != expected_validation:
        reasons.append("source replay is incomplete or unsuccessful")

    expected_order = [
        (conversation, turn)
        for conversation, count in enumerate(EXPECTED_CONVERSATION_TURNS)
        for turn in range(count)
    ]
    turns = source.get("turns")
    if not isinstance(turns, list) or len(turns) != EXPECTED_TURNS:
        reasons.append("turn set differs from the frozen replay")
        turns = []
    safe_turns: list[dict[str, Any]] = []
    for index, expected_identity in enumerate(expected_order):
        if index >= len(turns) or not isinstance(turns[index], dict):
            reasons.append(f"turn {index} is missing or invalid")
            continue
        row = {field: turns[index].get(field) for field in TURN_FIELDS}
        safe_turns.append(row)
        if (row["conversation"], row["turn"]) != expected_identity:
            reasons.append(f"turn {index} identity is invalid")
        if row["request_message_count"] != 2 + 2 * expected_identity[1]:
            reasons.append(f"turn {index} message count is invalid")
        if row["ok"] is not True or row["error_kind"] != "":
            reasons.append(f"turn {index} request failed")
        if not isinstance(row["finish_reason"], str):
            reasons.append(f"turn {index} finish reason is invalid")
        prompt = row["prompt_tokens"]
        cached = row["cached_tokens"]
        uncached = row["uncached_prompt_tokens"]
        completion = row["completion_tokens"]
        if not _is_integer(prompt) or prompt <= 0:
            reasons.append(f"turn {index} prompt tokens are invalid")
        if (not _is_integer(cached) or not _is_integer(uncached)
                or _is_integer(prompt)
                and (cached < 0 or cached > prompt
                     or uncached != prompt - cached)):
            reasons.append(f"turn {index} cache accounting is invalid")
        if not _is_integer(completion) or completion <= 0:
            reasons.append(f"turn {index} completion tokens are invalid")
        ttft = row["ttft_s"]
        latency = row["latency_s"]
        output_tps = row["output_tps_decode"]
        if not _finite(ttft) or ttft <= 0:
            reasons.append(f"turn {index} TTFT is invalid")
        if (not _finite(latency) or latency <= 0
                or _finite(ttft) and latency < ttft):
            reasons.append(f"turn {index} latency is invalid")
        if not _finite(output_tps) or output_tps <= 0:
            reasons.append(f"turn {index} output TPS is invalid")
        if not _digest(row["content_sha256"]):
            reasons.append(f"turn {index} output digest is invalid")

    aggregate = source.get("aggregate")
    if not isinstance(aggregate, dict):
        aggregate = {}
        reasons.append("aggregate metrics are missing")
    safe_aggregate = {field: aggregate.get(field) for field in AGGREGATE_FIELDS}
    if len(safe_turns) == EXPECTED_TURNS:
        ttfts = [float(row["ttft_s"]) for row in safe_turns
                 if _finite(row["ttft_s"])]
        output_rates = [float(row["output_tps_decode"]) for row in safe_turns
                        if _finite(row["output_tps_decode"])]
        prompt_total = sum(row["prompt_tokens"] for row in safe_turns
                           if _is_integer(row["prompt_tokens"]))
        cached_total = sum(row["cached_tokens"] for row in safe_turns
                           if _is_integer(row["cached_tokens"]))
        uncached_total = sum(row["uncached_prompt_tokens"] for row in safe_turns
                             if _is_integer(row["uncached_prompt_tokens"]))
        completion_total = sum(row["completion_tokens"] for row in safe_turns
                               if _is_integer(row["completion_tokens"]))
        ttft_total = sum(ttfts)
        output_p10 = _percentile(output_rates, 10) if output_rates else 0.0
        input_proxy = uncached_total / ttft_total if ttft_total else 0.0
        cache_proxy = cached_total / ttft_total if ttft_total else 0.0
        expected_metrics = {
            "ttft_p90_s": _percentile(ttfts, 90) if ttfts else 0.0,
            "output_tps_p10": output_p10,
            "cache_hit_rate": cached_total / prompt_total if prompt_total else 0.0,
            "input_tps_residual_proxy": input_proxy,
            "cache_tps_proxy": cache_proxy,
            "weighted_score_proxy": (
                output_p10 * 16.796 + input_proxy * 2.799
                + cache_proxy * 0.56
            ),
        }
        expected_counts = {
            "prompt_tokens": prompt_total,
            "cached_tokens": cached_total,
            "uncached_prompt_tokens": uncached_total,
            "completion_tokens": completion_total,
        }
        for field, expected in expected_metrics.items():
            if not _close(safe_aggregate[field], expected):
                reasons.append(f"aggregate {field} is inconsistent")
        for field, expected in expected_counts.items():
            if safe_aggregate[field] != expected:
                reasons.append(f"aggregate {field} is inconsistent")
        wall_s = safe_aggregate["wall_s"]
        latency_total = sum(
            float(row["latency_s"]) for row in safe_turns
            if _finite(row["latency_s"]))
        if not _finite(wall_s) or wall_s < latency_total:
            reasons.append("aggregate wall_s is invalid")

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "qualified": not reasons,
        "reasons": reasons,
        "scope": "selected-13-turn-supplemental-not-official-score",
        "label": label,
        "dataset": expected_dataset,
        "aggregate": safe_aggregate,
        "turns": safe_turns,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source_bytes = args.source.read_bytes()
    report = qualify(json.loads(source_bytes))
    report["source_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
