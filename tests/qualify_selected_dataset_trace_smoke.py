#!/usr/bin/env python3
"""Qualify a privacy-safe 13-request live cache-trace smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_prefix_cache_trace as trace_analyzer  # noqa: E402


SCHEMA = "bi100-selected-dataset-cache-trace-smoke-v1"
VERSION = 1
EXPECTED_DATASET_SHA256 = (
    "dac6afc77621b51dbc09cfa046c008a1e51a779bb771edcb27cb6a686f8884c8"
)
EXPECTED_TURNS = (4, 4, 3, 2)
EXPECTED_REQUESTS = sum(EXPECTED_TURNS)
TRACE_MARKER = b"[BI100_CACHE_TRACE] "
REQUIRED_TRACE_FIELDS = {
    "version",
    "trace_session_sha256",
    "ordinal",
    "request_id_sha256",
    "prompt_tokens",
    "prompt_allocated_blocks",
    "block_size",
    "capacity_blocks",
    "gdn_policy",
    "raw_kv_contiguous_hit_blocks",
    "effective_gdn_hit_blocks",
    "gdn_admissions",
    "gdn_evictions",
    "ttft_s",
    "request_latency_s",
    "time_in_queue_s",
    "observed_effective_cached_tokens",
    "total_tokens",
    "allocated_blocks",
    "full_blocks",
    "hash_encoding",
    "block_hashes",
    "generated_tokens",
    "observed_input_tps",
}
OPTIONAL_TRACE_FIELDS = {"observed_output_tps", "qualification_trace"}
TRACE_INTERNAL_FIELDS = {"_hashes", "_prompt_full_blocks"}
POLICY_NAMES = {"off", "fine32", "admission64", "admission64_m1_29"}
CONTROL_POLICY_NAMES = {"off", "fine32", "admission64"}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _dataset_contract(dataset_bytes: bytes) -> tuple[list[str], list[str]]:
    reasons: list[str] = []
    strings: list[str] = []
    if _sha256(dataset_bytes) != EXPECTED_DATASET_SHA256:
        reasons.append("selected dataset SHA-256 differs from the frozen gate")
        return strings, reasons
    try:
        dataset = json.loads(dataset_bytes)
    except (TypeError, ValueError):
        reasons.append("selected dataset JSON is invalid")
        return strings, reasons
    if not isinstance(dataset, list):
        reasons.append("selected dataset root must be a list")
        return strings, reasons
    turns = []
    for item in dataset:
        if not isinstance(item, dict):
            reasons.append("selected dataset conversation is invalid")
            continue
        system = item.get("system_prompt", "")
        questions = item.get("user_questions")
        if not isinstance(system, str) or not isinstance(questions, list) \
                or not all(isinstance(value, str) for value in questions):
            reasons.append("selected dataset messages are invalid")
            continue
        turns.append(len(questions))
        if system:
            strings.append(system)
        strings.extend(questions)
    if tuple(turns) != EXPECTED_TURNS:
        reasons.append("selected dataset turn shape differs from the frozen gate")
    return strings, reasons


def _known_dataset_text_is_absent(
    log_bytes: bytes,
    dataset_strings: list[str],
) -> bool:
    for value in dataset_strings:
        if len(value.strip()) < 8:
            continue
        variants = {
            value,
            json.dumps(value, ensure_ascii=False)[1:-1],
            json.dumps(value, ensure_ascii=True)[1:-1],
        }
        for variant in variants:
            if variant.encode("utf-8") in log_bytes:
                return False
    return True


def _validate_action_rows(
    record: dict[str, Any],
    field: str,
    reasons: list[str],
) -> None:
    rows = record.get(field)
    if not isinstance(rows, list):
        reasons.append(f"trace {field} must be a list")
        return
    for row in rows:
        if (not isinstance(row, dict)
                or set(row) != {"block_count", "digest_base64", "reason"}
                or not isinstance(row.get("block_count"), int)
                or isinstance(row.get("block_count"), bool)
                or row.get("block_count", 0) <= 0
                or not isinstance(row.get("digest_base64"), str)
                or not isinstance(row.get("reason"), str)):
            reasons.append(f"trace {field} contains an invalid action")
            return


def qualify(
    *,
    records: list[dict[str, Any]],
    analysis: Any,
    replay: Any,
    log_bytes: bytes,
    dataset_bytes: bytes,
    source_names: dict[str, str],
) -> dict[str, Any]:
    reasons: list[str] = []
    dataset_strings, dataset_reasons = _dataset_contract(dataset_bytes)
    reasons.extend(dataset_reasons)
    known_dataset_text_absent = _known_dataset_text_is_absent(
        log_bytes, dataset_strings)
    if not known_dataset_text_absent:
        reasons.append("known raw dataset text appears in the service log")

    trace_lines = [
        line for line in log_bytes.splitlines(keepends=True)
        if TRACE_MARKER in line
    ]
    if len(trace_lines) != EXPECTED_REQUESTS:
        reasons.append(
            f"service log must contain {EXPECTED_REQUESTS} trace lines")

    if len(records) != EXPECTED_REQUESTS:
        reasons.append(f"trace must contain {EXPECTED_REQUESTS} records")
    sessions = {
        record.get("trace_session_sha256") for record in records
        if isinstance(record, dict)
    }
    if len(sessions) != 1:
        reasons.append("trace records must belong to one runtime session")
    expected_ordinals = list(range(1, EXPECTED_REQUESTS + 1))
    if [record.get("ordinal") for record in records] != expected_ordinals:
        reasons.append("trace ordinals must be exactly 1 through 13")

    for record in records:
        if not isinstance(record, dict):
            reasons.append("trace record must be an object")
            continue
        public_fields = set(record) - TRACE_INTERNAL_FIELDS
        missing = REQUIRED_TRACE_FIELDS - public_fields
        unexpected = public_fields - REQUIRED_TRACE_FIELDS \
            - OPTIONAL_TRACE_FIELDS
        if missing:
            reasons.append(f"trace record fields missing: {sorted(missing)}")
        if unexpected:
            reasons.append(
                f"trace record fields unexpected: {sorted(unexpected)}")
        if record.get("version") != 4:
            reasons.append("trace record version must be 4")
        if record.get("gdn_policy") != "admission64":
            reasons.append("trace record GDN policy must be admission64")
        if record.get("block_size") != 16:
            reasons.append("trace record block size must be 16")
        for field in (
            "ttft_s",
            "request_latency_s",
            "time_in_queue_s",
            "observed_input_tps",
        ):
            if not _finite_nonnegative(record.get(field)):
                reasons.append(f"trace record {field} is incomplete")
        if record.get("generated_tokens", 0) > 1 and not \
                _finite_nonnegative(record.get("observed_output_tps")):
            reasons.append("trace record observed_output_tps is incomplete")
        for field in (
            "raw_kv_contiguous_hit_blocks",
            "effective_gdn_hit_blocks",
            "observed_effective_cached_tokens",
        ):
            value = record.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                reasons.append(f"trace record {field} must be non-negative")
        _validate_action_rows(record, "gdn_admissions", reasons)
        _validate_action_rows(record, "gdn_evictions", reasons)

    replay_aggregate: dict[str, Any] = {}
    if (not isinstance(replay, dict)
            or replay.get("schema") != "bi100-selected-dataset-replay-v1"
            or replay.get("validation", {}).get("complete_replay") is not True
            or replay.get("validation", {}).get("all_successful") is not True
            or replay.get("dataset", {}).get("sha256")
            != EXPECTED_DATASET_SHA256
            or replay.get("dataset", {}).get("turn_count")
            != EXPECTED_REQUESTS):
        reasons.append("selected replay source is invalid")
    else:
        replay_aggregate = replay.get("aggregate") or {}

    trace_prompt_tokens = sum(
        int(record.get("prompt_tokens", 0)) for record in records)
    trace_cached_tokens = sum(
        int(record.get("observed_effective_cached_tokens", 0))
        for record in records)
    trace_generated_tokens = sum(
        int(record.get("generated_tokens", 0)) for record in records)
    if replay_aggregate:
        if trace_prompt_tokens != replay_aggregate.get("prompt_tokens"):
            reasons.append("trace prompt-token aggregate differs from replay")
        if trace_cached_tokens != replay_aggregate.get("cached_tokens"):
            reasons.append("trace cached-token aggregate differs from replay")
        if trace_generated_tokens != replay_aggregate.get("completion_tokens"):
            reasons.append("trace generated-token aggregate differs from replay")

    if not isinstance(analysis, dict):
        reasons.append("trace analysis must be an object")
        analysis = {}
    session = next(iter(sessions)) if len(sessions) == 1 else None
    source_logs = analysis.get("source_logs")
    if (analysis.get("requests") != EXPECTED_REQUESTS
            or analysis.get("trace_version") != 4
            or analysis.get("qualification_trace") is not False
            or analysis.get("trace_session_sha256") != session
            or analysis.get("trace_ordinals") != {
                "first": 1,
                "last": EXPECTED_REQUESTS,
                "contiguous": True,
            }
            or analysis.get("prompt_tokens") != trace_prompt_tokens
            or analysis.get("generated_tokens") != trace_generated_tokens):
        reasons.append("trace analysis contract is invalid")
    if "qualification" in analysis:
        reasons.append("13-request trace analysis must not contain qualification")
    if (not isinstance(source_logs, list) or len(source_logs) != 1
            or source_logs[0].get("sha256") != _sha256(log_bytes)):
        reasons.append("trace analysis is not bound to the service log")
    policy_metrics = analysis.get("policy_metrics")
    control_metrics = analysis.get("control_policy_metrics")
    if not isinstance(policy_metrics, dict) or set(policy_metrics) != POLICY_NAMES:
        reasons.append("trace analysis policy set is invalid")
        policy_metrics = {}
    if (not isinstance(control_metrics, dict)
            or set(control_metrics) != CONTROL_POLICY_NAMES):
        reasons.append("trace analysis control-policy set is invalid")
    for name, metrics in policy_metrics.items():
        if (not isinstance(metrics, dict)
                or metrics.get("per_request_timing_projection_complete") is not True
                or not _finite_nonnegative(metrics.get("projected_ttft_p90_s"))
                or not _finite_nonnegative(
                    metrics.get("projected_sequential_wall_s"))):
            reasons.append(f"trace analysis timing is incomplete for {name}")

    trace_line_bytes = sum(len(line) for line in trace_lines)
    full_blocks = sum(int(record.get("full_blocks", 0)) for record in records)
    hash_payload_bytes = sum(
        len(str(record.get("block_hashes", "")).encode("ascii", "ignore"))
        for record in records)
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "qualified": not reasons,
        "reasons": reasons,
        "scope": "selected-13-request-trace-smoke-not-881-qualification",
        "qualification_authorized": False,
        "trace": {
            "version": 4,
            "requests": len(records),
            "trace_session_sha256": session,
            "ordinal_first": records[0].get("ordinal") if records else None,
            "ordinal_last": records[-1].get("ordinal") if records else None,
            "prompt_tokens": trace_prompt_tokens,
            "cached_tokens": trace_cached_tokens,
            "generated_tokens": trace_generated_tokens,
            "full_blocks": full_blocks,
            "all_policy_timing_complete": all(
                isinstance(metrics, dict)
                and metrics.get("per_request_timing_projection_complete") is True
                for metrics in policy_metrics.values()
            ) if policy_metrics else False,
        },
        "privacy": {
            "known_dataset_text_absent": known_dataset_text_absent,
            "trace_field_allowlist_passed": not any(
                reason.startswith("trace record fields") for reason in reasons),
            "contains_raw_messages": False if not reasons else None,
            "contains_raw_model_output_field": False,
        },
        "size": {
            "service_log_bytes": len(log_bytes),
            "trace_line_bytes": trace_line_bytes,
            "hash_payload_base64_bytes": hash_payload_bytes,
            "mean_trace_bytes_per_request": (
                trace_line_bytes / len(records) if records else None),
            "mean_trace_bytes_per_final_full_block": (
                trace_line_bytes / full_blocks if full_blocks else None),
            "linear_881_request_estimate_bytes": (
                round(trace_line_bytes / len(records) * 881)
                if records else None),
            "estimate_scope": "linear selected-sample diagnostic only",
        },
        "source_sha256": {
            "service_log": _sha256(log_bytes),
            "analysis": source_names["analysis"],
            "replay": source_names["replay"],
            "dataset": _sha256(dataset_bytes),
        },
    }


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=True, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    source_names = {}
    try:
        log_bytes = args.log.read_bytes()
        analysis_bytes = args.analysis.read_bytes()
        replay_bytes = args.replay.read_bytes()
        dataset_bytes = args.dataset.read_bytes()
        source_names = {
            "analysis": _sha256(analysis_bytes),
            "replay": _sha256(replay_bytes),
        }
        report = qualify(
            records=trace_analyzer.read([str(args.log)]),
            analysis=json.loads(analysis_bytes),
            replay=json.loads(replay_bytes),
            log_bytes=log_bytes,
            dataset_bytes=dataset_bytes,
            source_names=source_names,
        )
    except (OSError, TypeError, ValueError) as error:
        report = {
            "schema": SCHEMA,
            "version": VERSION,
            "qualified": False,
            "reasons": [
                f"trace smoke input validation failed: {type(error).__name__}: "
                f"{error}"
            ],
            "scope": "selected-13-request-trace-smoke-not-881-qualification",
            "qualification_authorized": False,
        }
    _atomic_write(args.out, report)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
