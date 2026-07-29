#!/usr/bin/env python3
"""Qualify privacy-safe TP4 real-activation shadow-reference reports."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


REPORT_SCHEMA = "bi100-fused-prefill-real-activation-shadow-v1"
RESULT_SCHEMA = "bi100-fused-prefill-shadow-qualification-v1"
CONTRACT_SCHEMA = "bi100-layered-quality-gate-contract-v2"
Json = dict[str, Any]


def _atomic_write(path: Path, value: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=True, indent=2,
                      sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _base_result(source_revision: str, runtime_identity: str,
                 run_id: str) -> Json:
    return {
        "schema": RESULT_SCHEMA,
        "version": 1,
        "source_revision": source_revision,
        "runtime_identity": runtime_identity,
        "run_id": run_id,
        "qualified": False,
        "promotion_authorized": False,
        "privacy": {
            "contains_prompts": False,
            "contains_model_outputs": False,
            "contains_tensor_values": False,
            "contains_token_ids": False,
            "contains_credentials": False,
        },
    }


def qualify(
    reports: list[Any],
    contract: Any,
    *,
    run_id: str,
    source_revision: str,
    runtime_identity: str,
) -> Json:
    result = _base_result(source_revision, runtime_identity, run_id)
    invalid_reasons = []
    numeric_failures = []
    if (
        not isinstance(contract, dict)
        or contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("version") != 2
        or not isinstance(contract.get("operator_shadow_reference"), dict)
    ):
        invalid_reasons.append("v2 contract identity is invalid")
        shadow = {}
    else:
        shadow = contract["operator_shadow_reference"]
    required_ranks = shadow.get("required_ranks", [])
    contexts = shadow.get("required_minimum_context_tokens", [])
    minimum_per_context = shadow.get(
        "minimum_observations_per_context_per_rank")
    relative_l2_limit = shadow.get("maximum_relative_l2")
    max_abs_limit = shadow.get("maximum_absolute_error")
    if (
        required_ranks != [0, 1, 2, 3]
        or contexts != [49152, 114688]
        or minimum_per_context != 2
        or not _finite(relative_l2_limit)
        or not _finite(max_abs_limit)
    ):
        invalid_reasons.append("operator shadow contract is invalid")

    rank_reports: dict[int, Json] = {}
    all_records = []
    expected_fields = {
        "schema", "version", "run_id", "pid", "rank", "status",
        "selection", "thresholds", "observations", "records", "privacy",
    }
    for ordinal, value in enumerate(reports):
        label = f"report[{ordinal}]"
        if not isinstance(value, dict) or set(value) != expected_fields:
            invalid_reasons.append(f"{label}: fields are invalid")
            continue
        rank = value.get("rank")
        if (
            value.get("schema") != REPORT_SCHEMA
            or value.get("version") != 1
            or value.get("run_id") != run_id
            or not isinstance(value.get("pid"), int)
            or isinstance(value["pid"], bool)
            or value["pid"] <= 0
            or not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank not in required_ranks
        ):
            invalid_reasons.append(f"{label}: identity is invalid")
            continue
        if rank in rank_reports:
            invalid_reasons.append(f"rank {rank}: duplicate report")
            continue
        rank_reports[rank] = value
        if value.get("selection") != {
            "minimum_context_tokens": contexts,
            "max_calls_per_context": minimum_per_context,
        }:
            invalid_reasons.append(f"rank {rank}: selection is invalid")
        if value.get("thresholds") != {
            "require_finite_candidate": True,
            "require_finite_reference": True,
            "maximum_relative_l2": relative_l2_limit,
            "maximum_absolute_error": max_abs_limit,
        }:
            invalid_reasons.append(f"rank {rank}: thresholds are invalid")
        privacy = value.get("privacy")
        if (
            not isinstance(privacy, dict)
            or set(privacy) != set(result["privacy"])
            or any(item is not False for item in privacy.values())
        ):
            invalid_reasons.append(f"rank {rank}: privacy contract is invalid")
        records = value.get("records")
        if not isinstance(records, list):
            invalid_reasons.append(f"rank {rank}: records are invalid")
            continue
        bucket_counts = {context: 0 for context in contexts}
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                invalid_reasons.append(
                    f"rank {rank} record {index}: structure is invalid")
                continue
            required_record_fields = {
                "index", "status", "bucket_min_context_tokens",
                "context_tokens", "query_shape", "query_heads", "kv_heads",
                "head_dim", "block_size", "candidate_finite",
                "reference_finite", "relative_l2", "max_abs",
                "error_stage", "error_type",
            }
            if set(record) != required_record_fields or record.get("index") != index:
                invalid_reasons.append(
                    f"rank {rank} record {index}: fields are invalid")
                continue
            bucket = record.get("bucket_min_context_tokens")
            context_tokens = record.get("context_tokens")
            shape = record.get("query_shape")
            bucket_index = (
                contexts.index(bucket) if bucket in contexts else -1)
            upper_bound = (
                contexts[bucket_index + 1]
                if 0 <= bucket_index < len(contexts) - 1 else None)
            if (
                bucket not in bucket_counts
                or not isinstance(context_tokens, int)
                or isinstance(context_tokens, bool)
                or context_tokens < bucket
                or (
                    upper_bound is not None
                    and context_tokens >= upper_bound
                )
                or not isinstance(shape, list)
                or len(shape) != 3
                or not isinstance(shape[0], int)
                or not 16 < shape[0] <= 8192
                or shape[1:] != [4, 256]
                or record.get("query_heads") != 4
                or record.get("kv_heads") != 1
                or record.get("head_dim") != 256
                or record.get("block_size") != 16
            ):
                invalid_reasons.append(
                    f"rank {rank} record {index}: production shape is invalid")
                continue
            bucket_counts[bucket] += 1
            status = record.get("status")
            if status == "fail":
                numeric_failures.append(
                    f"rank {rank} record {index}: hard numeric failure")
            elif status != "pass":
                invalid_reasons.append(
                    f"rank {rank} record {index}: result is incomplete")
            if (
                record.get("candidate_finite") is not True
                or record.get("reference_finite") is not True
                or not _finite(record.get("relative_l2"))
                or not _finite(record.get("max_abs"))
                or (
                    _finite(record.get("relative_l2"))
                    and float(record["relative_l2"]) < 0.0
                )
                or (
                    _finite(record.get("max_abs"))
                    and float(record["max_abs"]) < 0.0
                )
            ):
                if status == "fail":
                    numeric_failures.append(
                        f"rank {rank} record {index}: non-finite result")
                else:
                    invalid_reasons.append(
                        f"rank {rank} record {index}: metrics are invalid")
                continue
            if float(record["relative_l2"]) > float(relative_l2_limit):
                numeric_failures.append(
                    f"rank {rank} record {index}: relative L2 exceeds limit")
            if float(record["max_abs"]) > float(max_abs_limit):
                numeric_failures.append(
                    f"rank {rank} record {index}: max abs exceeds limit")
            all_records.append(record)
        for bucket, count in bucket_counts.items():
            if count != minimum_per_context:
                invalid_reasons.append(
                    f"rank {rank}: context bucket {bucket} has {count} records")
        if value.get("status") == "fail":
            numeric_failures.append(f"rank {rank}: report status is fail")
        elif value.get("status") != "pass":
            invalid_reasons.append(f"rank {rank}: report is incomplete")

    missing_ranks = sorted(set(required_ranks) - set(rank_reports))
    if missing_ranks:
        invalid_reasons.append(f"missing rank reports: {missing_ranks}")
    if numeric_failures:
        status = "fail"
    elif invalid_reasons:
        status = "invalid"
    else:
        status = "pass"
    result.update({
        "status": status,
        "qualified": status == "pass",
        "invalid_reasons": invalid_reasons,
        "numeric_failures": sorted(set(numeric_failures)),
        "rank_count": len(rank_reports),
        "observation_count": len(all_records),
        "maximum_relative_l2": (
            max(float(record["relative_l2"]) for record in all_records)
            if all_records else None),
        "maximum_absolute_error": (
            max(float(record["max_abs"]) for record in all_records)
            if all_records else None),
        "thresholds": {
            "maximum_relative_l2": relative_l2_limit,
            "maximum_absolute_error": max_abs_limit,
        },
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--runtime-identity", required=True)
    parser.add_argument(
        "--contract", type=Path,
        default=Path("quality/layered_quality_gate.v2.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.report_dir.glob("rank-*-pid-*.json"))
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    result = qualify(
        reports,
        json.loads(args.contract.read_text(encoding="utf-8")),
        run_id=args.run_id,
        source_revision=args.source_revision,
        runtime_identity=args.runtime_identity,
    )
    _atomic_write(args.out, result)
    return {"pass": 0, "fail": 1, "invalid": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
