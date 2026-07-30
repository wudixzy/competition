#!/usr/bin/env python3
"""Qualify FP32/FP16-rounding calibrated TP4 fused-prefill shadows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


REPORT_SCHEMA = (
    "bi100-fused-prefill-real-activation-calibrated-shadow-v1")
REPORT_SCHEMA_V2 = (
    "bi100-fused-prefill-real-activation-calibrated-shadow-v2")
RESULT_SCHEMA = (
    "bi100-fused-prefill-calibrated-shadow-qualification-v1")
RESULT_SCHEMA_V2 = (
    "bi100-fused-prefill-calibrated-shadow-qualification-v2")
CONTRACT_SCHEMA = "bi100-fused-prefill-numeric-adjudication-v1"
CONTRACT_SCHEMA_V2 = (
    "bi100-fused-prefill-real-activation-adjudication-v2")
CONTRACT_SHA256 = (
    "131e2ed8e0b34cc28a45486b9a9096d66c556759677b8bbd31024a33933d86b1"
)
CONTRACT_SHA256_V2 = (
    "ba37338f4d4112a1bd90e3e700334652a66ebb048f3cea7379ed21cdd3f3aceb"
)
Json = dict[str, Any]
METRIC_NAMES = (
    "relative_l2",
    "max_abs",
    "candidate_to_fp32_relative_l2",
    "candidate_to_fp32_max_abs",
    "rounded_to_fp32_relative_l2",
    "rounded_to_fp32_max_abs",
    "relative_l2_baseline_ratio",
    "max_abs_baseline_ratio",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


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


def _contract_values(
    contract: Any,
    contract_version: int = 1,
) -> tuple[Json, list[str]]:
    if contract_version not in {1, 2}:
        raise ValueError("contract version must be 1 or 2")
    source = contract if isinstance(contract, dict) else {}
    hard = source.get("hard_gates") or {}
    sampling = source.get("sampling") or {}
    expected_reference = {
        "implementation": "same-activation-pytorch-online-softmax",
        "accumulation_dtype": "float32",
        "rounded_baseline": "float32-reference-cast-to-float16",
        "same_input_tensors_required": True,
    }
    expected_hard = (
        {
            "candidate_and_reference_finite": True,
            "maximum_candidate_vs_rounded_relative_l2": 1.0e-5,
            "maximum_error_multiple_over_fp16_rounding": 2.0,
            "ratio_denominator_floor": 1.0e-12,
            "max_abs_fixed_threshold_role": "diagnostic_only",
            "semantic_evidence_may_waive_failure": False,
        }
        if contract_version == 1
        else {
            "candidate_and_reference_finite": True,
            "candidate_vs_rounded_relative_l2_role": "diagnostic_only",
            "maximum_error_multiple_over_fp16_rounding": 2.0,
            "ratio_denominator_floor": 1.0e-12,
            "max_abs_fixed_threshold_role": "diagnostic_only",
            "semantic_evidence_may_waive_failure": False,
        }
    )
    expected_sampling = {
        "required_ranks": [0, 1, 2, 3],
        "required_minimum_context_tokens": [49152, 114688],
        "minimum_observations_per_context_per_rank": 2,
        "query_length_min_exclusive": 16,
        "query_length_max_inclusive": 8192,
        "query_heads": 4,
        "kv_heads": 1,
        "head_dim": 256,
        "block_size": 16,
    }
    reasons = []
    expected_schema = (
        CONTRACT_SCHEMA if contract_version == 1 else CONTRACT_SCHEMA_V2)
    if (
        not isinstance(contract, dict)
        or contract.get("schema") != expected_schema
        or contract.get("version") != contract_version
        or set(contract) != {
            "schema", "version", "candidate_dtype", "reference",
            "hard_gates", "sampling", "execution", "promotion",
        }
    ):
        return {
            "hard": expected_hard,
            "sampling": expected_sampling,
            "contract_version": contract_version,
        }, ["numeric adjudication contract identity is invalid"]
    if (
        contract.get("candidate_dtype") != "float16"
        or contract.get("reference") != expected_reference
    ):
        reasons.append("numeric adjudication reference differs")
    if hard != expected_hard:
        reasons.append("numeric adjudication hard gates differ")
    if sampling != expected_sampling:
        reasons.append("numeric adjudication sampling differs")
    execution = contract.get("execution") or {}
    if (
        execution.get("finite_numeric_failure_action")
        != "record_and_continue_test_only"
        or execution.get("invalid_or_nonfinite_action") != "fail_fast"
        or execution.get("cross_arm_output_identity_role") != "diagnostic"
        or execution.get("task_capability_still_required") is not True
        or execution.get("performance_still_required") is not True
    ):
        reasons.append("numeric adjudication execution boundary differs")
    promotion = contract.get("promotion") or {}
    if (
        promotion.get("operator_surface_only") is not True
        or any(
            promotion.get(name) is not False
            for name in (
                "performance_authorized",
                "capability_authorized",
                "yaml_change_authorized",
                "main_merge_authorized",
                "production_promotion_authorized",
            )
        )
    ):
        reasons.append("numeric adjudication authorization boundary differs")
    return {
        "hard": expected_hard,
        "sampling": expected_sampling,
        "contract_version": contract_version,
    }, reasons


def _expected_observations(records: list[Any], expected: int) -> Json:
    completed = [
        record for record in records
        if (
            isinstance(record, dict)
            and record.get("status") in {"pass", "fail", "invalid"}
        )
    ]
    result = {
        "expected": expected,
        "reserved": len(records),
        "completed": len(completed),
        "passed": sum(
            record.get("status") == "pass"
            for record in records
            if isinstance(record, dict)
        ),
        "failed": sum(
            record.get("status") == "fail"
            for record in records
            if isinstance(record, dict)
        ),
        "invalid": sum(
            record.get("status") == "invalid"
            for record in records
            if isinstance(record, dict)
        ),
        "pending": sum(
            record.get("status") == "pending"
            for record in records
            if isinstance(record, dict)
        ),
    }
    for name in METRIC_NAMES:
        values = [
            record[name] for record in completed
            if isinstance(record.get(name), float)
        ]
        result[f"maximum_{name}"] = max(values) if values else None
    return result


def _expected_report_status(records: list[Any], expected: int) -> str:
    completed = [
        record for record in records
        if (
            isinstance(record, dict)
            and record.get("status") in {"pass", "fail", "invalid"}
        )
    ]
    if any(record.get("status") == "invalid" for record in completed):
        return "invalid"
    if any(record.get("status") == "fail" for record in completed):
        return "fail"
    if (
        len(completed) == expected
        and not any(
            isinstance(record, dict) and record.get("status") == "pending"
            for record in records
        )
    ):
        return "pass"
    return "collecting"


def _expected_thresholds(values: Json) -> Json:
    hard = values["hard"]
    thresholds = {
        "require_finite_candidate": True,
        "require_finite_reference": True,
        "maximum_error_multiple_over_fp16_rounding": (
            hard["maximum_error_multiple_over_fp16_rounding"]),
        "ratio_denominator_floor": hard["ratio_denominator_floor"],
        "fixed_max_abs_role": "diagnostic_only",
        "finite_failure_action": "record",
    }
    if values["contract_version"] == 1:
        thresholds["maximum_candidate_vs_rounded_relative_l2"] = (
            hard["maximum_candidate_vs_rounded_relative_l2"])
    else:
        thresholds["candidate_vs_rounded_relative_l2_role"] = (
            "diagnostic_only")
    return thresholds


def _record_qualified(record: Json, values: Json) -> bool:
    hard = values["hard"]
    return (
        record["candidate_finite"] is True
        and record["reference_finite"] is True
        and (
            values["contract_version"] == 2
            or record["relative_l2"]
            <= hard["maximum_candidate_vs_rounded_relative_l2"]
        )
        and record["candidate_to_fp32_relative_l2"]
        <= (
            hard["maximum_error_multiple_over_fp16_rounding"]
            * record["rounded_to_fp32_relative_l2"]
            + hard["ratio_denominator_floor"]
        )
        and record["candidate_to_fp32_max_abs"]
        <= (
            hard["maximum_error_multiple_over_fp16_rounding"]
            * record["rounded_to_fp32_max_abs"]
            + hard["ratio_denominator_floor"]
        )
    )


def qualify(
    reports: list[Any],
    contract: Any,
    *,
    run_id: str,
    source_revision: str,
    runtime_identity: str,
    contract_version: int = 1,
) -> Json:
    values, invalid_reasons = _contract_values(
        contract, contract_version)
    numeric_failures = []
    rank_reports: dict[int, Json] = {}
    accepted_records: list[Json] = []
    expected_report_fields = {
        "schema", "version", "run_id", "pid", "rank", "status",
        "selection", "thresholds", "observations", "records", "privacy",
    }
    base_record_fields = {
        "index", "status", "bucket_min_context_tokens",
        "context_tokens", "query_shape", "query_heads", "kv_heads",
        "head_dim", "block_size", "candidate_finite",
        "reference_finite", "relative_l2", "max_abs",
        "candidate_to_fp32_relative_l2",
        "candidate_to_fp32_max_abs",
        "rounded_to_fp32_relative_l2",
        "rounded_to_fp32_max_abs",
        "relative_l2_baseline_ratio", "max_abs_baseline_ratio",
        "error_stage", "error_type",
    }
    privacy_contract = {
        "contains_prompts": False,
        "contains_model_outputs": False,
        "contains_tensor_values": False,
        "contains_token_ids": False,
        "contains_credentials": False,
    }
    sampling = values.get("sampling", {})
    ranks = sampling.get("required_ranks", [])
    contexts = sampling.get("required_minimum_context_tokens", [])
    minimum = sampling.get(
        "minimum_observations_per_context_per_rank")
    expected_observations = len(contexts) * minimum
    expected_report_schema = (
        REPORT_SCHEMA if contract_version == 1 else REPORT_SCHEMA_V2)
    for ordinal, report in enumerate(reports):
        label = f"report[{ordinal}]"
        if not isinstance(report, dict) or set(report) != expected_report_fields:
            invalid_reasons.append(f"{label}: fields are invalid")
            continue
        rank = report.get("rank")
        if (
            report.get("schema") != expected_report_schema
            or report.get("version") != contract_version
            or report.get("run_id") != run_id
            or not isinstance(report.get("pid"), int)
            or isinstance(report["pid"], bool)
            or report["pid"] <= 0
            or not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank not in ranks
            or rank in rank_reports
        ):
            invalid_reasons.append(f"{label}: identity is invalid")
            continue
        rank_reports[rank] = report
        if report.get("selection") != {
            "minimum_context_tokens": contexts,
            "max_calls_per_context": minimum,
        }:
            invalid_reasons.append(f"rank {rank}: selection differs")
        if report.get("thresholds") != _expected_thresholds(values):
            invalid_reasons.append(f"rank {rank}: thresholds differ")
        if report.get("privacy") != privacy_contract:
            invalid_reasons.append(f"rank {rank}: privacy differs")
        records = report.get("records")
        if not isinstance(records, list):
            invalid_reasons.append(f"rank {rank}: records are invalid")
            continue
        if report.get("observations") != _expected_observations(
            records, expected_observations
        ):
            invalid_reasons.append(
                f"rank {rank}: observations are inconsistent")
        if report.get("status") != _expected_report_status(
            records, expected_observations
        ):
            invalid_reasons.append(f"rank {rank}: report status differs")
        counts = {context: 0 for context in contexts}
        for index, record in enumerate(records):
            if (
                not isinstance(record, dict)
                or set(record) != base_record_fields
                or record.get("index") != index
            ):
                invalid_reasons.append(
                    f"rank {rank} record {index}: fields are invalid")
                continue
            bucket = record.get("bucket_min_context_tokens")
            context_tokens = record.get("context_tokens")
            shape = record.get("query_shape")
            bucket_index = (
                contexts.index(bucket) if bucket in contexts else -1)
            upper = (
                contexts[bucket_index + 1]
                if 0 <= bucket_index < len(contexts) - 1 else None)
            if (
                bucket not in counts
                or not isinstance(context_tokens, int)
                or isinstance(context_tokens, bool)
                or context_tokens < bucket
                or (upper is not None and context_tokens >= upper)
                or not isinstance(shape, list)
                or len(shape) != 3
                or not isinstance(shape[0], int)
                or not (
                    sampling.get("query_length_min_exclusive", 0)
                    < shape[0]
                    <= sampling.get("query_length_max_inclusive", 0)
                )
                or shape[1:] != [
                    sampling.get("query_heads"),
                    sampling.get("head_dim"),
                ]
                or record.get("query_heads")
                != sampling.get("query_heads")
                or record.get("kv_heads") != sampling.get("kv_heads")
                or record.get("head_dim") != sampling.get("head_dim")
                or record.get("block_size") != sampling.get("block_size")
            ):
                invalid_reasons.append(
                    f"rank {rank} record {index}: shape is invalid")
                continue
            counts[bucket] += 1
            status = record.get("status")
            candidate_finite = record.get("candidate_finite")
            reference_finite = record.get("reference_finite")
            if status not in {"pass", "fail"}:
                invalid_reasons.append(
                    f"rank {rank} record {index}: result is incomplete")
                continue
            if reference_finite is not True:
                invalid_reasons.append(
                    f"rank {rank} record {index}: reference is invalid")
                continue
            if candidate_finite is False:
                if (
                    status != "fail"
                    or any(record.get(name) is not None for name in METRIC_NAMES)
                    or record.get("error_stage") is not None
                    or record.get("error_type") is not None
                ):
                    invalid_reasons.append(
                        f"rank {rank} record {index}: non-finite result "
                        "contract differs")
                else:
                    numeric_failures.append(
                        f"rank {rank} record {index}: candidate is non-finite")
                continue
            if candidate_finite is not True:
                invalid_reasons.append(
                    f"rank {rank} record {index}: candidate is invalid")
                continue
            if (
                any(
                    not _finite(record.get(name))
                    or float(record[name]) < 0.0
                    for name in METRIC_NAMES
                )
                or record.get("error_stage") is not None
                or record.get("error_type") is not None
            ):
                invalid_reasons.append(
                    f"rank {rank} record {index}: metrics are invalid")
                continue
            floor = values["hard"]["ratio_denominator_floor"]
            expected_relative_ratio = (
                record["candidate_to_fp32_relative_l2"]
                / max(record["rounded_to_fp32_relative_l2"], floor)
            )
            expected_max_ratio = (
                record["candidate_to_fp32_max_abs"]
                / max(record["rounded_to_fp32_max_abs"], floor)
            )
            if (
                not math.isclose(
                    record["relative_l2_baseline_ratio"],
                    expected_relative_ratio,
                    rel_tol=1.0e-9,
                    abs_tol=1.0e-12,
                )
                or not math.isclose(
                    record["max_abs_baseline_ratio"],
                    expected_max_ratio,
                    rel_tol=1.0e-9,
                    abs_tol=1.0e-12,
                )
            ):
                invalid_reasons.append(
                    f"rank {rank} record {index}: ratio is inconsistent")
                continue
            expected_status = (
                "pass" if _record_qualified(record, values) else "fail")
            if record.get("status") != expected_status:
                invalid_reasons.append(
                    f"rank {rank} record {index}: status is inconsistent")
                continue
            if expected_status == "fail":
                numeric_failures.append(
                    f"rank {rank} record {index}: calibrated hard gate failed")
            accepted_records.append(record)
        for bucket, count in counts.items():
            if count != minimum:
                invalid_reasons.append(
                    f"rank {rank}: context bucket {bucket} has {count} records")

    missing_ranks = sorted(set(ranks) - set(rank_reports))
    if missing_ranks:
        invalid_reasons.append(f"missing rank reports: {missing_ranks}")
    if invalid_reasons:
        status = "invalid"
    elif numeric_failures:
        status = "fail"
    else:
        status = "pass"
    return {
        "schema": (
            RESULT_SCHEMA if contract_version == 1 else RESULT_SCHEMA_V2),
        "version": contract_version,
        "source_revision": source_revision,
        "runtime_identity": runtime_identity,
        "run_id": run_id,
        "status": status,
        "qualified": status == "pass",
        "operator_surface_authorized": status == "pass",
        "capability_evaluated": False,
        "performance_evaluated": False,
        "yaml_change_authorized": False,
        "main_merge_authorized": False,
        "production_promotion_authorized": False,
        "invalid_reasons": invalid_reasons,
        "numeric_failures": sorted(set(numeric_failures)),
        "rank_count": len(rank_reports),
        "observation_count": len(accepted_records),
        "maxima": {
            name: (
                max(float(record[name]) for record in accepted_records)
                if accepted_records else None
            )
            for name in METRIC_NAMES
        },
        "contract_sha256": (
            CONTRACT_SHA256
            if contract_version == 1
            else CONTRACT_SHA256_V2),
        "privacy": privacy_contract,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--runtime-identity", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--contract-version", type=int, choices=(1, 2), default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    expected_contract_sha256 = (
        CONTRACT_SHA256
        if args.contract_version == 1
        else CONTRACT_SHA256_V2)
    if sha256(args.contract) != expected_contract_sha256:
        raise SystemExit("numeric adjudication contract SHA-256 differs")
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.report_dir.glob("rank-*-pid-*.json"))
    ]
    result = qualify(
        reports,
        json.loads(args.contract.read_text(encoding="utf-8")),
        run_id=args.run_id,
        source_revision=args.source_revision,
        runtime_identity=args.runtime_identity,
        contract_version=args.contract_version,
    )
    _atomic_write(args.out, result)
    return {"pass": 0, "fail": 1, "invalid": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
