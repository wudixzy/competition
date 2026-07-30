#!/usr/bin/env python3
"""Qualify four-way real-activation replay evidence for the funnel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any


REPORT_SCHEMA = "bi100-fused-prefill-activation-replay-v1"
RESULT_SCHEMA = "bi100-fused-prefill-activation-replay-qualification-v1"
CONTRACT_SCHEMA = "bi100-experiment-funnel-v1"
NUMERIC_CONTRACT_SCHEMA = "bi100-fused-prefill-numeric-adjudication-v1"
NUMERIC_CONTRACT_SHA256 = (
    "131e2ed8e0b34cc28a45486b9a9096d66c556759677b8bbd31024a33933d86b1"
)
TRIALS = 3
LSE_RELATIVE_L2_LIMIT = 1.0e-5
REPORT_FIELDS = {
    "schema", "version", "capture_source_revision",
    "candidate_source_revision", "runtime_identity", "instance",
    "visible_physical_gpu", "rank", "device_name", "torch_version",
    "bank", "candidate_extension", "records", "all_numeric_qualified",
    "privacy", "authorization",
}
RECORD_FIELDS = {
    "rank", "bucket_min_context_tokens", "call_ordinal",
    "context_tokens", "query_length", "case_sha256", "load_elapsed_s",
    "reference_timing", "candidate_timing", "candidate_speedup",
    "numeric",
}
TIMING_FIELDS = {
    "warmups", "trials", "cuda_trials_ms", "cuda_median_ms",
}
NUMERIC_FIELDS = {
    "candidate_finite", "reference_finite", "finite",
    "candidate_vs_rounded_relative_l2",
    "candidate_vs_rounded_max_abs_diagnostic",
    "candidate_to_fp32_relative_l2", "candidate_to_fp32_max_abs",
    "rounded_to_fp32_relative_l2", "rounded_to_fp32_max_abs",
    "candidate_lse_finite", "reference_lse_finite", "lse_finite",
    "lse_relative_l2", "qualified",
}
PRIVACY_CONTRACT = {
    "raw_tensors_persisted_in_report": False,
    "prompts_persisted_in_report": False,
    "model_outputs_persisted_in_report": False,
    "credentials_persisted_in_report": False,
}
AUTHORIZATION_CONTRACT = {
    "short_tp4_authorized": False,
    "main_or_yaml_change_authorized": False,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            json.dump(value, stream, ensure_ascii=True, indent=2,
                      sort_keys=True, allow_nan=False)
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


def _finite_positive(value: Any) -> bool:
    return _finite(value) and float(value) > 0.0


def _hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _contract_values(
    contract: Any,
    numeric_contract: Any,
) -> tuple[dict[str, Any], list[str]]:
    reasons = []
    expected = {
        "required_ranks": [0, 1, 2, 3],
        "required_buckets": [24576, 57344, 122880],
        "required_ordinals": [0, 4, 9],
        "minimum_speedup": 1.05,
        "maximum_regression": 0.02,
        "relative_l2_limit": 1.0e-5,
        "error_multiplier": 2.0,
        "ratio_floor": 1.0e-12,
    }
    if (
        not isinstance(contract, dict)
        or contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("version") != 1
        or not isinstance(contract.get("stages"), list)
    ):
        reasons.append("experiment funnel contract is invalid")
        stages = []
    else:
        stages = contract["stages"]
    l2_rows = [
        stage for stage in stages
        if isinstance(stage, dict) and stage.get("id") == "L2"
    ]
    if len(l2_rows) != 1:
        reasons.append("experiment funnel has no unique L2 stage")
        l2 = {}
    else:
        l2 = l2_rows[0]
    capture = _mapping(l2.get("capture"))
    screen = _mapping(l2.get("continuation_screen"))
    if (
        capture.get("required_tp_ranks")
        != expected["required_ranks"]
        or capture.get("required_context_buckets")
        != expected["required_buckets"]
        or capture.get("required_full_attention_call_ordinals")
        != expected["required_ordinals"]
        or capture.get("producer_path") != "baseline_pytorch_fallback"
        or capture.get("synthetic_prompt_only") is not True
        or capture.get("raw_bank_location") != "private_tmp_only"
        or capture.get("raw_bank_may_be_committed") is not False
        or screen.get("minimum_median_candidate_speedup")
        != expected["minimum_speedup"]
        or screen.get("maximum_single_case_regression_fraction")
        != expected["maximum_regression"]
        or screen.get("numeric_contract")
        != "fused_prefill_numeric_adjudication.v1.json"
    ):
        reasons.append("L2 funnel contract differs")

    numeric = numeric_contract if isinstance(numeric_contract, dict) else {}
    expected_reference = {
        "implementation": "same-activation-pytorch-online-softmax",
        "accumulation_dtype": "float32",
        "rounded_baseline": "float32-reference-cast-to-float16",
        "same_input_tensors_required": True,
    }
    expected_hard = {
        "candidate_and_reference_finite": True,
        "maximum_candidate_vs_rounded_relative_l2": 1.0e-5,
        "maximum_error_multiple_over_fp16_rounding": 2.0,
        "ratio_denominator_floor": 1.0e-12,
        "max_abs_fixed_threshold_role": "diagnostic_only",
        "semantic_evidence_may_waive_failure": False,
    }
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
    if (
        numeric.get("schema") != NUMERIC_CONTRACT_SCHEMA
        or numeric.get("version") != 1
        or set(numeric) != {
            "schema", "version", "candidate_dtype", "reference",
            "hard_gates", "sampling", "execution", "promotion",
        }
        or numeric.get("candidate_dtype") != "float16"
        or numeric.get("reference") != expected_reference
        or numeric.get("hard_gates") != expected_hard
        or numeric.get("sampling") != expected_sampling
    ):
        reasons.append("numeric adjudication contract differs")
    execution = _mapping(numeric.get("execution"))
    if (
        execution.get("finite_numeric_failure_action")
        != "record_and_continue_test_only"
        or execution.get("invalid_or_nonfinite_action") != "fail_fast"
        or execution.get("cross_arm_output_identity_role") != "diagnostic"
        or execution.get("task_capability_still_required") is not True
        or execution.get("performance_still_required") is not True
    ):
        reasons.append("numeric adjudication execution boundary differs")
    promotion = _mapping(numeric.get("promotion"))
    if (
        promotion.get("operator_surface_only") is not True
        or any(
            promotion.get(name) is not False
            for name in (
                "performance_authorized", "capability_authorized",
                "yaml_change_authorized", "main_merge_authorized",
                "production_promotion_authorized",
            )
        )
    ):
        reasons.append("numeric adjudication authorization differs")
    return expected, reasons


def _timing_median(
    timing: Any,
    *,
    label: str,
    invalid_reasons: list[str],
) -> float | None:
    if not isinstance(timing, dict) or set(timing) != TIMING_FIELDS:
        invalid_reasons.append(f"{label}: timing fields differ")
        return None
    trials = timing.get("cuda_trials_ms")
    median = timing.get("cuda_median_ms")
    if (
        timing.get("warmups") != 1
        or timing.get("trials") != TRIALS
        or not isinstance(trials, list)
        or len(trials) != TRIALS
        or not all(_finite_positive(value) for value in trials)
        or not _finite_positive(median)
    ):
        invalid_reasons.append(f"{label}: timing values are invalid")
        return None
    expected = statistics.median(float(value) for value in trials)
    if not math.isclose(
        float(median), expected, rel_tol=1.0e-9, abs_tol=1.0e-9
    ):
        invalid_reasons.append(f"{label}: timing median is inconsistent")
        return None
    return expected


def _numeric_qualified(
    numeric: Any,
    values: dict[str, Any],
    *,
    label: str,
    invalid_reasons: list[str],
    numeric_reasons: list[str],
) -> bool:
    if not isinstance(numeric, dict) or set(numeric) != NUMERIC_FIELDS:
        invalid_reasons.append(f"{label}: numeric fields differ")
        return False
    candidate_finite = numeric.get("candidate_finite")
    reference_finite = numeric.get("reference_finite")
    candidate_lse_finite = numeric.get("candidate_lse_finite")
    reference_lse_finite = numeric.get("reference_lse_finite")
    if not all(
        isinstance(value, bool)
        for value in (
            candidate_finite, reference_finite,
            candidate_lse_finite, reference_lse_finite,
            numeric.get("finite"), numeric.get("lse_finite"),
            numeric.get("qualified"),
        )
    ):
        invalid_reasons.append(f"{label}: numeric status is malformed")
        return False
    if numeric["finite"] is not (
            candidate_finite and reference_finite):
        invalid_reasons.append(f"{label}: finite summary is inconsistent")
    if numeric["lse_finite"] is not (
            candidate_lse_finite and reference_lse_finite):
        invalid_reasons.append(
            f"{label}: LSE finite summary is inconsistent")
    if reference_finite is not True or reference_lse_finite is not True:
        invalid_reasons.append(f"{label}: reference is nonfinite")
        calculated = False
    elif candidate_finite is not True or candidate_lse_finite is not True:
        numeric_reasons.append(f"{label}: candidate is nonfinite")
        calculated = False
    else:
        metric_names = NUMERIC_FIELDS - {
            "candidate_finite", "reference_finite", "finite",
            "candidate_lse_finite", "reference_lse_finite", "lse_finite",
            "qualified",
        }
        if any(
            not _finite(numeric.get(name))
            or float(numeric[name]) < 0.0
            for name in metric_names
        ):
            invalid_reasons.append(f"{label}: numeric metric is invalid")
            calculated = False
        else:
            calculated = bool(
                numeric["candidate_vs_rounded_relative_l2"]
                <= values["relative_l2_limit"]
                and numeric["candidate_to_fp32_relative_l2"]
                <= (
                    values["error_multiplier"]
                    * numeric["rounded_to_fp32_relative_l2"]
                    + values["ratio_floor"]
                )
                and numeric["candidate_to_fp32_max_abs"]
                <= (
                    values["error_multiplier"]
                    * numeric["rounded_to_fp32_max_abs"]
                    + values["ratio_floor"]
                )
                and numeric["lse_relative_l2"]
                <= LSE_RELATIVE_L2_LIMIT
            )
            if not calculated:
                numeric_reasons.append(
                    f"{label}: calibrated numeric gate failed")
    if numeric.get("qualified") is not calculated:
        invalid_reasons.append(
            f"{label}: reported numeric qualification is inconsistent")
    return calculated


def qualify(
    reports: list[Any],
    contract: Any,
    numeric_contract: Any,
    *,
    profile: str,
) -> dict[str, Any]:
    values, invalid_reasons = _contract_values(
        contract, numeric_contract)
    numeric_reasons: list[str] = []
    performance_reasons: list[str] = []
    coverage_reasons: list[str] = []
    accepted: list[dict[str, Any]] = []
    speedups: list[float] = []
    identities: set[tuple[str, str, str, str]] = set()
    ranks: set[int] = set()
    physical_gpus: set[int] = set()
    torch_versions: set[str] = set()
    artifact_identities: set[tuple[str, int]] = set()
    bank_run_ids: set[str] = set()
    bank_manifest_shas: set[str] = set()
    case_shas: set[str] = set()
    observed: set[tuple[int, int, int]] = set()
    if not isinstance(reports, list) or len(reports) != 4:
        invalid_reasons.append("exactly four rank reports are required")
        reports = reports if isinstance(reports, list) else []
    for index, report in enumerate(reports):
        label = f"report {index}"
        if (
            not isinstance(report, dict)
            or set(report) != REPORT_FIELDS
            or report.get("schema") != REPORT_SCHEMA
            or report.get("version") != 1
        ):
            invalid_reasons.append(f"{label}: fields or schema differ")
            continue
        rank = report.get("rank")
        gpu = report.get("visible_physical_gpu")
        records = report.get("records")
        artifact = report.get("candidate_extension")
        bank = report.get("bank")
        identity_values = (
            report.get("capture_source_revision"),
            report.get("candidate_source_revision"),
            report.get("runtime_identity"),
            report.get("instance"),
        )
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank not in values["required_ranks"]
            or rank in ranks
            or not isinstance(gpu, int)
            or isinstance(gpu, bool)
            or gpu != rank
            or gpu in physical_gpus
            or not isinstance(records, list)
            or not records
            or not all(isinstance(value, str) and value
                       for value in identity_values)
            or not _hex(identity_values[0], 40)
            or not _hex(identity_values[1], 40)
            or report.get("device_name") != "Iluvatar BI-V100"
            or not isinstance(report.get("torch_version"), str)
            or not report["torch_version"]
            or report.get("privacy") != PRIVACY_CONTRACT
            or report.get("authorization") != AUTHORIZATION_CONTRACT
        ):
            invalid_reasons.append(f"{label}: identity or structure is invalid")
            continue
        if (
            not isinstance(artifact, dict)
            or set(artifact) != {"path", "sha256", "size_bytes"}
            or not _hex(artifact.get("sha256"), 64)
            or not isinstance(artifact.get("path"), str)
            or not artifact["path"]
            or not isinstance(artifact.get("size_bytes"), int)
            or isinstance(artifact["size_bytes"], bool)
            or artifact["size_bytes"] <= 0
        ):
            invalid_reasons.append(f"{label}: artifact identity is invalid")
            continue
        if (
            not isinstance(bank, dict)
            or set(bank) != {
                "manifest", "manifest_sha256", "run_id", "record_count",
            }
            or not isinstance(bank.get("manifest"), str)
            or not bank["manifest"]
            or Path(bank["manifest"]).name
            != f"rank-{rank}.manifest.json"
            or not _hex(bank.get("manifest_sha256"), 64)
            or not isinstance(bank.get("run_id"), str)
            or not bank["run_id"]
            or not isinstance(bank.get("record_count"), int)
            or isinstance(bank["record_count"], bool)
            or bank.get("record_count") != len(records)
        ):
            invalid_reasons.append(f"{label}: bank identity is invalid")
            continue
        ranks.add(rank)
        physical_gpus.add(gpu)
        torch_versions.add(report["torch_version"])
        identities.add(identity_values)
        artifact_identities.add(
            (artifact["sha256"], artifact["size_bytes"]))
        bank_run_ids.add(bank["run_id"])
        bank_manifest_shas.add(bank["manifest_sha256"])
        report_numeric_results = []
        for record_index, record in enumerate(records):
            record_label = f"rank {rank} record {record_index}"
            if (
                not isinstance(record, dict)
                or set(record) != RECORD_FIELDS
                or record.get("rank") != rank
                or record.get("bucket_min_context_tokens")
                not in values["required_buckets"]
                or not isinstance(
                    record.get("bucket_min_context_tokens"), int)
                or isinstance(
                    record["bucket_min_context_tokens"], bool)
                or record.get("call_ordinal")
                not in values["required_ordinals"]
                or not isinstance(record.get("call_ordinal"), int)
                or isinstance(record["call_ordinal"], bool)
                or not isinstance(record.get("context_tokens"), int)
                or isinstance(record["context_tokens"], bool)
                or record["context_tokens"] % 16 != 0
                or not isinstance(record.get("query_length"), int)
                or isinstance(record["query_length"], bool)
                or not 16 < record["query_length"] <= 8192
                or (
                    record["context_tokens"] + record["query_length"]
                    > 262144
                )
                or not _hex(record.get("case_sha256"), 64)
                or not _finite(record.get("load_elapsed_s"))
                or float(record["load_elapsed_s"]) < 0.0
            ):
                invalid_reasons.append(
                    f"{record_label}: identity or shape is invalid")
                report_numeric_results.append(False)
                continue
            if record["case_sha256"] in case_shas:
                coverage_reasons.append(
                    f"{record_label}: duplicate activation case")
            else:
                case_shas.add(record["case_sha256"])
            bucket = record["bucket_min_context_tokens"]
            bucket_index = values["required_buckets"].index(bucket)
            upper = (
                values["required_buckets"][bucket_index + 1]
                if bucket_index + 1 < len(values["required_buckets"])
                else 262145
            )
            if not bucket <= record["context_tokens"] < upper:
                invalid_reasons.append(
                    f"{record_label}: context is outside its bucket")
            key = (rank, bucket, record["call_ordinal"])
            if key in observed:
                coverage_reasons.append(
                    f"{record_label}: duplicate matrix cell")
            else:
                observed.add(key)
            reference_ms = _timing_median(
                record.get("reference_timing"),
                label=f"{record_label} reference",
                invalid_reasons=invalid_reasons,
            )
            candidate_ms = _timing_median(
                record.get("candidate_timing"),
                label=f"{record_label} candidate",
                invalid_reasons=invalid_reasons,
            )
            reported_speedup = record.get("candidate_speedup")
            if (
                reference_ms is None
                or candidate_ms is None
                or not _finite_positive(reported_speedup)
            ):
                invalid_reasons.append(
                    f"{record_label}: replay speedup is invalid")
            else:
                calculated_speedup = reference_ms / candidate_ms
                if not math.isclose(
                    float(reported_speedup),
                    calculated_speedup,
                    rel_tol=1.0e-9,
                    abs_tol=1.0e-9,
                ):
                    invalid_reasons.append(
                        f"{record_label}: replay speedup is inconsistent")
                else:
                    speedups.append(calculated_speedup)
                    accepted.append(record)
            report_numeric_results.append(_numeric_qualified(
                record.get("numeric"),
                values,
                label=record_label,
                invalid_reasons=invalid_reasons,
                numeric_reasons=numeric_reasons,
            ))
        expected_all_numeric = bool(
            report_numeric_results and all(report_numeric_results))
        if report.get("all_numeric_qualified") is not expected_all_numeric:
            invalid_reasons.append(
                f"rank {rank}: aggregate numeric status is inconsistent")
    if ranks != set(values["required_ranks"]):
        coverage_reasons.append("replay does not cover all four TP ranks")
    if physical_gpus != set(values["required_ranks"]):
        coverage_reasons.append("replay does not cover four physical GPUs")
    if len(identities) != 1:
        invalid_reasons.append("replay reports have different run identities")
    if len(torch_versions) != 1:
        invalid_reasons.append("replay reports use different Torch runtimes")
    if len(artifact_identities) != 1:
        invalid_reasons.append("replay reports use different artifacts")
    if len(bank_run_ids) != 1:
        invalid_reasons.append("replay reports use different activation runs")
    if len(bank_manifest_shas) != 4:
        invalid_reasons.append(
            "replay reports do not bind four distinct bank manifests")

    if profile == "qualification":
        expected = {
            (rank, bucket, ordinal)
            for rank in values["required_ranks"]
            for bucket in values["required_buckets"]
            for ordinal in values["required_ordinals"]
        }
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        if missing:
            coverage_reasons.append(
                f"qualification profile is missing {len(missing)} cases")
        if extra:
            coverage_reasons.append(
                f"qualification profile has {len(extra)} unexpected cases")
    elif profile == "smoke":
        for rank in values["required_ranks"]:
            if not any(row[0] == rank for row in observed):
                coverage_reasons.append(
                    f"smoke profile has no case for rank {rank}")
    else:
        invalid_reasons.append("unknown replay qualification profile")

    median_speedup = statistics.median(speedups) if speedups else None
    minimum_case_speedup = min(speedups) if speedups else None
    if (
        median_speedup is None
        or median_speedup < values["minimum_speedup"]
    ):
        performance_reasons.append(
            "median replay speedup is below the continuation screen")
    minimum_allowed = 1.0 - values["maximum_regression"]
    if (
        minimum_case_speedup is None
        or minimum_case_speedup < minimum_allowed
    ):
        performance_reasons.append(
            "a replay case exceeds the allowed regression")

    execution_valid = not invalid_reasons and not numeric_reasons
    stage_qualified = bool(
        execution_valid
        and not performance_reasons
        and not coverage_reasons)
    return {
        "schema": RESULT_SCHEMA,
        "version": 1,
        "profile": profile,
        "execution_valid": execution_valid,
        "stage_qualified": stage_qualified,
        "invalid_reasons": invalid_reasons,
        "numeric_reasons": numeric_reasons,
        "performance_reasons": performance_reasons,
        "coverage_reasons": coverage_reasons,
        "report_count": len(reports),
        "record_count": len(accepted),
        "ranks": sorted(ranks),
        "median_candidate_speedup": median_speedup,
        "minimum_case_speedup": minimum_case_speedup,
        "contract_sha256": None,
        "numeric_contract_sha256": None,
        "authorization": {
            "short_tp4_authorized": (
                stage_qualified and profile == "qualification"),
            "long_context_authorized": False,
            "main_or_yaml_change_authorized": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--numeric-contract", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=("smoke", "qualification"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reports = [
        json.loads(path.read_text(encoding="ascii"))
        for path in args.report
    ]
    contract = json.loads(args.contract.read_text(encoding="ascii"))
    numeric_contract = json.loads(
        args.numeric_contract.read_text(encoding="ascii"))
    result = qualify(
        reports,
        contract,
        numeric_contract,
        profile=args.profile,
    )
    result["contract_sha256"] = _sha256(args.contract)
    result["numeric_contract_sha256"] = _sha256(args.numeric_contract)
    if result["numeric_contract_sha256"] != NUMERIC_CONTRACT_SHA256:
        result["invalid_reasons"].append(
            "numeric adjudication contract digest differs")
        result["execution_valid"] = False
        result["stage_qualified"] = False
        result["authorization"]["short_tp4_authorized"] = False
    _atomic_write(args.out, result)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if result["stage_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
