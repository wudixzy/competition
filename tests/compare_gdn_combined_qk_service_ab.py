#!/usr/bin/env python3
"""Qualify the fixed same-overlay M1-65 TP4 decode A/B."""

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

import gdn_combined_qk_decode_api as decode_api
import quality_runtime_contract as runtime_contract


Json = dict[str, Any]
SCHEMA = "bi100-gdn-combined-qk-service-ab-v1"
VERSION = 1
EXPECTED_CONFIG = {
    "requests": 3,
    "warmup": 1,
    "tokens": 1000,
    "seed": 20260727,
}
MIN_MEDIAN_PAIRED_SPEEDUP = 1.01
MIN_P10_RATIO = 0.98
FINAL_OUTPUT_TPS_P10 = 20.0
ALLOWED_ENV_DELTA = {"BI100_GDN_COMBINED_QK_NORM"}


def _atomic_write(path: Path, value: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _finite_positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _report_reasons(
    report: Any,
    label: str,
    expected_profile: str,
) -> list[str]:
    reasons: list[str] = []
    if not isinstance(report, dict):
        return [f"{label}: report is not an object"]
    if (
        report.get("schema") != decode_api.SCHEMA
        or report.get("version") != decode_api.VERSION
    ):
        reasons.append(f"{label}: report schema is invalid")
    if report.get("qualified") is not True:
        reasons.append(f"{label}: decode probe did not qualify")
    privacy = report.get("privacy")
    if privacy != {
        "contains_raw_requests": False,
        "contains_raw_model_outputs": False,
        "contains_credentials": False,
    }:
        reasons.append(f"{label}: privacy declaration is invalid")
    config = report.get("config")
    if not isinstance(config, dict):
        config = {}
        reasons.append(f"{label}: config is missing")
    for name, expected in EXPECTED_CONFIG.items():
        if config.get(name) != expected:
            reasons.append(
                f"{label}: config {name} must equal {expected!r}")
    if config.get("prompt_sha256") != hashlib.sha256(
            decode_api.PROMPT.encode("ascii")).hexdigest():
        reasons.append(f"{label}: prompt identity differs")

    wrapper = report.get("runtime_contract")
    contract = (
        wrapper.get("contract") if isinstance(wrapper, dict) else None)
    if not isinstance(contract, dict):
        reasons.append(f"{label}: runtime contract is missing")
    else:
        runtime = report.get("runtime") or {}
        expected_runtime = {
            "source_revision": runtime.get("source_revision"),
            "runtime_identity": runtime.get("runtime_identity"),
            "instance": runtime.get("instance"),
            "gpu_count": 4,
            "tensor_parallel_size": 4,
            "max_model_len": 262144,
            "model_path": contract.get("model_path"),
            "tokenizer_path": contract.get("tokenizer_path"),
            "served_model_name": "llm",
        }
        try:
            digest = runtime_contract.validate_runtime_contract(
                contract, expected_runtime, require_cache_trace=True)
        except runtime_contract.RuntimeContractError as error:
            reasons.append(f"{label}: {error}")
        else:
            if not isinstance(wrapper, dict) or wrapper.get("sha256") != digest:
                reasons.append(f"{label}: runtime contract digest differs")
        environment = contract.get("environment") or {}
        expected_kernel = runtime_contract.KERNEL_PROFILES[expected_profile]
        for name, expected in expected_kernel.items():
            if environment.get(name) != expected:
                reasons.append(
                    f"{label}: runtime {name} must equal {expected}")

    rows = report.get("requests")
    if not isinstance(rows, list) or len(rows) != EXPECTED_CONFIG["requests"]:
        reasons.append(f"{label}: measured request set is incomplete")
        rows = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            reasons.append(f"{label}: request {index} is invalid")
            continue
        for field, expected in (
            ("ok", True),
            ("http_status", 200),
            ("completion_tokens", EXPECTED_CONFIG["tokens"]),
            ("finish_reason", "length"),
        ):
            if row.get(field) != expected:
                reasons.append(
                    f"{label}: request {index} {field} differs")
        if not _finite_positive(row.get("output_tps")):
            reasons.append(
                f"{label}: request {index} output TPS is invalid")
        for field in ("first_output_sha256", "semantic_output_sha256"):
            value = row.get(field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef"
                       for character in value)
            ):
                reasons.append(
                    f"{label}: request {index} {field} is invalid")
    return reasons


def compare(control: Any, candidate: Any) -> Json:
    reasons = _report_reasons(
        control, "control", "strict-reference")
    reasons.extend(_report_reasons(
        candidate, "candidate", "strict-reference-combined-qk"))
    exact_rows: list[Json] = []
    paired_speedups: list[float] = []

    if isinstance(control, dict) and isinstance(candidate, dict):
        for field in (
            "source_revision", "runtime_identity", "runtime_overlay_sha256",
            "instance", "gpu_count", "tensor_parallel_size", "max_model_len",
        ):
            if (
                (control.get("runtime") or {}).get(field)
                != (candidate.get("runtime") or {}).get(field)
            ):
                reasons.append(f"A/B runtime differs in {field}")
        control_contract = (
            (control.get("runtime_contract") or {}).get("contract") or {})
        candidate_contract = (
            (candidate.get("runtime_contract") or {}).get("contract") or {})
        for field in (
            "source_revision", "runtime_identity", "runtime_overlay_sha256",
            "instance", "base_image", "command", "gpu_count",
            "tensor_parallel_size", "max_model_len", "model_path",
            "tokenizer_path", "served_model_name", "cache_trace_enabled",
        ):
            if control_contract.get(field) != candidate_contract.get(field):
                reasons.append(f"A/B runtime contract differs in {field}")
        control_env = control_contract.get("environment") or {}
        candidate_env = candidate_contract.get("environment") or {}
        changed_env = {
            name for name in set(control_env) | set(candidate_env)
            if control_env.get(name) != candidate_env.get(name)
        }
        if changed_env != ALLOWED_ENV_DELTA:
            reasons.append(
                "A/B environment delta must contain only "
                "BI100_GDN_COMBINED_QK_NORM")

        control_rows = control.get("requests") or []
        candidate_rows = candidate.get("requests") or []
        if len(control_rows) == len(candidate_rows):
            for index, (left, right) in enumerate(
                    zip(control_rows, candidate_rows)):
                row_reasons = []
                for field in (
                    "http_status", "prompt_tokens", "completion_tokens",
                    "finish_reason", "content_chars", "reasoning_chars",
                    "tool_call_fragments", "first_output_sha256",
                    "semantic_output_sha256",
                ):
                    if left.get(field) != right.get(field):
                        row_reasons.append(f"{field} differs")
                if (
                    _finite_positive(left.get("output_tps"))
                    and _finite_positive(right.get("output_tps"))
                ):
                    paired_speedups.append(
                        float(right["output_tps"])
                        / float(left["output_tps"]))
                exact_rows.append({
                    "index": index,
                    "qualified": not row_reasons,
                    "reasons": row_reasons,
                })
                reasons.extend(
                    f"request {index}: {reason}" for reason in row_reasons)

    control_summary = (
        control.get("summary") if isinstance(control, dict) else {}) or {}
    candidate_summary = (
        candidate.get("summary") if isinstance(candidate, dict) else {}) or {}
    control_p10 = control_summary.get("output_tps_p10")
    candidate_p10 = candidate_summary.get("output_tps_p10")
    p10_ratio = (
        float(candidate_p10) / float(control_p10)
        if _finite_positive(control_p10)
        and _finite_positive(candidate_p10) else None)
    paired_median = (
        statistics.median(paired_speedups) if paired_speedups else None)
    if p10_ratio is None or p10_ratio < MIN_P10_RATIO:
        reasons.append(
            f"candidate output TPS P10 ratio is below {MIN_P10_RATIO:g}")
    if (
        paired_median is None
        or paired_median < MIN_MEDIAN_PAIRED_SPEEDUP
    ):
        reasons.append(
            "candidate median paired speedup is below "
            f"{MIN_MEDIAN_PAIRED_SPEEDUP:g}x")

    qualified = (
        not reasons
        and len(exact_rows) == EXPECTED_CONFIG["requests"]
        and len(paired_speedups) == EXPECTED_CONFIG["requests"])
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "qualified": qualified,
        "model_output_non_regression_authorized": (
            bool(exact_rows) and all(row["qualified"] for row in exact_rows)),
        "endpoint_performance_qualified": qualified,
        "final_output_tps_gate_passed": (
            _finite_positive(candidate_p10)
            and float(candidate_p10) >= FINAL_OUTPUT_TPS_P10),
        "production_promotion_authorized": False,
        "limits": {
            "median_paired_speedup": MIN_MEDIAN_PAIRED_SPEEDUP,
            "p10_ratio": MIN_P10_RATIO,
            "final_output_tps_p10": FINAL_OUTPUT_TPS_P10,
        },
        "observed": {
            "control_output_tps_p10": control_p10,
            "candidate_output_tps_p10": candidate_p10,
            "p10_ratio": p10_ratio,
            "median_paired_speedup": paired_median,
        },
        "requests": exact_rows,
        "reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        json.loads(args.control.read_text(encoding="utf-8")),
        json.loads(args.candidate.read_text(encoding="utf-8")),
    )
    _atomic_write(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
