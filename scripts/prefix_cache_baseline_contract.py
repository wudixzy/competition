#!/usr/bin/env python3
"""Validate attested inputs for an 881-request prefix-cache projection."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import string
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import quality_runtime_contract as runtime_contract  # noqa: E402


Json = dict[str, Any]
BASELINE_SCHEMA = "bi100-prefix-cache-baseline-contract-v1"
WORKLOAD_SCHEMA = "bi100-private-881-workload-manifest-v1"
EXPECTED_REQUESTS = 881
EXPECTED_TRACE_VERSION = 4
EXPECTED_BLOCK_SIZE = 16
SCORE_FORMULA = (
    "output_tps_p10*16.796+input_tps*2.799+cache_tps*0.56"
)

BASELINE_ENVIRONMENT = {
    "BI100_ATTN_COREX_FUSED_PREFILL": "0",
    "BI100_GDN_CACHE_POLICY": "fine32",
    "BI100_GDN_RESTORE_MODE": "direct",
    "BI100_KV_EVICTION_POLICY": "lru",
}

WORKLOAD_FIELDS = {
    "schema", "version", "workload_kind", "name", "author_or_org",
    "source_url", "license", "revision", "captured_at_utc", "split",
    "request_count", "request_order_sha256", "source_artifact_sha256",
    "source_artifact_kind", "selection_rule", "transformation",
    "redistribution_allowed", "contains_restricted_evaluation_data",
    "snapshot_redistributed",
}
METRICS_FIELDS = {
    "score_kind", "aggregation", "attempted_requests",
    "successful_requests", "error_requests", "output_tps_p10",
    "input_tps", "cache_tps", "ttft_p90_s", "cache_hit_rate",
    "success_rate", "weighted_score", "formula",
}
BASELINE_FIELDS = {
    "schema", "version", "run_id", "runtime_contract",
    "workload_manifest", "trace", "metrics", "metrics_source",
    "metrics_transformation", "attestation",
}
WRAPPER_FIELDS = {"sha256", "file_sha256", "value"}
TRACE_FIELDS = {
    "version", "session_sha256", "request_count", "block_size",
    "request_order_sha256", "records_sha256", "logs",
}
ARTIFACT_FIELDS = {"bytes", "sha256"}
ATTESTATION_FIELDS = {
    "trace_metrics_same_service_run_asserted",
    "metrics_cover_exact_trace_request_order_asserted",
    "contains_raw_requests_or_outputs",
    "qualification_scope",
}


class BaselineContractError(RuntimeError):
    pass


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise BaselineContractError(reason)


def _nonempty(value: Any, field: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()),
             f"{field} must be a non-empty string")
    return value.strip()


def _finite(value: Any, field: str, *, minimum: float = 0.0) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and value >= minimum,
        f"{field} must be finite and at least {minimum}",
    )
    return float(value)


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in string.hexdigits for character in value)
    )


def sha256_json(value: Any) -> str:
    canonical = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> Json:
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def request_order_sha256(records: Sequence[Json]) -> str:
    return sha256_json([
        record.get("request_id_sha256") for record in records
    ])


def trace_records_sha256(records: Sequence[Json]) -> str:
    public_records = [
        {
            key: value for key, value in record.items()
            if not key.startswith("_")
        }
        for record in records
    ]
    return sha256_json(public_records)


def trace_identity(records: Sequence[Json], log_paths: Sequence[Path]) -> Json:
    _require(len(records) == EXPECTED_REQUESTS,
             f"trace must contain exactly {EXPECTED_REQUESTS} requests")
    _require(
        [record.get("ordinal") for record in records]
        == list(range(1, EXPECTED_REQUESTS + 1)),
        "trace request order is not contiguous",
    )
    versions = {record.get("version") for record in records}
    _require(versions == {EXPECTED_TRACE_VERSION},
             "trace version is not the fixed v4 contract")
    sessions = {record.get("trace_session_sha256") for record in records}
    _require(len(sessions) == 1 and _is_hex(next(iter(sessions)), 16),
             "trace session identity is invalid")
    block_sizes = {record.get("block_size") for record in records}
    _require(block_sizes == {EXPECTED_BLOCK_SIZE},
             "trace block size is not the fixed 16-token contract")
    _require(len(log_paths) > 0, "at least one trace log is required")
    return {
        "version": EXPECTED_TRACE_VERSION,
        "session_sha256": next(iter(sessions)).lower(),
        "request_count": EXPECTED_REQUESTS,
        "block_size": EXPECTED_BLOCK_SIZE,
        "request_order_sha256": request_order_sha256(records),
        "records_sha256": trace_records_sha256(records),
        "logs": [artifact(path) for path in log_paths],
    }


def validate_workload_manifest(
    value: Any,
    *,
    expected_trace: Json | None = None,
) -> str:
    _require(isinstance(value, dict) and set(value) == WORKLOAD_FIELDS,
             "workload manifest fields are invalid")
    _require(value["schema"] == WORKLOAD_SCHEMA and value["version"] == 1,
             "workload manifest schema or version is invalid")
    _require(value["workload_kind"] == "restricted_official_881",
             "workload manifest is not the restricted official 881 workload")
    for field in (
        "name", "author_or_org", "license", "revision", "captured_at_utc",
        "split", "source_artifact_kind", "selection_rule", "transformation",
    ):
        _nonempty(value[field], f"workload manifest {field}")
    _require(value["revision"].strip().lower() != "latest",
             "workload manifest revision cannot be latest")
    _require(
        value["source_url"] is None
        or isinstance(value["source_url"], str) and bool(value["source_url"]),
        "workload manifest source_url must be null or non-empty",
    )
    _require(value["request_count"] == EXPECTED_REQUESTS,
             "workload manifest request count is not 881")
    for field in ("request_order_sha256", "source_artifact_sha256"):
        _require(runtime_contract.is_sha256(value[field]),
                 f"workload manifest {field} is invalid")
    _require(value["redistribution_allowed"] is False,
             "restricted workload cannot be marked redistributable")
    _require(value["contains_restricted_evaluation_data"] is True,
             "workload restriction marker is missing")
    _require(value["snapshot_redistributed"] is False,
             "restricted workload snapshot must not be redistributed")
    if expected_trace is not None:
        _require(
            value["request_order_sha256"]
            == expected_trace["request_order_sha256"],
            "workload request order does not match the trace",
        )
        _require(
            value["source_artifact_sha256"]
            == expected_trace["records_sha256"],
            "workload source artifact does not match the trace records",
        )
    _reject_credentials(value, "workload manifest")
    return sha256_json(value)


def validate_runtime_contract(value: Any) -> str:
    _require(isinstance(value, dict), "runtime contract must be an object")
    expected = {
        "source_revision": value.get("source_revision"),
        "runtime_identity": value.get("runtime_identity"),
        "runtime_overlay_sha256": value.get("runtime_overlay_sha256"),
        "instance": value.get("instance"),
        "gpu_count": 4,
        "tensor_parallel_size": 4,
        "max_model_len": 262144,
        "model_path": value.get("model_path"),
        "tokenizer_path": value.get("tokenizer_path"),
        "served_model_name": "llm",
    }
    try:
        digest = runtime_contract.validate_runtime_contract(
            value, expected, require_cache_trace=True)
    except runtime_contract.RuntimeContractError as error:
        raise BaselineContractError(
            f"runtime contract is invalid: {error}") from error
    _require(value["model_path"] == value["tokenizer_path"],
             "runtime model and tokenizer paths differ")
    environment = value["environment"]
    for name, expected_value in BASELINE_ENVIRONMENT.items():
        _require(
            environment.get(name) == expected_value,
            f"baseline runtime requires {name}={expected_value}",
        )
    return digest


def weighted_score(metrics: Json) -> float:
    return (
        float(metrics["output_tps_p10"]) * 16.796
        + float(metrics["input_tps"]) * 2.799
        + float(metrics["cache_tps"]) * 0.56
    )


def validate_metrics(value: Any) -> Json:
    _require(isinstance(value, dict) and set(value) == METRICS_FIELDS,
             "baseline metrics fields are invalid")
    _require(value["score_kind"] in {"local_881_proxy", "official_platform"},
             "baseline metrics score_kind is invalid")
    _nonempty(value["aggregation"], "baseline metrics aggregation")
    _require(value["formula"] == SCORE_FORMULA,
             "baseline metrics score formula is invalid")
    for field in (
        "attempted_requests", "successful_requests", "error_requests",
    ):
        _require(
            isinstance(value[field], int) and not isinstance(value[field], bool)
            and value[field] >= 0,
            f"baseline metrics {field} must be a non-negative integer",
        )
    _require(value["attempted_requests"] == EXPECTED_REQUESTS,
             "baseline metrics do not cover exactly 881 requests")
    _require(
        value["successful_requests"] + value["error_requests"]
        == value["attempted_requests"],
        "baseline metrics request counts are inconsistent",
    )
    for field in (
        "output_tps_p10", "input_tps", "cache_tps", "ttft_p90_s",
        "cache_hit_rate", "success_rate", "weighted_score",
    ):
        _finite(value[field], f"baseline metrics {field}")
    for field in ("cache_hit_rate", "success_rate"):
        _require(value[field] <= 1.0,
                 f"baseline metrics {field} must be at most one")
    expected_success = (
        value["successful_requests"] / value["attempted_requests"])
    _require(
        math.isclose(value["success_rate"], expected_success,
                     rel_tol=0.0, abs_tol=1e-12),
        "baseline metrics success rate is inconsistent with request counts",
    )
    expected_score = weighted_score(value)
    _require(
        math.isclose(value["weighted_score"], expected_score,
                     rel_tol=1e-6, abs_tol=1e-3),
        "baseline metrics weighted score is inconsistent with the formula",
    )
    return value


def _validate_artifact(value: Any, field: str) -> None:
    _require(isinstance(value, dict) and set(value) == ARTIFACT_FIELDS,
             f"{field} artifact fields are invalid")
    _require(
        isinstance(value["bytes"], int) and not isinstance(value["bytes"], bool)
        and value["bytes"] >= 0,
        f"{field} artifact size is invalid",
    )
    _require(runtime_contract.is_sha256(value["sha256"]),
             f"{field} artifact SHA-256 is invalid")


def _reject_credentials(value: Any, field: str) -> None:
    serialized = json.dumps(value, ensure_ascii=True).lower()
    markers = (
        "begin openssh private key", "github_pat_", "ghp_",
        "modelhub_access_token", "proxy-authorization", "password=",
    )
    _require(not any(marker in serialized for marker in markers),
             f"{field} contains a credential marker")


def validate_baseline_contract(
    value: Any,
    *,
    expected_trace: Json | None = None,
) -> str:
    _require(isinstance(value, dict) and set(value) == BASELINE_FIELDS,
             "baseline contract fields are invalid")
    _require(value["schema"] == BASELINE_SCHEMA and value["version"] == 1,
             "baseline contract schema or version is invalid")
    _nonempty(value["run_id"], "baseline contract run_id")

    runtime_wrapper = value["runtime_contract"]
    _require(
        isinstance(runtime_wrapper, dict)
        and set(runtime_wrapper) == WRAPPER_FIELDS,
        "runtime contract wrapper fields are invalid",
    )
    runtime_digest = validate_runtime_contract(runtime_wrapper["value"])
    _require(runtime_wrapper["sha256"] == runtime_digest,
             "runtime contract canonical SHA-256 is inconsistent")
    _require(runtime_contract.is_sha256(runtime_wrapper["file_sha256"]),
             "runtime contract file SHA-256 is invalid")

    workload_wrapper = value["workload_manifest"]
    _require(
        isinstance(workload_wrapper, dict)
        and set(workload_wrapper) == WRAPPER_FIELDS,
        "workload manifest wrapper fields are invalid",
    )
    workload_digest = validate_workload_manifest(
        workload_wrapper["value"], expected_trace=expected_trace)
    _require(workload_wrapper["sha256"] == workload_digest,
             "workload manifest canonical SHA-256 is inconsistent")
    _require(runtime_contract.is_sha256(workload_wrapper["file_sha256"]),
             "workload manifest file SHA-256 is invalid")

    trace = value["trace"]
    _require(isinstance(trace, dict) and set(trace) == TRACE_FIELDS,
             "baseline trace identity fields are invalid")
    _require(
        trace["version"] == EXPECTED_TRACE_VERSION
        and trace["request_count"] == EXPECTED_REQUESTS
        and trace["block_size"] == EXPECTED_BLOCK_SIZE,
        "baseline trace identity is not the fixed 881/v4/block16 contract",
    )
    _require(_is_hex(trace["session_sha256"], 16),
             "baseline trace session identity is invalid")
    for field in ("request_order_sha256", "records_sha256"):
        _require(runtime_contract.is_sha256(trace[field]),
                 f"baseline trace {field} is invalid")
    _require(isinstance(trace["logs"], list) and bool(trace["logs"]),
             "baseline trace log artifacts are missing")
    for index, item in enumerate(trace["logs"]):
        _validate_artifact(item, f"baseline trace log {index}")
    if expected_trace is not None:
        _require(trace == expected_trace,
                 "baseline contract trace identity does not match input logs")

    validate_metrics(value["metrics"])
    _validate_artifact(value["metrics_source"], "baseline metrics source")
    _nonempty(
        value["metrics_transformation"],
        "baseline metrics transformation",
    )

    attestation = value["attestation"]
    _require(
        isinstance(attestation, dict)
        and set(attestation) == ATTESTATION_FIELDS,
        "baseline attestation fields are invalid",
    )
    _require(attestation["trace_metrics_same_service_run_asserted"] is True,
             "trace and metrics same-run attestation is missing")
    _require(
        attestation["metrics_cover_exact_trace_request_order_asserted"] is True,
        "metrics request-order attestation is missing",
    )
    _require(attestation["contains_raw_requests_or_outputs"] is False,
             "baseline contract cannot contain raw requests or outputs")
    _require(
        attestation["qualification_scope"]
        == "offline_cache_phase_gate_only",
        "baseline attestation scope is invalid",
    )
    _reject_credentials(value, "baseline contract")
    return sha256_json(value)


def load_baseline_contract(
    path: Path,
    *,
    expected_trace: Json | None = None,
) -> tuple[Json, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    digest = validate_baseline_contract(
        value, expected_trace=expected_trace)
    return value, digest
