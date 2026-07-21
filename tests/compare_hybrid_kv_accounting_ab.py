#!/usr/bin/env python3
"""Compare the fixed M1-49 legacy/candidate startup and pressure runs."""

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


TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from compare_cpu_kv_pressure_ab import compare as compare_pressure


SCHEMA = "bi100-hybrid-kv-accounting-ab-v1"
STARTUP_SCHEMA = "bi100-hybrid-kv-startup-v1"
VERSION = 1
MIN_CAPACITY_RATIO = 3.5
MAX_WARM_REGRESSION_RATIO = 1.02
ACCOUNTING_MODE_PLACEHOLDER = "<accounting-mode>"
EXPECTED_FULL_ATTENTION_ORDINALS = [
    3, 7, 11, 15, 19, 23, 27, 31, 35, 39,
]


def _finite_positive(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def _positive_int(value: Any) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)
            and value > 0)


def _validate_startup(
    value: Any,
    *,
    label: str,
    mode: str,
    attention_layers: int,
    reasons: list[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        reasons.append(f"{label} startup must be an object")
        return {}
    if value.get("schema") != STARTUP_SCHEMA or value.get("version") != VERSION:
        reasons.append(f"{label} startup schema is invalid")
    if value.get("qualified") is not True:
        reasons.append(f"{label} startup is not qualified")
    if value.get("mode") != mode or value.get("config_mode") != mode:
        reasons.append(f"{label} startup mode must equal {mode}")
    if value.get("observed_attention_layers") != attention_layers:
        reasons.append(
            f"{label} startup attention layers must equal {attention_layers}")
    if value.get("observed_layer_count") != 40:
        reasons.append(f"{label} startup must describe 40 model layers")
    for field in (
        "observed_gpu_blocks",
        "observed_cpu_blocks",
        "observed_gpu_tokens",
        "required_gpu_blocks",
    ):
        if not _finite_positive(value.get(field)):
            reasons.append(f"{label} startup {field} is invalid")
    max_model_len = value.get("max_model_len_required")
    block_size = value.get("block_size")
    gpu_blocks = value.get("observed_gpu_blocks")
    gpu_tokens = value.get("observed_gpu_tokens")
    required_gpu_blocks = value.get("required_gpu_blocks")
    max_seq_len = value.get("observed_max_seq_len")
    if _positive_int(max_model_len) and _positive_int(block_size):
        expected_required = math.ceil(max_model_len / block_size)
        if required_gpu_blocks != expected_required:
            reasons.append(
                f"{label} required_gpu_blocks is internally inconsistent")
    if _positive_int(gpu_blocks) and _positive_int(block_size):
        if gpu_tokens != gpu_blocks * block_size:
            reasons.append(
                f"{label} observed_gpu_tokens is internally inconsistent")
    if (_positive_int(max_seq_len) and _positive_int(max_model_len)
            and max_seq_len < max_model_len):
        reasons.append(f"{label} observed_max_seq_len is below the contract")

    expected_bytes_per_block = 655_360 if mode == "legacy40" else 163_840
    if value.get("expected_kv_bytes_per_block") != expected_bytes_per_block:
        reasons.append(
            f"{label} expected_kv_bytes_per_block must equal "
            f"{expected_bytes_per_block}")
    if value.get("tensor_parallel_size") != 4:
        reasons.append(f"{label} tensor_parallel_size must equal 4")
    if value.get("rank_kv_heads") != 1 or value.get("head_dim") != 256:
        reasons.append(f"{label} rank-local KV geometry is invalid")
    if value.get("dtype") != "float16" or value.get("dtype_bytes") != 2:
        reasons.append(f"{label} dtype contract must be float16")

    accounting_reports = value.get("runtime_accounting_reports")
    if value.get("full_attention_ordinals") != EXPECTED_FULL_ATTENTION_ORDINALS:
        reasons.append(f"{label} full-attention ordinals are invalid")

    if (not isinstance(accounting_reports, list)
            or len(accounting_reports) != 4):
        reasons.append(f"{label} runtime accounting reports are invalid")
    else:
        observed_ranks = sorted(
            report.get("tp_rank") for report in accounting_reports
            if isinstance(report, dict))
        expected_ranks = [0, 1, 2, 3]
        reports_valid = len(observed_ranks) == 4 and \
            observed_ranks == expected_ranks
        for report in accounting_reports:
            if not isinstance(report, dict):
                reports_valid = False
                continue
            expected_report = {
                "tp_rank": report.get("tp_rank"),
                "env_mode": mode,
                "config_mode": mode,
                "configured_kv_layers": attention_layers,
                "full_attention_layers": 10,
                "full_attention_ordinals": EXPECTED_FULL_ATTENTION_ORDINALS,
            }
            reports_valid = reports_valid and report == expected_report
        if not reports_valid:
            reasons.append(f"{label} runtime accounting reports are invalid")

    runtime_contract = value.get("runtime_contract")
    runtime_digest = value.get("runtime_contract_sha256")
    if not isinstance(runtime_contract, dict):
        reasons.append(f"{label} runtime contract is missing")
    else:
        encoded = json.dumps(
            runtime_contract, ensure_ascii=True, sort_keys=True,
            separators=(",", ":")).encode("ascii")
        expected_digest = hashlib.sha256(encoded).hexdigest()
        if runtime_digest != expected_digest:
            reasons.append(f"{label} runtime contract digest is invalid")
        service = runtime_contract.get("service")
        if (not isinstance(service, dict)
                or service.get("accounting") != mode):
            reasons.append(
                f"{label} runtime contract accounting mode is invalid")
        invariant = _canonical_runtime_contract(runtime_contract)
        invariant_encoded = json.dumps(
            invariant, ensure_ascii=True, sort_keys=True,
            separators=(",", ":")).encode("ascii")
        expected_invariant_digest = hashlib.sha256(
            invariant_encoded).hexdigest()
        if value.get("runtime_contract_invariant_sha256") != \
                expected_invariant_digest:
            reasons.append(
                f"{label} invariant runtime contract digest is invalid")
    digest = value.get("service_log_sha256")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)):
        reasons.append(f"{label} startup service log digest is invalid")
    return value


def _canonical_runtime_contract(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    canonical = json.loads(json.dumps(value, ensure_ascii=True))
    service = canonical.get("service")
    if isinstance(service, dict):
        service["accounting"] = ACCOUNTING_MODE_PLACEHOLDER
    return canonical


def _startup_invariants(value: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "observed_layer_count",
        "full_attention_ordinals",
        "max_model_len_required",
        "block_size",
        "required_gpu_blocks",
        "observed_max_seq_len",
        "num_key_value_heads",
        "rank_kv_heads",
        "head_dim",
        "tensor_parallel_size",
        "dtype",
        "dtype_bytes",
    )
    invariant = {field: value.get(field) for field in fields}
    invariant["runtime_contract"] = _canonical_runtime_contract(
        value.get("runtime_contract"))
    return invariant


def _validate_pressure_startup_contract(
    pressure: Any,
    startup: dict[str, Any],
    label: str,
    reasons: list[str],
) -> None:
    if not isinstance(pressure, dict) or not isinstance(pressure.get("params"), dict):
        reasons.append(f"{label} pressure params are missing")
        return
    params = pressure["params"]
    if params.get("block_size") != startup.get("block_size"):
        reasons.append(f"{label} pressure block size differs from startup")
    max_model_len = startup.get("max_model_len_required")
    max_tokens = params.get("max_tokens")
    for field in ("target_prompt_tokens", "pressure_prompt_tokens"):
        tokens = params.get(field)
        if (_positive_int(tokens) and _positive_int(max_tokens)
                and _positive_int(max_model_len)
                and tokens + max_tokens > max_model_len):
            reasons.append(
                f"{label} pressure {field} exceeds startup max model length")


def _capacity_ratio(
    legacy: dict[str, Any],
    candidate: dict[str, Any],
    field: str,
    reasons: list[str],
) -> float | None:
    left = legacy.get(field)
    right = candidate.get(field)
    if not _finite_positive(left) or not _finite_positive(right):
        return None
    ratio = right / left
    if ratio < MIN_CAPACITY_RATIO:
        reasons.append(
            f"candidate {field} ratio {ratio:.6f} is below "
            f"{MIN_CAPACITY_RATIO:.1f}")
    return ratio


def _find_pressure_row(report: dict[str, Any], name: str) -> dict[str, Any] | None:
    rows = report.get("requests")
    if not isinstance(rows, list):
        return None
    return next(
        (row for row in rows
         if isinstance(row, dict) and row.get("name") == name),
        None,
    )


def compare(
    legacy_startup: Any,
    candidate_startup: Any,
    legacy_pressure: Any,
    candidate_pressure: Any,
) -> dict[str, Any]:
    reasons: list[str] = []
    legacy = _validate_startup(
        legacy_startup,
        label="legacy",
        mode="legacy40",
        attention_layers=40,
        reasons=reasons,
    )
    candidate = _validate_startup(
        candidate_startup,
        label="candidate",
        mode="full_attention",
        attention_layers=10,
        reasons=reasons,
    )
    for field in ("max_model_len_required", "block_size", "required_gpu_blocks"):
        if legacy.get(field) != candidate.get(field):
            reasons.append(f"startup {field} differs between arms")
    legacy_contract = legacy.get("runtime_contract")
    candidate_contract = candidate.get("runtime_contract")
    if _startup_invariants(legacy) != _startup_invariants(candidate):
        reasons.append("startup invariants differ outside accounting capacity")
    if _canonical_runtime_contract(legacy_contract) != \
            _canonical_runtime_contract(candidate_contract):
        reasons.append("runtime contract differs outside accounting mode")
    if legacy.get("runtime_contract_invariant_sha256") != candidate.get(
            "runtime_contract_invariant_sha256"):
        reasons.append("invariant runtime contract digest differs between arms")
    if legacy.get("runtime_contract_sha256") == candidate.get(
            "runtime_contract_sha256"):
        reasons.append("runtime contract does not capture accounting mode")

    _validate_pressure_startup_contract(
        legacy_pressure, legacy, "legacy", reasons)
    _validate_pressure_startup_contract(
        candidate_pressure, candidate, "candidate", reasons)

    gpu_ratio = _capacity_ratio(
        legacy, candidate, "observed_gpu_blocks", reasons)
    cpu_ratio = _capacity_ratio(
        legacy, candidate, "observed_cpu_blocks", reasons)

    pressure = compare_pressure(legacy_pressure, candidate_pressure)
    if pressure.get("qualified") is not True:
        reasons.extend(
            f"pressure: {reason}"
            for reason in pressure.get("reasons", ["comparison failed"])
        )

    immediate_warm = _find_pressure_row(pressure, "target_immediate_warm")
    warm_ratio = (
        immediate_warm.get("elapsed_ratio")
        if isinstance(immediate_warm, dict) else None)
    if not _finite_positive(warm_ratio):
        reasons.append("pressure comparison is missing immediate-warm ratio")
    elif warm_ratio > MAX_WARM_REGRESSION_RATIO:
        reasons.append(
            f"immediate-warm ratio {warm_ratio:.6f} exceeds "
            f"{MAX_WARM_REGRESSION_RATIO:.2f}")

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "qualified": not reasons,
        "reasons": reasons,
        "gates": {
            "minimum_capacity_ratio": MIN_CAPACITY_RATIO,
            "maximum_warm_regression_ratio": MAX_WARM_REGRESSION_RATIO,
        },
        "capacity": {
            "legacy_gpu_blocks": legacy.get("observed_gpu_blocks"),
            "candidate_gpu_blocks": candidate.get("observed_gpu_blocks"),
            "gpu_block_ratio": gpu_ratio,
            "legacy_cpu_blocks": legacy.get("observed_cpu_blocks"),
            "candidate_cpu_blocks": candidate.get("observed_cpu_blocks"),
            "cpu_block_ratio": cpu_ratio,
        },
        "immediate_warm_ratio": warm_ratio,
        "startup": {
            "legacy": legacy,
            "candidate": candidate,
        },
        "pressure": pressure,
    }


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
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
    parser.add_argument("--legacy-startup", type=Path, required=True)
    parser.add_argument("--candidate-startup", type=Path, required=True)
    parser.add_argument("--legacy-pressure", type=Path, required=True)
    parser.add_argument("--candidate-pressure", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = compare(
            _load(args.legacy_startup),
            _load(args.candidate_startup),
            _load(args.legacy_pressure),
            _load(args.candidate_pressure),
        )
    except Exception as error:
        report = {
            "schema": SCHEMA,
            "version": VERSION,
            "qualified": False,
            "reasons": [f"cannot compare M1-49 evidence: {error}"],
        }
    _atomic_write(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
