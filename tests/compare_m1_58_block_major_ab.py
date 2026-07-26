#!/usr/bin/env python3
"""Qualify the fixed M1-58 TP4 block-major CacheEngine A/B."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import tempfile
from pathlib import Path
import re
from typing import Any


Json = dict[str, Any]
SCHEMA = "bi100-m1-58-block-major-ab-v1"
VERSION = 1
STARTUP_SCHEMA = "bi100-hybrid-kv-startup-v1"
PRESSURE_SCHEMA = "bi100-cpu-kv-offload-pressure-api-v1"
RUNTIME_SCHEMA = "bi100-bare-host-runtime-identity-v1"
PREFLIGHT_SCHEMA = "bi100-gpu-preflight-comparison-v1"
TARGET_TOKENS = 65_536
PRESSURE_TOKENS = 135_040
PRESSURE_COUNT = 9
MAX_TOKENS = 8
BLOCK_SIZE = 16
MIN_CACHED_TOKENS = 65_504
MIN_RESTORE_SPEEDUP = 1.20
MAX_NON_TRANSFER_RATIO = 1.02
MAX_CAPACITY_DELTA_DRIFT_BLOCKS = 32
EXPECTED_RUN_ID = "m158-block-major-fixed-20260726"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class InputError(ValueError):
    pass


def _load(path: Path) -> Json:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InputError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise InputError(f"{path} must contain one JSON object")
    return value


def _int(value: Any, minimum: int = 0) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)
            and value >= minimum)


def _finite(value: Any, *, positive: bool = False) -> bool:
    return (isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and (not positive or value > 0))


def _validate_qualified(
    value: Any,
    label: str,
    schema: str,
    reasons: list[str],
) -> Json:
    if not isinstance(value, dict):
        reasons.append(f"{label} must be an object")
        return {}
    if value.get("schema") != schema or value.get("version") != VERSION:
        reasons.append(f"{label} schema is invalid")
    if value.get("qualified") is not True:
        reasons.append(f"{label} is not qualified")
    if value.get("reasons") not in ([], None):
        reasons.append(f"{label} contains failure reasons")
    return value


def _validate_lifecycle(
    runtime_identity: Any,
    preflights: Any,
    reasons: list[str],
) -> tuple[Json, Json]:
    runtime = _validate_qualified(
        runtime_identity, "runtime identity", RUNTIME_SCHEMA, reasons)
    preflight = _validate_qualified(
        preflights, "GPU preflight comparison", PREFLIGHT_SCHEMA, reasons)
    stages = preflight.get("stages")
    labels = (
        [stage.get("label") for stage in stages]
        if isinstance(stages, list)
        and all(isinstance(stage, dict) for stage in stages)
        else None
    )
    if labels != ["before_control", "after_control", "after_candidate"]:
        reasons.append("GPU preflight stages are incomplete or out of order")
    return runtime, preflight


def _startup_service(report: Json, label: str,
                     reasons: list[str]) -> Json:
    runtime = report.get("runtime_contract")
    if not isinstance(runtime, dict):
        reasons.append(f"{label} runtime contract is missing")
        return {}
    service = runtime.get("service")
    if not isinstance(service, dict):
        reasons.append(f"{label} service contract is missing")
        return {}
    return service


def _validate_startups(
    control: Any,
    candidate: Any,
    reasons: list[str],
) -> tuple[Json, Json, Json]:
    control_report = _validate_qualified(
        control, "control startup", STARTUP_SCHEMA, reasons)
    candidate_report = _validate_qualified(
        candidate, "candidate startup", STARTUP_SCHEMA, reasons)
    for label, report, selector in (
        ("control", control_report, "0"),
        ("candidate", candidate_report, "1"),
    ):
        expected = {
            "mode": "full_attention",
            "config_mode": "full_attention",
            "expected_attention_layers": 10,
            "observed_attention_layers": 10,
            "observed_layer_count": 40,
            "dtype": "float16",
            "expected_kv_bytes_per_block": 163_840,
            "max_model_len_required": 262_144,
            "block_size": BLOCK_SIZE,
            "required_gpu_blocks": 16_384,
            "observed_max_seq_len": 262_144,
            "block_major_cpu_kv": selector == "1",
        }
        for field, value in expected.items():
            if report.get(field) != value:
                reasons.append(
                    f"{label} startup {field} must equal {value!r}")
        if not _int(report.get("observed_gpu_blocks"), 16_384):
            reasons.append(f"{label} startup GPU capacity is insufficient")
        if not _int(report.get("observed_cpu_blocks"), 1):
            reasons.append(f"{label} startup CPU capacity is invalid")
        service = _startup_service(report, label, reasons)
        expected_service = {
            "accounting": "full_attention",
            "cpu_kv_offload": "1",
            "block_major_cpu_kv": selector,
            "block_major_cpu_kv_trace": "0",
            "cache_trace": "0",
            "gdn_cache_policy": "admission64",
            "gdn_restore_mode": "direct",
            "fused_prefill": "0",
            "kv_eviction_policy": "lru",
            "max_model_len": "262144",
            "tensor_parallel_size": "4",
            "max_num_seqs": "1",
            "max_num_batched_tokens": "8192",
        }
        for field, value in expected_service.items():
            if service.get(field) != value:
                reasons.append(
                    f"{label} service {field} must equal {value!r}")

    control_capacity = control_report.get("block_major_capacity_reports")
    control_cache = control_report.get("block_major_cache_reports")
    if control_capacity != [] or control_cache != []:
        reasons.append("control startup contains block-major reports")
    candidate_capacity = candidate_report.get(
        "block_major_capacity_reports")
    candidate_cache = candidate_report.get("block_major_cache_reports")
    if not isinstance(candidate_capacity, list) or len(candidate_capacity) != 4:
        reasons.append("candidate must contain four capacity reports")
    if not isinstance(candidate_cache, list) or len(candidate_cache) != 4:
        reasons.append("candidate must contain four cache reports")

    control_runtime = copy.deepcopy(control_report.get("runtime_contract"))
    candidate_runtime = copy.deepcopy(candidate_report.get("runtime_contract"))
    if isinstance(control_runtime, dict) and isinstance(candidate_runtime, dict):
        for value in (control_runtime, candidate_runtime):
            service = value.get("service")
            if isinstance(service, dict):
                service["block_major_cpu_kv"] = "<selector>"
        if control_runtime != candidate_runtime:
            reasons.append(
                "startup runtime contracts differ outside block-major selector")

    control_blocks = control_report.get("observed_gpu_blocks")
    candidate_blocks = candidate_report.get("observed_gpu_blocks")
    capacity_delta = None
    if _int(control_blocks, 1) and _int(candidate_blocks, 1):
        capacity_delta = control_blocks - candidate_blocks
        lower = 1024 - MAX_CAPACITY_DELTA_DRIFT_BLOCKS
        upper = 1024 + MAX_CAPACITY_DELTA_DRIFT_BLOCKS
        if not lower <= capacity_delta <= upper:
            reasons.append(
                "candidate GPU block delta is inconsistent with the "
                f"1024-block reserve: {capacity_delta}")
    else:
        reasons.append("startup GPU block counts are invalid")
    if (control_report.get("observed_cpu_blocks")
            != candidate_report.get("observed_cpu_blocks")):
        reasons.append("CPU block capacity differs across A/B arms")

    return control_report, candidate_report, {
        "control_gpu_blocks": control_blocks,
        "candidate_gpu_blocks": candidate_blocks,
        "gpu_block_delta": capacity_delta,
    }


def _request_sequence() -> list[str]:
    return [
        "target_cold",
        "target_immediate_warm",
        *(f"pressure_cold_{index:04d}"
          for index in range(PRESSURE_COUNT)),
        "target_after_pressure",
        "target_refreshed",
    ]


def _validate_pressure(
    value: Any,
    label: str,
    reasons: list[str],
) -> tuple[Json, dict[str, Json]]:
    report = _validate_qualified(
        value, f"{label} pressure", PRESSURE_SCHEMA, reasons)
    validation = report.get("validation")
    if (not isinstance(validation, dict)
            or validation.get("qualified") is not True
            or validation.get("reasons") not in ([], None)):
        reasons.append(f"{label} pressure validation is not qualified")
    params = report.get("params")
    expected_params = {
        "target_prompt_tokens": TARGET_TOKENS,
        "pressure_prompt_tokens": PRESSURE_TOKENS,
        "pressure_count": PRESSURE_COUNT,
        "max_tokens": MAX_TOKENS,
        "timeout_s": 900.0,
        "run_id": EXPECTED_RUN_ID,
        "mode": "candidate",
        "block_size": BLOCK_SIZE,
        "min_candidate_cached": MIN_CACHED_TOKENS,
        "max_control_cached": 16,
    }
    if not isinstance(params, dict):
        reasons.append(f"{label} pressure params are missing")
        params = {}
    for field, expected in expected_params.items():
        if params.get(field) != expected:
            reasons.append(
                f"{label} pressure {field} must equal {expected!r}")

    requests = report.get("requests")
    result: dict[str, Json] = {}
    if not isinstance(requests, list):
        reasons.append(f"{label} requests must be a list")
        return report, result
    names = []
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            reasons.append(f"{label} request {index} is not an object")
            continue
        name = request.get("name")
        names.append(name)
        if not isinstance(name, str) or name in result:
            reasons.append(f"{label} request names are invalid")
            continue
        result[name] = request
        if request.get("status") != "ok":
            reasons.append(f"{label} request {name} is not ok")
            continue
        summary = request.get("summary")
        if not isinstance(summary, dict):
            reasons.append(f"{label} request {name} summary is missing")
            continue
        for field in ("prompt_tokens", "cached_tokens", "completion_tokens"):
            if not _int(summary.get(field), 0):
                reasons.append(
                    f"{label} request {name} {field} is invalid")
        if not _finite(summary.get("elapsed_s"), positive=True):
            reasons.append(
                f"{label} request {name} elapsed_s is invalid")
        if summary.get("finish_reason") not in ("stop", "length"):
            reasons.append(
                f"{label} request {name} finish_reason is invalid")
        digest = summary.get("message_sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            reasons.append(
                f"{label} request {name} message digest is invalid")
    if names != _request_sequence():
        reasons.append(f"{label} request sequence is incomplete or changed")
    return report, result


def _summary(requests: dict[str, Json], name: str) -> Json | None:
    request = requests.get(name)
    summary = request.get("summary") if isinstance(request, dict) else None
    return summary if isinstance(summary, dict) else None


def _elapsed_sum(requests: dict[str, Json],
                 names: list[str]) -> float | None:
    values = []
    for name in names:
        summary = _summary(requests, name)
        elapsed = summary.get("elapsed_s") if summary else None
        if not _finite(elapsed, positive=True):
            return None
        values.append(float(elapsed))
    return sum(values)


def _validate_pressure_ab(
    control: Any,
    candidate: Any,
    reasons: list[str],
) -> tuple[list[Json], Json]:
    control_report, control_requests = _validate_pressure(
        control, "control", reasons)
    candidate_report, candidate_requests = _validate_pressure(
        candidate, "candidate", reasons)
    control_params = control_report.get("params")
    candidate_params = candidate_report.get("params")
    if isinstance(control_params, dict) and isinstance(candidate_params, dict):
        left = dict(control_params)
        right = dict(candidate_params)
        left.pop("json_out", None)
        right.pop("json_out", None)
        if left != right:
            reasons.append("pressure params differ outside json_out")

    rows = []
    for name in _request_sequence():
        left = _summary(control_requests, name)
        right = _summary(candidate_requests, name)
        if left is None or right is None:
            continue
        for field in (
            "prompt_tokens",
            "cached_tokens",
            "completion_tokens",
            "finish_reason",
            "message_sha256",
        ):
            if left.get(field) != right.get(field):
                reasons.append(f"request {name} {field} differs across arms")
        rows.append({
            "name": name,
            "control_elapsed_s": left.get("elapsed_s"),
            "candidate_elapsed_s": right.get("elapsed_s"),
            "elapsed_ratio": (
                right["elapsed_s"] / left["elapsed_s"]
                if _finite(left.get("elapsed_s"), positive=True)
                and _finite(right.get("elapsed_s"), positive=True)
                else None
            ),
            "cached_tokens": left.get("cached_tokens"),
            "message_sha256": left.get("message_sha256"),
        })

    for label, requests in (
        ("control", control_requests),
        ("candidate", candidate_requests),
    ):
        cold = _summary(requests, "target_cold")
        after = _summary(requests, "target_after_pressure")
        if cold is None or cold.get("cached_tokens") != 0:
            reasons.append(f"{label} target_cold is not cache-cold")
        if after is None or after.get("cached_tokens", -1) < MIN_CACHED_TOKENS:
            reasons.append(
                f"{label} target_after_pressure did not restore the prefix")

    control_after = _summary(control_requests, "target_after_pressure")
    candidate_after = _summary(candidate_requests, "target_after_pressure")
    restore_ratio = None
    if control_after is not None and candidate_after is not None:
        left = control_after.get("elapsed_s")
        right = candidate_after.get("elapsed_s")
        if _finite(left, positive=True) and _finite(right, positive=True):
            restore_ratio = right / left
            if restore_ratio > 1.0 / MIN_RESTORE_SPEEDUP:
                reasons.append(
                    "block-major restored request speedup is below "
                    f"{MIN_RESTORE_SPEEDUP:.2f}x")

    cold_names = [
        "target_cold",
        *(f"pressure_cold_{index:04d}"
          for index in range(PRESSURE_COUNT)),
    ]
    warm_names = ["target_immediate_warm", "target_refreshed"]
    control_cold = _elapsed_sum(control_requests, cold_names)
    candidate_cold = _elapsed_sum(candidate_requests, cold_names)
    control_warm = _elapsed_sum(control_requests, warm_names)
    candidate_warm = _elapsed_sum(candidate_requests, warm_names)
    cold_ratio = (
        candidate_cold / control_cold
        if control_cold and candidate_cold is not None else None)
    warm_ratio = (
        candidate_warm / control_warm
        if control_warm and candidate_warm is not None else None)
    if cold_ratio is None or cold_ratio > MAX_NON_TRANSFER_RATIO:
        reasons.append(
            "aggregate cold path regressed by more than 2%")
    if warm_ratio is None or warm_ratio > MAX_NON_TRANSFER_RATIO:
        reasons.append(
            "aggregate GPU-warm path regressed by more than 2%")

    return rows, {
        "restore_elapsed_ratio": restore_ratio,
        "restore_speedup": (
            1.0 / restore_ratio if restore_ratio and restore_ratio > 0
            else None),
        "cold_elapsed_ratio": cold_ratio,
        "gpu_warm_elapsed_ratio": warm_ratio,
    }


def compare(
    control_startup: Any,
    candidate_startup: Any,
    control_pressure: Any,
    candidate_pressure: Any,
    runtime_identity: Any,
    preflight_comparison: Any,
) -> Json:
    reasons: list[str] = []
    runtime, preflight = _validate_lifecycle(
        runtime_identity, preflight_comparison, reasons)
    _, _, capacity = _validate_startups(
        control_startup, candidate_startup, reasons)
    rows, performance = _validate_pressure_ab(
        control_pressure, candidate_pressure, reasons)
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "qualified": not reasons,
        "reasons": reasons,
        "source_revision": runtime.get("source_revision"),
        "runtime_tree_sha256": runtime.get("runtime_tree_sha256"),
        "preflight_stage_count": len(preflight.get("stages", [])),
        "capacity": capacity,
        "performance": performance,
        "requests": rows,
        "limits": {
            "min_restore_speedup": MIN_RESTORE_SPEEDUP,
            "max_non_transfer_ratio": MAX_NON_TRANSFER_RATIO,
            "max_capacity_delta_drift_blocks": (
                MAX_CAPACITY_DELTA_DRIFT_BLOCKS),
        },
    }


def _atomic_write(path: Path, value: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-startup", type=Path, required=True)
    parser.add_argument("--candidate-startup", type=Path, required=True)
    parser.add_argument("--control-pressure", type=Path, required=True)
    parser.add_argument("--candidate-pressure", type=Path, required=True)
    parser.add_argument("--runtime-identity", type=Path, required=True)
    parser.add_argument("--preflight-comparison", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = compare(
            _load(args.control_startup),
            _load(args.candidate_startup),
            _load(args.control_pressure),
            _load(args.candidate_pressure),
            _load(args.runtime_identity),
            _load(args.preflight_comparison),
        )
    except InputError as error:
        report = {
            "schema": SCHEMA,
            "version": VERSION,
            "qualified": False,
            "reasons": [str(error)],
            "requests": [],
        }
    _atomic_write(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
