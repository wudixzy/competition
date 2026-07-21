#!/usr/bin/env python3
"""Build a privacy-safe qualification record for M1-49 long-context gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


SCHEMA = "bi100-m1-49-long-context-qualification-v1"
VERSION = 1
AB_SCHEMA = "bi100-hybrid-kv-accounting-ab-v1"
STARTUP_SCHEMA = "bi100-hybrid-kv-startup-v1"
PREFLIGHT_SCHEMA = "bi100-gpu-preflight-comparison-v1"
LONG_SCHEMA = "bi100-long-context-safe-gate-v1"
MULTIMODAL_SCHEMA = "bi100-multimodal-prefix-isolation-qualification-v2"
EXPECTED_SMOKE_TESTS = (
    "test_basic_chat",
    "test_thinking_disabled_variants",
    "test_tool_choice_none",
    "test_response_format_json_object",
    "test_response_format_json_schema",
    "test_streaming_sse",
    "test_bad_request_4xx",
    "test_prefix_cache",
)
EXPECTED_MULTIMODAL_CHECKS = {
    "cold_has_no_hit",
    "different_image_isolated",
    "red_cold_warm_exact",
    "same_image_hits",
    "semantic_colors_observed",
}
MAX_FREE_MEMORY_DROP_BYTES = 1_073_741_824
EXPECTED_LONG_CONTRACTS = {
    "long_131k_exact": {
        "target_prompt_tokens": 131_000,
        "max_tokens": 256,
        "min_cached_tokens": 130_992,
        "min_completion_tokens": 256,
        "equivalence_mode": "exact",
    },
    "long_235k_warm_repeat": {
        "target_prompt_tokens": 235_000,
        "max_tokens": 1_000,
        "min_cached_tokens": 234_992,
        "min_completion_tokens": 1_000,
        "equivalence_mode": "warm-repeat",
    },
    "long_262k_capacity": {
        "target_prompt_tokens": 262_000,
        "max_tokens": 16,
        "min_cached_tokens": 261_984,
        "min_completion_tokens": 16,
        "equivalence_mode": "exact",
    },
}
STARTUP_INVARIANT_FIELDS = (
    "mode",
    "config_mode",
    "expected_attention_layers",
    "observed_attention_layers",
    "observed_layer_count",
    "full_attention_ordinals",
    "num_key_value_heads",
    "rank_kv_heads",
    "head_dim",
    "tensor_parallel_size",
    "dtype",
    "dtype_bytes",
    "expected_kv_bytes_per_block",
    "max_model_len_required",
    "block_size",
    "required_gpu_blocks",
    "observed_max_seq_len",
    "runtime_contract",
    "runtime_contract_sha256",
    "runtime_contract_invariant_sha256",
)
LONG_REQUEST_FIELDS = (
    "prompt_tokens",
    "cached_tokens",
    "completion_tokens",
    "finish_reason",
    "message_sha256",
    "elapsed_s",
)


def _valid_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _source(
    value: Any,
    *,
    name: str,
    schema: str,
    reasons: list[str],
    version: int = VERSION,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        reasons.append(f"{name} must be an object")
        return {}
    if value.get("schema") != schema or value.get("version") != version:
        reasons.append(f"{name} schema/version is invalid")
    if value.get("qualified") is not True:
        reasons.append(f"{name} is not qualified")
    return value


def _safe_smoke(value: Any, reasons: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        reasons.append("smoke must be an object")
        return {}
    if value.get("ok") is not True or value.get("mode") != "quick":
        reasons.append("smoke quick suite is not qualified")
    tests = value.get("tests")
    if not isinstance(tests, list):
        reasons.append("smoke tests are missing")
        tests = []
    names = tuple(
        test.get("name") for test in tests if isinstance(test, dict)
    )
    if names != EXPECTED_SMOKE_TESTS:
        reasons.append(
            "smoke test order differs from the fixed quick suite")
    if any(
        not isinstance(test, dict)
        or test.get("ok") is not True
        or test.get("error") not in (None, "")
        or not _finite_positive(test.get("elapsed_s"))
        for test in tests
    ):
        reasons.append("one or more smoke tests failed")
    return {
        "mode": value.get("mode"),
        "ok": value.get("ok"),
        "tests": [
            {
                "name": test.get("name"),
                "ok": test.get("ok"),
                "elapsed_s": test.get("elapsed_s"),
            }
            for test in tests
            if isinstance(test, dict)
        ],
    }


def _validate_preflight(
    value: dict[str, Any],
    reasons: list[str],
) -> list[str]:
    if value.get("max_free_memory_drop_bytes") != MAX_FREE_MEMORY_DROP_BYTES:
        reasons.append(
            "preflight free-memory drop gate differs from 1073741824 bytes")
    stages = value.get("stages")
    if not isinstance(stages, list):
        reasons.append("preflight stages are missing")
        return []
    labels = [
        stage.get("label") for stage in stages if isinstance(stage, dict)
    ]
    if labels != ["before_long", "after_long"]:
        reasons.append(
            "preflight stages must be before_long followed by after_long")
    for stage in stages:
        if not isinstance(stage, dict):
            reasons.append("preflight stage must be an object")
            continue
        label = stage.get("label", "unknown")
        if stage.get("qualified") is not True:
            reasons.append(f"preflight stage is not qualified: {label}")
        if stage.get("matmul_size") != 1024:
            reasons.append(f"preflight matmul size is invalid: {label}")
        if stage.get("timeout_s") != 25:
            reasons.append(f"preflight timeout is invalid: {label}")
        results = stage.get("results")
        if not isinstance(results, list) or len(results) != 4:
            reasons.append(
                f"preflight stage must contain four GPU results: {label}")
            continue
        gpus = [
            result.get("gpu") for result in results
            if isinstance(result, dict)
        ]
        if gpus != [0, 1, 2, 3]:
            reasons.append(f"preflight GPU order is invalid: {label}")
        for result in results:
            if not isinstance(result, dict):
                reasons.append(f"preflight GPU result is invalid: {label}")
                continue
            gpu = result.get("gpu")
            if result.get("ok") is not True:
                reasons.append(f"preflight GPU failed: {label}/{gpu}")
            if (not isinstance(result.get("device_name"), str)
                    or not result.get("device_name")):
                reasons.append(
                    f"preflight GPU device name is invalid: {label}/{gpu}")
            if result.get("device_capability") != [7, 0]:
                reasons.append(
                    f"preflight GPU capability is invalid: {label}/{gpu}")
            free = result.get("free")
            total = result.get("total")
            if (not _is_integer(free) or not _is_integer(total)
                    or free <= 0 or total <= 0 or free > total):
                reasons.append(
                    f"preflight GPU memory is invalid: {label}/{gpu}")
            checksum = result.get("checksum")
            if (not _finite_positive(checksum)
                    or not math.isclose(
                        checksum, float(1024 ** 3),
                        rel_tol=0.0, abs_tol=0.5)):
                reasons.append(
                    f"preflight GPU checksum is invalid: {label}/{gpu}")
    return labels


def _validate_multimodal(
    value: dict[str, Any],
    reasons: list[str],
) -> dict[str, Any]:
    checks = value.get("checks")
    if not isinstance(checks, dict) or set(checks) != EXPECTED_MULTIMODAL_CHECKS:
        reasons.append("multimodal check set differs from the fixed gate")
        checks = {} if not isinstance(checks, dict) else checks
    if any(checks.get(name) is not True
           for name in EXPECTED_MULTIMODAL_CHECKS):
        reasons.append("one or more multimodal checks failed")
    if value.get("reasons") != []:
        reasons.append("multimodal qualification contains failure reasons")
    if not _valid_digest(value.get("source_sha256")):
        reasons.append("multimodal source digest is invalid")
    return {name: checks.get(name) for name in sorted(EXPECTED_MULTIMODAL_CHECKS)}


def _validate_long_report(
    value: dict[str, Any],
    *,
    name: str,
    contract: dict[str, Any],
    reasons: list[str],
) -> dict[str, Any]:
    if value.get("reasons") != []:
        reasons.append(f"{name} contains failure reasons")
    requests = value.get("requests")
    if not isinstance(requests, dict):
        reasons.append(f"{name} requests are missing")
        return {}
    expected_names = {"first", "second"}
    if contract["equivalence_mode"] == "warm-repeat":
        expected_names.add("third")
    if set(requests) != expected_names:
        reasons.append(f"{name} request set differs from the fixed gate")

    safe: dict[str, Any] = {}
    for request_name in sorted(expected_names):
        request = requests.get(request_name)
        if not isinstance(request, dict):
            reasons.append(f"{name}/{request_name} must be an object")
            continue
        record = {field: request.get(field) for field in LONG_REQUEST_FIELDS}
        safe[request_name] = record
        if record["prompt_tokens"] != contract["target_prompt_tokens"]:
            reasons.append(f"{name}/{request_name} prompt token count is invalid")
        cached = record["cached_tokens"]
        minimum_cached = (
            0 if request_name == "first" else contract["min_cached_tokens"])
        if (not _is_integer(cached) or cached < minimum_cached
                or (request_name == "first" and cached != 0)):
            reasons.append(f"{name}/{request_name} cached tokens are invalid")
        completion = record["completion_tokens"]
        if (not _is_integer(completion)
                or completion < contract["min_completion_tokens"]):
            reasons.append(
                f"{name}/{request_name} completion tokens are invalid")
        if not isinstance(record["finish_reason"], str):
            reasons.append(f"{name}/{request_name} finish reason is invalid")
        if not _valid_digest(record["message_sha256"]):
            reasons.append(f"{name}/{request_name} message digest is invalid")
        if not _finite_positive(record["elapsed_s"]):
            reasons.append(f"{name}/{request_name} elapsed time is invalid")

    equivalent_names = (
        ("first", "second")
        if contract["equivalence_mode"] == "exact"
        else ("second", "third")
    )
    left = safe.get(equivalent_names[0])
    right = safe.get(equivalent_names[1])
    if left is not None and right is not None:
        for field in ("completion_tokens", "finish_reason", "message_sha256"):
            if left.get(field) != right.get(field):
                reasons.append(f"{name} equivalent requests differ in {field}")
    return safe


def qualify(
    *,
    ab: Any,
    startup: Any,
    preflight: Any,
    smoke: Any,
    multimodal: Any,
    long_reports: dict[str, Any],
    source_sha256: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    ab_report = _source(
        ab, name="ab", schema=AB_SCHEMA, reasons=reasons)
    startup_report = _source(
        startup, name="startup", schema=STARTUP_SCHEMA, reasons=reasons)
    preflight_report = _source(
        preflight, name="preflight", schema=PREFLIGHT_SCHEMA, reasons=reasons)
    multimodal_report = _source(
        multimodal,
        name="multimodal",
        schema=MULTIMODAL_SCHEMA,
        reasons=reasons,
        version=2,
    )
    safe_smoke = _safe_smoke(smoke, reasons)

    candidate = (
        ab_report.get("startup", {}).get("candidate", {})
        if isinstance(ab_report.get("startup"), dict)
        else {}
    )
    if not isinstance(candidate, dict):
        candidate = {}
        reasons.append("A/B candidate startup is missing")
    for field in STARTUP_INVARIANT_FIELDS:
        if startup_report.get(field) != candidate.get(field):
            reasons.append(
                f"long-context startup differs from A/B candidate in {field}")
    if startup_report.get("mode") != "full_attention":
        reasons.append("long-context startup must use full_attention")
    if startup_report.get("observed_attention_layers") != 10:
        reasons.append("long-context startup must allocate ten KV layers")

    stage_labels = _validate_preflight(preflight_report, reasons)
    multimodal_checks = _validate_multimodal(multimodal_report, reasons)

    safe_long: dict[str, Any] = {}
    for name, contract in EXPECTED_LONG_CONTRACTS.items():
        report = _source(
            long_reports.get(name),
            name=name,
            schema=LONG_SCHEMA,
            reasons=reasons,
        )
        if report.get("contract") != contract:
            reasons.append(f"{name} contract differs from the fixed gate")
        if not _valid_digest(report.get("source_sha256")):
            reasons.append(f"{name} safe report source digest is invalid")
        safe_requests = _validate_long_report(
            report, name=name, contract=contract, reasons=reasons)
        safe_long[name] = {
            "contract": report.get("contract"),
            "qualified": report.get("qualified"),
            "requests": safe_requests,
            "source_sha256": report.get("source_sha256"),
        }

    expected_sources = {
        "ab",
        "startup",
        "preflight",
        "smoke",
        "multimodal",
        *EXPECTED_LONG_CONTRACTS,
    }
    if set(source_sha256) != expected_sources:
        reasons.append("source digest set differs from the fixed evidence set")
    for name in sorted(expected_sources):
        if not _valid_digest(source_sha256.get(name)):
            reasons.append(f"source digest is invalid: {name}")

    capacity = ab_report.get("capacity")
    if not isinstance(capacity, dict):
        capacity = {}
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "qualified": not reasons,
        "reasons": reasons,
        "scope": "hybrid-kv-capacity-correctness-not-prefill-speed",
        "ab_capacity": {
            "legacy_gpu_blocks": capacity.get("legacy_gpu_blocks"),
            "candidate_gpu_blocks": capacity.get("candidate_gpu_blocks"),
            "gpu_block_ratio": capacity.get("gpu_block_ratio"),
            "legacy_cpu_blocks": capacity.get("legacy_cpu_blocks"),
            "candidate_cpu_blocks": capacity.get("candidate_cpu_blocks"),
            "cpu_block_ratio": capacity.get("cpu_block_ratio"),
        },
        "candidate_startup": {
            "mode": startup_report.get("mode"),
            "attention_layers": startup_report.get(
                "observed_attention_layers"),
            "gpu_blocks": startup_report.get("observed_gpu_blocks"),
            "cpu_blocks": startup_report.get("observed_cpu_blocks"),
            "gpu_tokens": startup_report.get("observed_gpu_tokens"),
            "runtime_contract_sha256": startup_report.get(
                "runtime_contract_sha256"),
            "service_log_sha256": startup_report.get("service_log_sha256"),
        },
        "preflight": {
            "qualified": preflight_report.get("qualified"),
            "stage_labels": stage_labels,
        },
        "smoke": safe_smoke,
        "multimodal": {
            "qualified": multimodal_report.get("qualified"),
            "checks": multimodal_checks,
            "source_sha256": multimodal_report.get("source_sha256"),
        },
        "long_context": safe_long,
        "source_sha256": source_sha256,
    }


def _load(path: Path) -> tuple[Any, str]:
    payload = path.read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
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
    parser.add_argument("--ab", type=Path, required=True)
    parser.add_argument("--startup", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--multimodal", type=Path, required=True)
    parser.add_argument("--long-131k", type=Path, required=True)
    parser.add_argument("--long-235k", type=Path, required=True)
    parser.add_argument("--long-262k", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    paths = {
        "ab": args.ab,
        "startup": args.startup,
        "preflight": args.preflight,
        "smoke": args.smoke,
        "multimodal": args.multimodal,
        "long_131k_exact": args.long_131k,
        "long_235k_warm_repeat": args.long_235k,
        "long_262k_capacity": args.long_262k,
    }
    try:
        loaded = {name: _load(path) for name, path in paths.items()}
        report = qualify(
            ab=loaded["ab"][0],
            startup=loaded["startup"][0],
            preflight=loaded["preflight"][0],
            smoke=loaded["smoke"][0],
            multimodal=loaded["multimodal"][0],
            long_reports={
                name: loaded[name][0] for name in EXPECTED_LONG_CONTRACTS
            },
            source_sha256={
                name: digest for name, (_, digest) in loaded.items()
            },
        )
    except Exception as error:
        report = {
            "schema": SCHEMA,
            "version": VERSION,
            "qualified": False,
            "reasons": [f"cannot qualify M1-49 long-context evidence: {error}"],
        }
    _atomic_write(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
