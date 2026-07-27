#!/usr/bin/env python3
"""Validate privacy-safe, reproducible quality-run runtime contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


Json = dict[str, Any]
BASE_IMAGE = (
    "harbor.4pd.io/modelhubxc/enginex-iluvatar/"
    "bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3"
)
FIELDS = {
    "schema", "version", "source_revision", "runtime_identity",
    "runtime_overlay_sha256", "instance", "gpu_count",
    "tensor_parallel_size", "max_model_len", "model_path", "tokenizer_path",
    "served_model_name", "base_image", "command", "environment",
    "cache_trace_enabled", "optimization_label",
}
FIXED_SERVICE_ENVIRONMENT = {
    "BI100_ALLOW_PREFIX_GUARD_CAP": "0",
    "BI100_ATTN_COREX_HEAD_RMS_NORM": "1",
    "BI100_ATTN_COREX_PAGED_GATHER": "1",
    "BI100_CACHE_TRACE": "1",
    "BI100_CPU_KV_OFFLOAD": "0",
    "BI100_DNN_CHUNK": "4096",
    "BI100_EXECUTOR_STARTUP_DEBUG": "1",
    "BI100_FORCE_PAGED_ATTN_V2": "0",
    "BI100_GDN_ALLOW_NAN_ZERO": "0",
    "BI100_GDN_COREX_BETA_DECAY": "1",
    "BI100_GDN_COREX_CAUSAL_CONV": "1",
    "BI100_GDN_COREX_GATED_NORM": "1",
    "BI100_GDN_COREX_QK_MAP": "1",
    "BI100_GDN_FINITE_CHECK": "0",
    "BI100_HYBRID_KV_ACCOUNTING": "full_attention",
    "BI100_MOE_COREX_EXACT_REDUCE": "1",
    "BI100_MOE_COREX_WEIGHT_GATHER": "1",
    "BI100_MOE_FUSED_ACTIVATION": "1",
    "BI100_PAGED_ATTN_DIAGNOSTICS": "0",
    "BI100_PREFIX_BLOCKS_PER_TILE": "32",
    "BI100_PREFIX_DTYPE": "float16",
    "BI100_PREFIX_MODEL_FINGERPRINT": "Qwen3.6-35B-A3B",
    "BI100_PREFIX_TP_SIZE": "4",
    "BI100_PROFILE": "0",
    "BI100_PROFILE_FILTER": "",
    "BI100_PROFILE_INCLUDE_STARTUP": "0",
    "BI100_PROFILE_MODE": "sync",
    "BI100_PYTORCH_DECODE_THRESHOLD": "32768",
    "BI100_UNSET_CUDA_VISIBLE_DEVICES": "1",
    "ENABLE_CUSTOM_IPC": "1",
    "PYTHONFAULTHANDLER": "1",
    "PYTHONUNBUFFERED": "1",
    "VLLM_ENGINE_ITERATION_TIMEOUT_S": "3600",
}
KERNEL_PROFILES = {
    "submission": {
        "BI100_GDN_COMBINED_QK_NORM": "0",
        "BI100_GDN_COREX_PACKED_DECODE": "1",
        "BI100_MOE_COREX_DIRECT_ROUTED": "1",
    },
    "strict-reference": {
        "BI100_GDN_COMBINED_QK_NORM": "0",
        "BI100_GDN_COREX_PACKED_DECODE": "0",
        "BI100_MOE_COREX_DIRECT_ROUTED": "0",
    },
    "strict-reference-combined-qk": {
        "BI100_GDN_COMBINED_QK_NORM": "1",
        "BI100_GDN_COREX_PACKED_DECODE": "0",
        "BI100_MOE_COREX_DIRECT_ROUTED": "0",
    },
}
VARIABLE_SERVICE_ENVIRONMENT = {
    "BI100_ATTN_COREX_FUSED_PREFILL": {"0", "1"},
    "BI100_GDN_CACHE_POLICY": {"fine32", "admission64"},
    "BI100_GDN_RESTORE_MODE": {"direct", "aligned"},
    "BI100_KV_EVICTION_POLICY": {"lru", "frequency"},
}
RUNTIME_PATH_ENVIRONMENT = {"BI100_RUNTIME_SITE_PACKAGES"}
KERNEL_PROFILE_ENVIRONMENT = set(next(iter(KERNEL_PROFILES.values())))
SERVICE_ENVIRONMENT_FIELDS = (
    set(FIXED_SERVICE_ENVIRONMENT)
    | set(VARIABLE_SERVICE_ENVIRONMENT)
    | KERNEL_PROFILE_ENVIRONMENT
    | RUNTIME_PATH_ENVIRONMENT
)


class RuntimeContractError(RuntimeError):
    pass


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeContractError(reason)


def is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def is_git_revision(value: Any) -> bool:
    return (isinstance(value, str) and 7 <= len(value) <= 64
            and all(character in "0123456789abcdef" for character in value))


def sha256_json(value: Any) -> str:
    canonical = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def service_command(model_path: str) -> list[str]:
    return [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--model", model_path,
        "--served-model-name", "llm",
        "--max-model-len", "262144",
        "--gpu-memory-utilization", "0.9",
        "--trust-remote-code",
        "--tensor-parallel-size", "4",
        "--max-num-seqs", "1",
        "--disable-log-requests",
        "--disable-frontend-multiprocessing",
        "--max-num-batched-tokens", "8192",
        "--enable-chunked-prefill",
        "--max-seq-len-to-capture", "32768",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
        "--reasoning-parser", "qwen3",
        "--enable-prefix-caching",
    ]


def service_environment(
    runtime_site_packages: str,
    *,
    gdn_cache_policy: str,
    gdn_restore_mode: str,
    fused_prefill: str,
    kv_eviction_policy: str,
    kernel_profile: str = "submission",
) -> dict[str, str]:
    if kernel_profile not in KERNEL_PROFILES:
        raise RuntimeContractError(
            f"unknown quality kernel profile: {kernel_profile}")
    value = dict(FIXED_SERVICE_ENVIRONMENT)
    value.update(KERNEL_PROFILES[kernel_profile])
    value.update({
        "BI100_ATTN_COREX_FUSED_PREFILL": fused_prefill,
        "BI100_GDN_CACHE_POLICY": gdn_cache_policy,
        "BI100_GDN_RESTORE_MODE": gdn_restore_mode,
        "BI100_KV_EVICTION_POLICY": kv_eviction_policy,
        "BI100_RUNTIME_SITE_PACKAGES": runtime_site_packages,
    })
    return value


def load_runtime_contract(
    path: Path,
    expected: Json,
    *,
    require_cache_trace: bool,
) -> tuple[Json, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    digest = validate_runtime_contract(
        value, expected, require_cache_trace=require_cache_trace)
    return value, digest


def validate_runtime_contract(
    value: Any,
    expected: Json,
    *,
    require_cache_trace: bool,
) -> str:
    _require(isinstance(value, dict) and set(value) == FIELDS,
             "runtime contract fields are invalid")
    _require(value["schema"] == "bi100-quality-runtime-contract-v1"
             and value["version"] == 1,
             "runtime contract schema or version is invalid")
    for field, expected_value in expected.items():
        _require(value.get(field) == expected_value,
                 f"runtime contract {field} differs from the run")
    _require(is_git_revision(value["source_revision"]),
             "runtime contract source revision is invalid")
    _require(set(value["source_revision"]) != {"0"},
             "runtime contract source revision is still a placeholder")
    _require(is_sha256(value["runtime_overlay_sha256"]),
             "runtime contract overlay identity is invalid")
    _require(set(value["runtime_overlay_sha256"]) != {"0"},
             "runtime contract overlay identity is still a placeholder")
    _require(value["base_image"] == BASE_IMAGE,
             "runtime contract base image is obsolete or invalid")
    if require_cache_trace:
        _require(value["cache_trace_enabled"] is True,
                 "quality runtime contract must enable cache trace")
    _require(isinstance(value["optimization_label"], str)
             and bool(value["optimization_label"]),
             "runtime contract optimization label is invalid")
    command = value["command"]
    _require(command == service_command(value["model_path"]),
             "runtime contract command differs from the fixed quality command")
    environment = value["environment"]
    _require(isinstance(environment, dict)
             and all(isinstance(key, str) and key
                     and isinstance(item, str)
                     for key, item in environment.items()),
             "runtime contract environment is invalid")
    blocked_names = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")
    _require(not any(fragment in key.upper()
                     for key in environment for fragment in blocked_names),
             "runtime contract contains a secret-bearing environment name")
    serialized = json.dumps(value, ensure_ascii=True).lower()
    blocked_values = (
        "begin openssh private key", "github_pat_", "ghp_",
        "modelhub_access_token", "proxy-authorization",
    )
    _require(not any(marker in serialized for marker in blocked_values),
             "runtime contract contains a credential marker")
    _require(set(environment) == SERVICE_ENVIRONMENT_FIELDS,
             "runtime contract environment fields are invalid")
    for name, expected_value in FIXED_SERVICE_ENVIRONMENT.items():
        _require(environment.get(name) == expected_value,
                 f"runtime contract environment {name} is not fixed")
    for name, choices in VARIABLE_SERVICE_ENVIRONMENT.items():
        _require(environment.get(name) in choices,
                 f"runtime contract environment {name} is invalid")
    kernel_values = {
        name: environment.get(name)
        for name in next(iter(KERNEL_PROFILES.values()))
    }
    _require(
        any(kernel_values == profile for profile in KERNEL_PROFILES.values()),
        "runtime contract kernel profile is invalid",
    )
    runtime_site_packages = environment.get("BI100_RUNTIME_SITE_PACKAGES")
    _require(isinstance(runtime_site_packages, str)
             and Path(runtime_site_packages).is_absolute(),
             "runtime contract overlay path is invalid")
    if require_cache_trace:
        _require(environment.get("BI100_CACHE_TRACE") == "1",
                 "runtime contract must set BI100_CACHE_TRACE=1")
    return sha256_json(value)
