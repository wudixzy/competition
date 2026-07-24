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
    _require(isinstance(command, list) and bool(command)
             and all(isinstance(item, str) and item for item in command),
             "runtime contract command is invalid")
    environment = value["environment"]
    _require(isinstance(environment, dict)
             and all(isinstance(key, str) and key
                     and isinstance(item, str)
                     for key, item in environment.items()),
             "runtime contract environment is invalid")
    if require_cache_trace:
        _require(environment.get("BI100_CACHE_TRACE") == "1",
                 "runtime contract must set BI100_CACHE_TRACE=1")
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
    return sha256_json(value)
