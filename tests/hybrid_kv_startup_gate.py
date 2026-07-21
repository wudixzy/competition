#!/usr/bin/env python3
"""Qualify one M1-49 hybrid-KV startup without retaining raw logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any


SCHEMA = "bi100-hybrid-kv-startup-v1"
VERSION = 1
MODE_ENV = "BI100_HYBRID_KV_ACCOUNTING"
MODE_CONFIG = "bi100_hybrid_kv_accounting_mode"
EXPECTED_LAYERS = {"legacy40": 40, "full_attention": 10}
MAX_SEQ_RE = re.compile(r"\bmax_seq_len=(\d+)\b")
GPU_BLOCK_RE = re.compile(r"# GPU blocks:\s*(\d+)\b")
CPU_BLOCK_RE = re.compile(r"# CPU blocks:\s*(\d+)\b")
DTYPE_RE = re.compile(r"\bdtype=torch\.(float16|bfloat16|float32)\b")
BLOCK_SIZE_RE = re.compile(r"\bblock_size=(\d+)\b")
SWAP_SPACE_RE = re.compile(r"\bswap_space=(\d+(?:\.\d+)?)\b")
SERVICE_CONTRACT_RE = re.compile(
    r"^\[BI100\] M1-49 runtime contract; (?P<fields>[^\r\n]+)$",
    re.MULTILINE,
)
MODEL_ACCOUNTING_RE = re.compile(
    r"\[BI100\] Qwen hybrid KV accounting; "
    r"tp_rank=(\d+) "
    r"env_mode=(legacy40|full_attention|<unset>) "
    r"config_mode=(legacy40|full_attention) "
    r"configured_kv_layers=(\d+) full_attention_layers=(\d+) "
    r"full_attention_ordinals=([0-9,]+)(?=\r?$)",
    re.MULTILINE,
)
DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}
ACCOUNTING_MODE_PLACEHOLDER = "<accounting-mode>"
FIXED_SERVICE_CONTRACT = {
    "host": "0.0.0.0",
    "port": "8000",
    "served_model_name": "llm",
    "max_model_len": "262144",
    "tensor_parallel_size": "4",
    "gpu_memory_utilization": "0.9",
    "max_num_seqs": "1",
    "max_num_batched_tokens": "8192",
    "max_seq_len_to_capture": "32768",
    "chunked_prefill": "1",
    "prefix_caching": "1",
    "trust_remote_code": "1",
    "disable_log_requests": "1",
    "disable_frontend_multiprocessing": "1",
    "auto_tool_choice": "1",
    "tool_call_parser": "qwen3_coder",
    "reasoning_parser": "qwen3",
    "enable_custom_ipc": "1",
    "moe_direct": "1",
    "gdn_packed": "1",
    "cpu_kv_offload": "0",
    "gdn_cache_policy": "admission64",
    "gdn_restore_mode": "direct",
    "cache_trace": "0",
    "fused_prefill": "0",
    "kv_eviction_policy": "lru",
}


def evaluate(
    log_text: str,
    *,
    mode: str,
    config_mode: Any,
    layers_block_type: Any,
    full_attention_ordinals: Any,
    num_key_value_heads: Any,
    head_dim: Any,
    max_model_len: int,
    block_size: int,
    tensor_parallel_size: int,
    runtime_contract: dict[str, Any],
    contract_reasons: list[str] | None = None,
) -> dict[str, Any]:
    reasons = list(contract_reasons or [])
    if mode not in EXPECTED_LAYERS:
        reasons.append(f"unsupported mode: {mode!r}")

    expected_layers = EXPECTED_LAYERS.get(mode)
    if config_mode != mode:
        reasons.append(
            f"serialized config mode must equal {mode!r}, got {config_mode!r}")

    attention_layers = None
    layer_count = None
    if (isinstance(layers_block_type, list)
            and all(isinstance(item, str) for item in layers_block_type)):
        layer_count = len(layers_block_type)
        attention_layers = layers_block_type.count("attention")
    else:
        reasons.append("layers_block_type must be a list of strings")
    if layer_count != 40:
        reasons.append(f"layers_block_type must contain 40 layers, got {layer_count}")
    if attention_layers != expected_layers:
        reasons.append(
            f"attention layer count must equal {expected_layers}, "
            f"got {attention_layers}")

    if (not isinstance(full_attention_ordinals, list)
            or not all(isinstance(item, int) and not isinstance(item, bool)
                       for item in full_attention_ordinals)):
        reasons.append("full-attention ordinals must be a list of integers")
        full_attention_ordinals = None
    elif len(full_attention_ordinals) != 10:
        reasons.append(
            "model must expose exactly 10 full-attention ordinals, got "
            f"{len(full_attention_ordinals)}")

    if (not isinstance(num_key_value_heads, int)
            or isinstance(num_key_value_heads, bool)
            or num_key_value_heads <= 0):
        reasons.append("num_key_value_heads must be a positive integer")
    if (not isinstance(head_dim, int) or isinstance(head_dim, bool)
            or head_dim <= 0):
        reasons.append("head_dim must be a positive integer")

    max_seq_values = [int(value) for value in MAX_SEQ_RE.findall(log_text)]
    gpu_block_values = [int(value) for value in GPU_BLOCK_RE.findall(log_text)]
    cpu_block_values = [int(value) for value in CPU_BLOCK_RE.findall(log_text)]
    dtype_values = DTYPE_RE.findall(log_text)
    accounting_reports = [
        {
            "tp_rank": int(tp_rank),
            "env_mode": env_mode,
            "config_mode": report_mode,
            "configured_kv_layers": int(configured),
            "full_attention_layers": int(full_attention),
            "full_attention_ordinals": [
                int(value) for value in ordinals.split(",")],
        }
        for tp_rank, env_mode, report_mode, configured, full_attention, ordinals
        in MODEL_ACCOUNTING_RE.findall(log_text)
    ]
    max_seq_len = max_seq_values[-1] if max_seq_values else None
    gpu_blocks = gpu_block_values[-1] if gpu_block_values else None
    cpu_blocks = cpu_block_values[-1] if cpu_block_values else None
    required_gpu_blocks = math.ceil(max_model_len / block_size)
    dtype = dtype_values[-1] if dtype_values else None
    dtype_bytes = DTYPE_BYTES.get(dtype) if dtype is not None else None
    rank_kv_heads = (
        max(1, num_key_value_heads // tensor_parallel_size)
        if isinstance(num_key_value_heads, int)
        and not isinstance(num_key_value_heads, bool)
        and num_key_value_heads > 0 else None)
    expected_bytes_per_block = (
        expected_layers * 2 * block_size * rank_kv_heads * head_dim * dtype_bytes
        if all(isinstance(value, int) for value in (
            expected_layers, rank_kv_heads, head_dim, dtype_bytes)) else None)

    if max_seq_len is None:
        reasons.append("startup log is missing max_seq_len")
    elif max_seq_len < max_model_len:
        reasons.append(
            f"max_seq_len {max_seq_len} is below {max_model_len}")
    if gpu_blocks is None:
        reasons.append("startup log is missing GPU block count")
    elif gpu_blocks < required_gpu_blocks:
        reasons.append(
            f"GPU blocks {gpu_blocks} are below required {required_gpu_blocks}")
    if cpu_blocks is None:
        reasons.append("startup log is missing CPU block count")
    elif cpu_blocks <= 0:
        reasons.append("CPU block count must be positive")
    if dtype is None:
        reasons.append("startup log is missing model dtype")
    if len(accounting_reports) != tensor_parallel_size:
        reasons.append(
            "startup log must contain exactly one hybrid-KV accounting "
            f"report per TP rank; got {len(accounting_reports)}")
    observed_ranks = sorted(
        report["tp_rank"] for report in accounting_reports)
    expected_ranks = list(range(tensor_parallel_size))
    if observed_ranks != expected_ranks:
        reasons.append(
            f"runtime accounting TP ranks must equal {expected_ranks}, "
            f"got {observed_ranks}")
    for index, report in enumerate(accounting_reports):
        expected_report = {
            "tp_rank": report["tp_rank"],
            "env_mode": mode,
            "config_mode": mode,
            "configured_kv_layers": expected_layers,
            "full_attention_layers": 10,
            "full_attention_ordinals": full_attention_ordinals,
        }
        if report != expected_report:
            reasons.append(
                f"runtime accounting report {index} differs from "
                f"{expected_report}: {report}")

    contract_json = json.dumps(
        runtime_contract, ensure_ascii=True, sort_keys=True,
        separators=(",", ":")).encode("ascii")
    runtime_contract_sha256 = hashlib.sha256(contract_json).hexdigest()
    invariant_contract = json.loads(contract_json.decode("ascii"))
    invariant_service = invariant_contract.get("service")
    if isinstance(invariant_service, dict):
        invariant_service["accounting"] = ACCOUNTING_MODE_PLACEHOLDER
    invariant_json = json.dumps(
        invariant_contract, ensure_ascii=True, sort_keys=True,
        separators=(",", ":")).encode("ascii")
    runtime_contract_invariant_sha256 = hashlib.sha256(
        invariant_json).hexdigest()

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "mode": mode,
        "config_mode": config_mode,
        "expected_attention_layers": expected_layers,
        "observed_attention_layers": attention_layers,
        "observed_layer_count": layer_count,
        "full_attention_ordinals": full_attention_ordinals,
        "num_key_value_heads": num_key_value_heads,
        "rank_kv_heads": rank_kv_heads,
        "head_dim": head_dim,
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": dtype,
        "dtype_bytes": dtype_bytes,
        "expected_kv_bytes_per_block": expected_bytes_per_block,
        "runtime_accounting_reports": accounting_reports,
        "runtime_contract": runtime_contract,
        "runtime_contract_sha256": runtime_contract_sha256,
        "runtime_contract_invariant_sha256": (
            runtime_contract_invariant_sha256),
        "max_model_len_required": max_model_len,
        "block_size": block_size,
        "required_gpu_blocks": required_gpu_blocks,
        "observed_max_seq_len": max_seq_len,
        "observed_gpu_blocks": gpu_blocks,
        "observed_cpu_blocks": cpu_blocks,
        "observed_gpu_tokens": (
            gpu_blocks * block_size if gpu_blocks is not None else None),
        "qualified": not reasons,
        "reasons": reasons,
    }


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


def _load_config_contract(
    model_path: Path,
) -> tuple[Any, Any, Any, Any, Any]:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True)
    text_config = getattr(config, "text_config", config)
    layer_types = getattr(text_config, "layer_types", None)
    full_attention_ordinals = (
        [index for index, layer_type in enumerate(layer_types)
         if layer_type == "full_attention"]
        if isinstance(layer_types, list) else None)
    return (
        getattr(config, MODE_CONFIG, None),
        getattr(config, "layers_block_type", None),
        full_attention_ordinals,
        getattr(text_config, "num_key_value_heads", None),
        getattr(text_config, "head_dim", None),
    )


def _runtime_contract(
    log_text: str,
    model_path: Path,
    *,
    mode: str,
    max_model_len: int,
    block_size: int,
    tensor_parallel_size: int,
) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    matches = SERVICE_CONTRACT_RE.findall(log_text)
    service: dict[str, str] = {}
    if len(matches) != 1:
        reasons.append(
            "service log must contain exactly one M1-49 runtime contract; "
            f"got {len(matches)}")
    if matches:
        for token in matches[-1].split():
            if token.count("=") != 1:
                reasons.append(f"invalid runtime contract token: {token!r}")
                continue
            name, value = token.split("=", 1)
            if not name or not value:
                reasons.append(f"invalid runtime contract token: {token!r}")
            elif name in service:
                reasons.append(f"duplicate runtime contract field: {name}")
            else:
                service[name] = value

    expected_service = dict(FIXED_SERVICE_CONTRACT)
    expected_service.update({
        "accounting": mode,
        "model_path": str(model_path),
        "runtime_site_packages": os.environ.get(
            "BI100_RUNTIME_SITE_PACKAGES", "system"),
        "max_model_len": str(max_model_len),
        "tensor_parallel_size": str(tensor_parallel_size),
    })
    missing = sorted(set(expected_service) - set(service))
    unexpected = sorted(set(service) - set(expected_service))
    if missing:
        reasons.append(f"runtime contract fields missing: {missing}")
    if unexpected:
        reasons.append(f"runtime contract fields unexpected: {unexpected}")
    for name, expected in expected_service.items():
        if service.get(name) != expected:
            reasons.append(
                f"runtime contract {name} must equal {expected!r}, "
                f"got {service.get(name)!r}")

    max_seq_values = [int(value) for value in MAX_SEQ_RE.findall(log_text)]
    block_size_values = [int(value) for value in BLOCK_SIZE_RE.findall(log_text)]
    swap_space_values = [float(value) for value in SWAP_SPACE_RE.findall(log_text)]
    dtype_values = DTYPE_RE.findall(log_text)
    engine = {
        "max_seq_len": max_seq_values[-1] if max_seq_values else None,
        "block_size": block_size_values[-1] if block_size_values else None,
        "swap_space_gib": swap_space_values[-1] if swap_space_values else None,
        "dtype": dtype_values[-1] if dtype_values else None,
    }
    expected_engine = {
        "max_seq_len": max_model_len,
        "block_size": block_size,
        "swap_space_gib": 4.0,
        "dtype": "float16",
    }
    for name, expected in expected_engine.items():
        if engine.get(name) != expected:
            reasons.append(
                f"parsed engine contract {name} must equal {expected!r}, "
                f"got {engine.get(name)!r}")

    config_path = model_path / "config.json"
    try:
        config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    except OSError as error:
        reasons.append(f"cannot hash model config: {error}")
        config_sha256 = None
    return {
        "model_config_sha256": config_sha256,
        "service": service,
        "engine": engine,
    }, reasons


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--mode", choices=tuple(EXPECTED_LAYERS), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=262_144)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.max_model_len <= 0:
        parser.error("--max-model-len must be positive")
    if args.block_size <= 0:
        parser.error("--block-size must be positive")
    if args.tensor_parallel_size <= 0:
        parser.error("--tensor-parallel-size must be positive")
    if os.environ.get(MODE_ENV) != args.mode:
        parser.error(f"{MODE_ENV} must equal --mode")

    log_bytes = args.log.read_bytes()
    log_text = log_bytes.decode("utf-8", errors="replace")
    try:
        (config_mode, layers_block_type, full_attention_ordinals,
         num_key_value_heads,
         head_dim) = _load_config_contract(args.model_path)
        runtime_contract, contract_reasons = _runtime_contract(
            log_text,
            args.model_path,
            mode=args.mode,
            max_model_len=args.max_model_len,
            block_size=args.block_size,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        report = evaluate(
            log_text,
            mode=args.mode,
            config_mode=config_mode,
            layers_block_type=layers_block_type,
            full_attention_ordinals=full_attention_ordinals,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            max_model_len=args.max_model_len,
            block_size=args.block_size,
            tensor_parallel_size=args.tensor_parallel_size,
            runtime_contract=runtime_contract,
            contract_reasons=contract_reasons,
        )
    except Exception as error:
        report = {
            "schema": SCHEMA,
            "version": VERSION,
            "mode": args.mode,
            "qualified": False,
            "reasons": [f"config contract probe failed: {error}"],
        }
    report["service_log_sha256"] = hashlib.sha256(log_bytes).hexdigest()
    _atomic_write(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
