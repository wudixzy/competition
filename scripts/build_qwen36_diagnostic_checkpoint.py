#!/usr/bin/env python3
"""Build an exact-shape, reduced-depth Qwen3.6 diagnostic checkpoint.

The builder copies tensor payload bytes directly from the source safetensors
files. It keeps all non-language-layer tensors and the first N complete
3-GDN + 1-full-attention cycles. It does not deserialize, cast, quantize, or
otherwise transform any tensor.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
from typing import Any, BinaryIO, Iterable


SCHEMA = "qwen36-diagnostic-checkpoint-v1"
VERSION = 1
INDEX_NAME = "model.safetensors.index.json"
MANIFEST_NAME = "diagnostic-checkpoint-manifest.json"
LAYER_PATTERN = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>[0-9]+)\.")
SHARD_PATTERN = re.compile(r".*\.safetensors$")
COPY_CHUNK_BYTES = 8 * 1024 * 1024

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

TARGET_TEXT_CONTRACT = {
    "model_type": "qwen3_5_moe_text",
    "dtype": "bfloat16",
    "attn_output_gate": True,
    "hidden_size": 2048,
    "head_dim": 256,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "linear_conv_kernel_dim": 4,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "mamba_ssm_dtype": "float32",
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512,
    "mtp_num_hidden_layers": 1,
    "max_position_embeddings": 262144,
    "full_attention_interval": 4,
}

LINEAR_REQUIRED_SUFFIXES = {
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.out_proj.weight",
}
FULL_REQUIRED_SUFFIXES = {
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
}
COMMON_REQUIRED_SUFFIXES = {
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
    "mlp.gate.weight",
    "mlp.shared_expert.gate_proj.weight",
    "mlp.shared_expert.up_proj.weight",
    "mlp.shared_expert.down_proj.weight",
    "mlp.shared_expert_gate.weight",
}
GLOBAL_REQUIRED_WEIGHTS = {
    "model.language_model.embed_tokens.weight",
    "model.language_model.norm.weight",
    "lm_head.weight",
}


class CheckpointError(RuntimeError):
    """Raised when a source or generated checkpoint violates the contract."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise CheckpointError(f"expected JSON object in {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_nbytes(metadata: dict[str, Any], *, name: str) -> int:
    dtype = metadata.get("dtype")
    shape = metadata.get("shape")
    offsets = metadata.get("data_offsets")
    if dtype not in DTYPE_BYTES:
        raise CheckpointError(f"{name}: unsupported safetensors dtype {dtype!r}")
    if (not isinstance(shape, list)
            or any(not isinstance(size, int) or size < 0 for size in shape)):
        raise CheckpointError(f"{name}: invalid shape {shape!r}")
    if (not isinstance(offsets, list) or len(offsets) != 2
            or any(not isinstance(offset, int) for offset in offsets)):
        raise CheckpointError(f"{name}: invalid data_offsets {offsets!r}")
    start, end = offsets
    if start < 0 or end < start:
        raise CheckpointError(f"{name}: invalid data range {offsets!r}")
    expected = math.prod(shape) * DTYPE_BYTES[dtype]
    if end - start != expected:
        raise CheckpointError(
            f"{name}: payload is {end - start} bytes, expected {expected}")
    return expected


def read_safetensors_header(
    path: Path,
) -> tuple[int, dict[str, dict[str, Any]], dict[str, str] | None]:
    try:
        file_size = path.stat().st_size
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise CheckpointError(f"{path}: truncated safetensors prefix")
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size < 2 or header_size > file_size - 8:
                raise CheckpointError(
                    f"{path}: invalid safetensors header size {header_size}")
            raw_header = stream.read(header_size)
    except OSError as error:
        raise CheckpointError(f"cannot read safetensors header {path}: {error}") from error

    try:
        header = json.loads(raw_header)
    except json.JSONDecodeError as error:
        raise CheckpointError(
            f"{path}: invalid safetensors header JSON: {error}") from error
    if not isinstance(header, dict):
        raise CheckpointError(f"{path}: safetensors header must be an object")

    raw_metadata = header.pop("__metadata__", None)
    if raw_metadata is not None:
        if (not isinstance(raw_metadata, dict)
                or any(not isinstance(key, str)
                       or not isinstance(value, str)
                       for key, value in raw_metadata.items())):
            raise CheckpointError(f"{path}: invalid __metadata__")
        metadata: dict[str, str] | None = raw_metadata
    else:
        metadata = None

    tensors: dict[str, dict[str, Any]] = {}
    ranges: list[tuple[int, int, str]] = []
    for name, tensor_metadata in header.items():
        if not isinstance(name, str) or not isinstance(tensor_metadata, dict):
            raise CheckpointError(f"{path}: invalid tensor entry {name!r}")
        _tensor_nbytes(tensor_metadata, name=f"{path.name}:{name}")
        start, end = tensor_metadata["data_offsets"]
        tensors[name] = tensor_metadata
        ranges.append((start, end, name))

    expected_start = 0
    for start, end, name in sorted(ranges):
        if start != expected_start:
            raise CheckpointError(
                f"{path}:{name}: non-contiguous or overlapping payload; "
                f"expected offset {expected_start}, got {start}")
        expected_start = end
    if 8 + header_size + expected_start != file_size:
        raise CheckpointError(
            f"{path}: payload ends at {8 + header_size + expected_start}, "
            f"file size is {file_size}")
    return header_size, tensors, metadata


def _layer_index(weight_name: str) -> int | None:
    match = LAYER_PATTERN.match(weight_name)
    return int(match.group("layer")) if match else None


def _selected_weight(weight_name: str, layer_count: int) -> bool:
    layer = _layer_index(weight_name)
    return layer is None or layer < layer_count


def _require_target_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("model_type") != "qwen3_5_moe":
        raise CheckpointError(
            "source model_type must be 'qwen3_5_moe', got "
            f"{config.get('model_type')!r}")
    architectures = config.get("architectures")
    if (not isinstance(architectures, list)
            or "Qwen3_5MoeForCausalLM" not in architectures):
        raise CheckpointError(
            "source architectures must contain Qwen3_5MoeForCausalLM")
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise CheckpointError("source config lacks text_config")
    differences = {
        key: {"expected": expected, "actual": text_config.get(key)}
        for key, expected in TARGET_TEXT_CONTRACT.items()
        if text_config.get(key) != expected
    }
    if differences:
        raise CheckpointError(
            "source does not match the Qwen3.6-35B-A3B tensor contract: "
            + json.dumps(differences, sort_keys=True))
    if config.get("vision_config", {}).get("out_hidden_size") != 2048:
        raise CheckpointError(
            "source vision_config.out_hidden_size must remain 2048")
    return text_config


def _require_cycle_layout(
    text_config: dict[str, Any],
    cycle_count: int,
) -> tuple[int, list[str]]:
    if cycle_count < 1:
        raise CheckpointError("cycle_count must be positive")
    source_layers = text_config.get("num_hidden_layers")
    layer_types = text_config.get("layer_types")
    if not isinstance(source_layers, int) or source_layers < 4:
        raise CheckpointError("source num_hidden_layers is invalid")
    if not isinstance(layer_types, list) or len(layer_types) != source_layers:
        raise CheckpointError(
            "source layer_types must match num_hidden_layers")
    expected_cycle = [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    for offset in range(0, source_layers, 4):
        if layer_types[offset:offset + 4] != expected_cycle:
            raise CheckpointError(
                f"source layer cycle at {offset} is not 3 GDN + 1 full attention")
    layer_count = cycle_count * 4
    if layer_count > source_layers:
        raise CheckpointError(
            f"requested {layer_count} layers from a {source_layers}-layer source")
    return layer_count, list(layer_types[:layer_count])


def _validate_weight_names(
    weight_map: dict[str, str],
    text_config: dict[str, Any],
    selected_layer_count: int,
) -> None:
    if not GLOBAL_REQUIRED_WEIGHTS.issubset(weight_map):
        missing = sorted(GLOBAL_REQUIRED_WEIGHTS - set(weight_map))
        raise CheckpointError(f"source checkpoint lacks global weights: {missing}")

    source_layer_count = text_config["num_hidden_layers"]
    observed_layers = {
        layer for name in weight_map
        if (layer := _layer_index(name)) is not None
    }
    expected_layers = set(range(source_layer_count))
    if observed_layers != expected_layers:
        raise CheckpointError(
            "source layer index set differs from config: "
            f"missing={sorted(expected_layers - observed_layers)} "
            f"extra={sorted(observed_layers - expected_layers)}")

    layer_types = text_config["layer_types"]
    for layer in range(selected_layer_count):
        prefix = f"model.language_model.layers.{layer}."
        suffixes = {
            name[len(prefix):] for name in weight_map if name.startswith(prefix)
        }
        required = set(COMMON_REQUIRED_SUFFIXES)
        if layer_types[layer] == "linear_attention":
            required.update(LINEAR_REQUIRED_SUFFIXES)
        else:
            required.update(FULL_REQUIRED_SUFFIXES)
        if not required.issubset(suffixes):
            raise CheckpointError(
                f"layer {layer} lacks required target weights: "
                f"{sorted(required - suffixes)}")

    if not any(name.startswith("model.visual.") for name in weight_map):
        raise CheckpointError(
            "source lacks visual weights; multimodal structure would be lost")
    if not any(name.startswith("mtp.") for name in weight_map):
        raise CheckpointError(
            "source lacks MTP weights; non-layer structure would be incomplete")


def _load_source(
    source: Path,
    cycle_count: int,
) -> dict[str, Any]:
    source = source.resolve()
    config_path = source / "config.json"
    index_path = source / INDEX_NAME
    if not source.is_dir():
        raise CheckpointError(f"source is not a directory: {source}")
    if not config_path.is_file() or not index_path.is_file():
        raise CheckpointError(
            f"source must contain config.json and {INDEX_NAME}")

    config = _load_json(config_path)
    text_config = _require_target_config(config)
    layer_count, selected_layer_types = _require_cycle_layout(
        text_config, cycle_count)
    index = _load_json(index_path)
    weight_map = index.get("weight_map")
    if (not isinstance(weight_map, dict) or not weight_map
            or any(not isinstance(name, str) or not isinstance(shard, str)
                   for name, shard in weight_map.items())):
        raise CheckpointError("source index has an invalid weight_map")
    invalid_shards = sorted({
        shard for shard in weight_map.values()
        if Path(shard).name != shard or not SHARD_PATTERN.fullmatch(shard)
    })
    if invalid_shards:
        raise CheckpointError(
            f"source index contains unsafe shard names: {invalid_shards}")
    _validate_weight_names(weight_map, text_config, layer_count)

    shard_headers: dict[str, dict[str, Any]] = {}
    shard_header_sizes: dict[str, int] = {}
    shard_metadata: dict[str, dict[str, str] | None] = {}
    selected_by_shard: dict[str, list[str]] = {}
    selected_payload_bytes = 0

    for weight_name, shard_name in weight_map.items():
        shard_path = source / shard_name
        if shard_name not in shard_headers:
            header_size, tensors, metadata = read_safetensors_header(shard_path)
            shard_header_sizes[shard_name] = header_size
            shard_headers[shard_name] = tensors
            shard_metadata[shard_name] = metadata
        if weight_name not in shard_headers[shard_name]:
            raise CheckpointError(
                f"index maps {weight_name} to {shard_name}, but header lacks it")
        if _selected_weight(weight_name, layer_count):
            selected_by_shard.setdefault(shard_name, []).append(weight_name)
            selected_payload_bytes += _tensor_nbytes(
                shard_headers[shard_name][weight_name], name=weight_name)

    indexed_names = set(weight_map)
    header_names = {
        name for tensors in shard_headers.values() for name in tensors
    }
    if indexed_names != header_names:
        raise CheckpointError(
            "source index/header tensor sets differ: "
            f"index_only={len(indexed_names - header_names)} "
            f"header_only={len(header_names - indexed_names)}")

    return {
        "source": source,
        "config": config,
        "text_config": text_config,
        "index": index,
        "weight_map": weight_map,
        "layer_count": layer_count,
        "selected_layer_types": selected_layer_types,
        "selected_by_shard": selected_by_shard,
        "shard_headers": shard_headers,
        "shard_header_sizes": shard_header_sizes,
        "shard_metadata": shard_metadata,
        "selected_payload_bytes": selected_payload_bytes,
        "selected_weight_count": sum(map(len, selected_by_shard.values())),
    }


def _patched_config(plan: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(plan["config"])
    text_config = config["text_config"]
    text_config["num_hidden_layers"] = plan["layer_count"]
    text_config["layer_types"] = plan["selected_layer_types"]

    if "layers_block_type" in config:
        mode = config.get("bi100_hybrid_kv_accounting_mode", "legacy40")
        if mode == "full_attention":
            config["layers_block_type"] = [
                "attention" if layer_type == "full_attention" else layer_type
                for layer_type in plan["selected_layer_types"]
            ]
        elif mode == "legacy40":
            config["layers_block_type"] = [
                "attention"
            ] * plan["layer_count"]
        else:
            raise CheckpointError(
                f"unsupported serialized hybrid KV accounting mode: {mode!r}")
    return config


def _encoded_header(
    tensors: list[tuple[str, dict[str, Any]]],
    metadata: dict[str, str] | None,
) -> tuple[bytes, int]:
    output_header: dict[str, Any] = {}
    if metadata is not None:
        output_header["__metadata__"] = metadata
    offset = 0
    for name, tensor_metadata in tensors:
        size = _tensor_nbytes(tensor_metadata, name=name)
        output_header[name] = {
            "dtype": tensor_metadata["dtype"],
            "shape": tensor_metadata["shape"],
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(
        output_header, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    padding = (-len(encoded)) % 8
    return encoded + b" " * padding, offset


def _copy_exact(
    source: BinaryIO,
    destination: BinaryIO,
    size: int,
    digest: Any,
) -> None:
    remaining = size
    while remaining:
        chunk = source.read(min(COPY_CHUNK_BYTES, remaining))
        if not chunk:
            raise CheckpointError(
                f"source shard ended with {remaining} bytes left to copy")
        destination.write(chunk)
        digest.update(chunk)
        remaining -= len(chunk)


def _write_filtered_shard(
    source_path: Path,
    output_path: Path,
    source_header_size: int,
    source_tensors: dict[str, dict[str, Any]],
    source_metadata: dict[str, str] | None,
    selected_names: Iterable[str],
) -> dict[str, Any]:
    ordered = sorted(
        ((name, source_tensors[name]) for name in selected_names),
        key=lambda item: item[1]["data_offsets"][0],
    )
    encoded_header, payload_bytes = _encoded_header(ordered, source_metadata)
    prefix = struct.pack("<Q", len(encoded_header))
    digest = hashlib.sha256()
    digest.update(prefix)
    digest.update(encoded_header)

    with source_path.open("rb") as source, output_path.open("wb") as output:
        output.write(prefix)
        output.write(encoded_header)
        for name, metadata in ordered:
            start, end = metadata["data_offsets"]
            source.seek(8 + source_header_size + start)
            _copy_exact(source, output, end - start, digest)
        output.flush()
        os.fsync(output.fileno())

    expected_size = 8 + len(encoded_header) + payload_bytes
    actual_size = output_path.stat().st_size
    if actual_size != expected_size:
        raise CheckpointError(
            f"{output_path}: wrote {actual_size} bytes, expected {expected_size}")
    return {
        "file": output_path.name,
        "source_file": source_path.name,
        "tensor_count": len(ordered),
        "payload_bytes": payload_bytes,
        "file_bytes": actual_size,
        "sha256": digest.hexdigest(),
    }


def _skip_asset(relative: Path) -> bool:
    if any(part.startswith(".") for part in relative.parts):
        return True
    if relative.name in {"config.json", INDEX_NAME, MANIFEST_NAME}:
        return True
    if SHARD_PATTERN.fullmatch(relative.name):
        return True
    if relative.name.endswith(("~", ".tmp", ".swp", ".swo", ".pyc", ".pyo")):
        return True
    if any(part in {"__pycache__", ".git"} for part in relative.parts):
        return True
    return False


def _copy_assets(source: Path, output: Path) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for source_path in sorted(source.rglob("*")):
        if not source_path.is_file():
            continue
        relative = source_path.relative_to(source)
        if _skip_asset(relative):
            continue
        output_path = output / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, output_path)
        source_digest = sha256_file(source_path)
        output_digest = sha256_file(output_path)
        if source_digest != output_digest:
            raise CheckpointError(f"asset copy digest mismatch: {relative}")
        assets.append({
            "file": relative.as_posix(),
            "bytes": output_path.stat().st_size,
            "sha256": output_digest,
        })
    return assets


def _builder_identity() -> dict[str, Any]:
    script = Path(__file__).resolve()
    return {
        "path": script.name,
        "sha256": sha256_file(script),
    }


def describe_plan(source: Path, cycle_count: int) -> dict[str, Any]:
    plan = _load_source(source, cycle_count)
    text_config = plan["text_config"]
    return {
        "source": str(plan["source"]),
        "source_layer_count": text_config["num_hidden_layers"],
        "retained_layer_count": plan["layer_count"],
        "retained_layer_indices": list(range(plan["layer_count"])),
        "retained_layer_types": plan["selected_layer_types"],
        "retained_weight_count": plan["selected_weight_count"],
        "retained_payload_bytes": plan["selected_payload_bytes"],
        "retained_payload_gib": round(
            plan["selected_payload_bytes"] / 2**30, 4),
        "source_shard_count": len(plan["shard_headers"]),
        "output_shard_count": len(plan["selected_by_shard"]),
        "preserves_non_layer_weights": True,
        "preserves_visual_weights": True,
        "preserves_mtp_weights": True,
        "tensor_bytes_transformed": False,
    }


def build_checkpoint(
    source: Path,
    output: Path,
    cycle_count: int,
) -> dict[str, Any]:
    plan = _load_source(source, cycle_count)
    output = output.resolve()
    if output.exists():
        raise CheckpointError(f"output already exists: {output}")
    try:
        output.relative_to(plan["source"])
    except ValueError:
        pass
    else:
        raise CheckpointError(
            "output must not be created inside the source checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)

    free_bytes = shutil.disk_usage(output.parent).free
    estimated_bytes = plan["selected_payload_bytes"] + 512 * 1024 * 1024
    if free_bytes < estimated_bytes:
        raise CheckpointError(
            f"insufficient free space under {output.parent}: "
            f"need at least {estimated_bytes}, have {free_bytes}")

    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.build-", dir=output.parent))
    try:
        assets = _copy_assets(plan["source"], temporary)
        patched_config = _patched_config(plan)
        _write_json(temporary / "config.json", patched_config)

        source_shards = sorted(plan["selected_by_shard"])
        width = max(5, len(str(len(source_shards))))
        shard_name_map = {
            source_name: (
                f"model-{index:0{width}d}-of-"
                f"{len(source_shards):0{width}d}.safetensors"
            )
            for index, source_name in enumerate(source_shards, start=1)
        }
        shard_records: list[dict[str, Any]] = []
        output_weight_map: dict[str, str] = {}
        for source_name in source_shards:
            output_name = shard_name_map[source_name]
            selected_names = plan["selected_by_shard"][source_name]
            record = _write_filtered_shard(
                plan["source"] / source_name,
                temporary / output_name,
                plan["shard_header_sizes"][source_name],
                plan["shard_headers"][source_name],
                plan["shard_metadata"][source_name],
                selected_names,
            )
            shard_records.append(record)
            for weight_name in selected_names:
                output_weight_map[weight_name] = output_name

        output_index = {
            "metadata": {"total_size": plan["selected_payload_bytes"]},
            "weight_map": output_weight_map,
        }
        _write_json(temporary / INDEX_NAME, output_index)

        config_digest = sha256_file(temporary / "config.json")
        index_digest = sha256_file(temporary / INDEX_NAME)
        source_config_digest = sha256_file(plan["source"] / "config.json")
        source_index_digest = sha256_file(plan["source"] / INDEX_NAME)
        manifest = {
            "schema": SCHEMA,
            "version": VERSION,
            "generated_at_utc": dt.datetime.now(
                dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "builder": _builder_identity(),
            "source": {
                "path": str(plan["source"]),
                "config_sha256": source_config_digest,
                "index_sha256": source_index_digest,
                "declared_weight_bytes": plan["index"].get(
                    "metadata", {}).get("total_size"),
                "architecture": plan["config"]["architectures"][0],
                "model_type": plan["config"]["model_type"],
            },
            "diagnostic": {
                "cycle_count": cycle_count,
                "layer_count": plan["layer_count"],
                "retained_layer_indices": list(range(plan["layer_count"])),
                "layer_types": plan["selected_layer_types"],
                "preserves_all_non_language_layer_weights": True,
                "preserves_visual_tower": True,
                "preserves_mtp": True,
                "tensor_payload_transform": "none-byte-for-byte-copy",
                "dtype": plan["text_config"]["dtype"],
                "max_position_embeddings": plan[
                    "text_config"]["max_position_embeddings"],
            },
            "tensor_contract": {
                key: plan["text_config"][key]
                for key in TARGET_TEXT_CONTRACT
            },
            "output": {
                "path": str(output),
                "config_sha256": config_digest,
                "index_sha256": index_digest,
                "weight_count": len(output_weight_map),
                "weight_payload_bytes": plan["selected_payload_bytes"],
                "shard_count": len(shard_records),
                "shards": shard_records,
                "assets": assets,
            },
            "limitations": [
                "Diagnostic depth changes model capability and output quality.",
                "Results do not qualify production throughput or official score.",
                "TP4 and full 40-layer quality gates remain mandatory.",
            ],
        }
        _write_json(temporary / MANIFEST_NAME, manifest)
        os.replace(temporary, output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build an exact-shape Qwen3.6-35B-A3B checkpoint retaining "
            "complete 3-GDN + 1-full-attention cycles."
        ))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the source and print the retained tensor plan",
    )
    args = parser.parse_args(argv)
    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run is used")

    try:
        if args.dry_run:
            result = describe_plan(args.source, args.cycles)
        else:
            result = build_checkpoint(args.source, args.output, args.cycles)
    except CheckpointError as error:
        parser.exit(2, f"checkpoint error: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
