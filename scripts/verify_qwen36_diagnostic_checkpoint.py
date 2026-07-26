#!/usr/bin/env python3
"""Verify structure, identity, and optionally bytes of a diagnostic model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, BinaryIO

from build_qwen36_diagnostic_checkpoint import (
    CheckpointError,
    COPY_CHUNK_BYTES,
    INDEX_NAME,
    MANIFEST_NAME,
    SCHEMA,
    VERSION,
    _load_json,
    _load_source,
    _patched_config,
    _tensor_nbytes,
    read_safetensors_header,
    sha256_file,
)


REPORT_SCHEMA = "qwen36-diagnostic-checkpoint-verification-v1"


def _compare_range(
    source: BinaryIO,
    source_offset: int,
    output: BinaryIO,
    output_offset: int,
    size: int,
    *,
    tensor_name: str,
) -> None:
    source.seek(source_offset)
    output.seek(output_offset)
    remaining = size
    while remaining:
        read_size = min(COPY_CHUNK_BYTES, remaining)
        source_chunk = source.read(read_size)
        output_chunk = output.read(read_size)
        if source_chunk != output_chunk:
            raise CheckpointError(
                f"tensor payload differs from source: {tensor_name}")
        if len(source_chunk) != read_size:
            raise CheckpointError(
                f"truncated tensor while comparing: {tensor_name}")
        remaining -= read_size


def _verify_asset(
    checkpoint: Path,
    record: dict[str, Any],
    *,
    full_hash: bool,
) -> None:
    relative = record.get("file")
    if not isinstance(relative, str):
        raise CheckpointError("manifest asset record lacks file")
    path = checkpoint / relative
    if not path.is_file():
        raise CheckpointError(f"manifest asset is missing: {relative}")
    if path.stat().st_size != record.get("bytes"):
        raise CheckpointError(f"manifest asset size differs: {relative}")
    if full_hash and sha256_file(path) != record.get("sha256"):
        raise CheckpointError(f"manifest asset digest differs: {relative}")


def verify_checkpoint(
    source: Path,
    checkpoint: Path,
    *,
    full_hash: bool = False,
    compare_source_bytes: bool = False,
) -> dict[str, Any]:
    source = source.resolve()
    checkpoint = checkpoint.resolve()
    manifest_path = checkpoint / MANIFEST_NAME
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != SCHEMA or manifest.get("version") != VERSION:
        raise CheckpointError("diagnostic manifest schema/version mismatch")
    diagnostic = manifest.get("diagnostic")
    output_record = manifest.get("output")
    source_record = manifest.get("source")
    if not all(isinstance(value, dict) for value in (
            diagnostic, output_record, source_record)):
        raise CheckpointError("diagnostic manifest sections are invalid")
    cycle_count = diagnostic.get("cycle_count")
    if not isinstance(cycle_count, int):
        raise CheckpointError("manifest cycle_count is invalid")

    plan = _load_source(source, cycle_count)
    if sha256_file(source / "config.json") != source_record.get("config_sha256"):
        raise CheckpointError("source config identity differs from manifest")
    if sha256_file(source / INDEX_NAME) != source_record.get("index_sha256"):
        raise CheckpointError("source index identity differs from manifest")

    expected_config = _patched_config(plan)
    actual_config = _load_json(checkpoint / "config.json")
    if actual_config != expected_config:
        raise CheckpointError(
            "diagnostic config is not the exact reduced-depth source config")
    if sha256_file(checkpoint / "config.json") != output_record.get(
            "config_sha256"):
        raise CheckpointError("diagnostic config digest differs from manifest")

    output_index = _load_json(checkpoint / INDEX_NAME)
    if sha256_file(checkpoint / INDEX_NAME) != output_record.get("index_sha256"):
        raise CheckpointError("diagnostic index digest differs from manifest")
    output_weight_map = output_index.get("weight_map")
    if not isinstance(output_weight_map, dict):
        raise CheckpointError("diagnostic index weight_map is invalid")
    expected_names = {
        name for names in plan["selected_by_shard"].values() for name in names
    }
    if set(output_weight_map) != expected_names:
        raise CheckpointError(
            "diagnostic weight set differs from selected source tensors: "
            f"missing={len(expected_names - set(output_weight_map))} "
            f"extra={len(set(output_weight_map) - expected_names)}")
    if output_record.get("weight_count") != len(output_weight_map):
        raise CheckpointError("manifest weight_count differs from index")
    if output_index.get("metadata", {}).get(
            "total_size") != plan["selected_payload_bytes"]:
        raise CheckpointError("diagnostic index total_size is incorrect")

    shard_records = output_record.get("shards")
    if not isinstance(shard_records, list) or not shard_records:
        raise CheckpointError("manifest shard records are invalid")
    shard_by_output = {
        record.get("file"): record for record in shard_records
        if isinstance(record, dict)
    }
    referenced_shards = set(output_weight_map.values())
    if set(shard_by_output) != referenced_shards:
        raise CheckpointError("manifest shard set differs from index")
    filesystem_shards = {
        path.name for path in checkpoint.glob("*.safetensors")
        if path.is_file()
    }
    if filesystem_shards != referenced_shards:
        raise CheckpointError(
            "filesystem shard set differs from index: "
            f"missing={sorted(referenced_shards - filesystem_shards)} "
            f"extra={sorted(filesystem_shards - referenced_shards)}")

    output_headers: dict[str, tuple[int, dict[str, dict[str, Any]]]] = {}
    for output_name in sorted(referenced_shards):
        record = shard_by_output[output_name]
        output_path = checkpoint / output_name
        header_size, tensors, _ = read_safetensors_header(output_path)
        output_headers[output_name] = (header_size, tensors)
        mapped_names = {
            name for name, shard in output_weight_map.items()
            if shard == output_name
        }
        if set(tensors) != mapped_names:
            raise CheckpointError(
                f"{output_name}: header tensor set differs from index")
        payload_bytes = sum(
            _tensor_nbytes(metadata, name=name)
            for name, metadata in tensors.items()
        )
        if payload_bytes != record.get("payload_bytes"):
            raise CheckpointError(
                f"{output_name}: payload size differs from manifest")
        if output_path.stat().st_size != record.get("file_bytes"):
            raise CheckpointError(
                f"{output_name}: file size differs from manifest")
        if full_hash and sha256_file(output_path) != record.get("sha256"):
            raise CheckpointError(
                f"{output_name}: SHA-256 differs from manifest")

        source_name = record.get("source_file")
        if source_name not in plan["shard_headers"]:
            raise CheckpointError(
                f"{output_name}: unknown source shard {source_name!r}")
        source_tensors = plan["shard_headers"][source_name]
        for name, output_metadata in tensors.items():
            if output_metadata["dtype"] != source_tensors[name]["dtype"]:
                raise CheckpointError(f"{name}: dtype differs from source")
            if output_metadata["shape"] != source_tensors[name]["shape"]:
                raise CheckpointError(f"{name}: shape differs from source")

        if compare_source_bytes:
            source_path = source / source_name
            source_data_offset = 8 + plan["shard_header_sizes"][source_name]
            output_data_offset = 8 + header_size
            with source_path.open("rb") as source_stream, output_path.open(
                    "rb") as output_stream:
                ordered_names = sorted(
                    tensors,
                    key=lambda name: tensors[name]["data_offsets"][0],
                )
                for name in ordered_names:
                    source_start, source_end = source_tensors[
                        name]["data_offsets"]
                    output_start, output_end = tensors[name]["data_offsets"]
                    size = source_end - source_start
                    if output_end - output_start != size:
                        raise CheckpointError(
                            f"{name}: output payload size differs from source")
                    _compare_range(
                        source_stream,
                        source_data_offset + source_start,
                        output_stream,
                        output_data_offset + output_start,
                        size,
                        tensor_name=name,
                    )

    assets = output_record.get("assets")
    if not isinstance(assets, list):
        raise CheckpointError("manifest assets list is invalid")
    for asset in assets:
        if not isinstance(asset, dict):
            raise CheckpointError("manifest asset record is invalid")
        _verify_asset(checkpoint, asset, full_hash=full_hash)

    report = {
        "schema": REPORT_SCHEMA,
        "version": 1,
        "qualified": True,
        "source": str(source),
        "checkpoint": str(checkpoint),
        "cycle_count": cycle_count,
        "layer_count": plan["layer_count"],
        "layer_types": plan["selected_layer_types"],
        "weight_count": len(output_weight_map),
        "weight_payload_bytes": plan["selected_payload_bytes"],
        "shard_count": len(referenced_shards),
        "visual_weight_count": sum(
            name.startswith("model.visual.") for name in output_weight_map),
        "mtp_weight_count": sum(
            name.startswith("mtp.") for name in output_weight_map),
        "full_hash_checked": full_hash,
        "source_payload_bytes_compared": compare_source_bytes,
        "tensor_contract_preserved": True,
        "production_promotion_authorized": False,
    }
    return report


def _atomic_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2,
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
    parser = argparse.ArgumentParser(
        description="Verify a reduced-depth Qwen3.6 diagnostic checkpoint")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--full-hash", action="store_true")
    parser.add_argument("--compare-source-bytes", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    try:
        report = verify_checkpoint(
            args.source,
            args.checkpoint,
            full_hash=args.full_hash,
            compare_source_bytes=args.compare_source_bytes,
        )
    except CheckpointError as error:
        report = {
            "schema": REPORT_SCHEMA,
            "version": 1,
            "qualified": False,
            "error": str(error),
            "production_promotion_authorized": False,
        }
        if args.json_out is not None:
            _atomic_json(args.json_out, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 1
    if args.json_out is not None:
        _atomic_json(args.json_out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
