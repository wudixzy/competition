#!/usr/bin/env python3
"""Derive the TP4 rank-0 attention slice from a TP1 activation bank.

The source remains a private, real-weight TP1 capture. The derived bank is an
intermediate operator screen only: Q heads 0..3 and KV head 0 match the
contiguous projection shard assigned to logical TP4 rank 0, but no distributed
TP4 execution is claimed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


SOURCE_BANK_SCHEMA = "bi100-fused-prefill-activation-bank-v1"
SOURCE_CASE_SCHEMA = "bi100-fused-prefill-activation-case-v1"
DERIVED_BANK_SCHEMA = "bi100-m1-176-tp1-derived-tp4-rank0-bank-v1"
DERIVED_CASE_SCHEMA = "bi100-m1-176-tp1-derived-tp4-rank0-case-v1"
BLOCK_SIZE = 16
HEAD_DIM = 256
SOURCE_QUERY_HEADS = 16
SOURCE_KV_HEADS = 2
TARGET_QUERY_HEADS = 4
TARGET_KV_HEADS = 1
FROZEN_SELECTION = {
    "context_buckets": [24576, 57344, 122880],
    "full_attention_call_ordinals": [0],
}
SOURCE_PRIVACY = {
    "raw_activation_files_private": True,
    "raw_activation_files_may_be_committed": False,
    "contains_prompts": False,
    "contains_model_outputs": False,
    "contains_token_ids": False,
    "contains_credentials": False,
}
TENSOR_NAMES = {
    "query", "key", "value", "key_cache", "value_cache", "block_table",
}
SOURCE_MANIFEST_FIELDS = {
    "schema", "version", "run_id", "rank", "source_revision",
    "runtime_identity", "producer", "synthetic_prompt_attestation",
    "selection", "record_count", "records", "privacy",
}
SOURCE_RECORD_FIELDS = {
    "bucket_min_context_tokens", "call_ordinal", "context_tokens",
    "query_length", "file", "sha256", "size_bytes",
    "compact_physical_blocks", "logical_blocks", "tensors",
}
SOURCE_CASE_FIELDS = {
    "schema", "version", "context_tokens", "scale", "rank", "bucket",
    "call_ordinal", "tensors",
}
DERIVATION = {
    "source_tensor_parallel_size": 1,
    "target_tensor_parallel_size": 4,
    "target_logical_tp_rank": 0,
    "query_head_indices": [0, 1, 2, 3],
    "key_value_head_indices": [0],
    "projection_partition": "contiguous",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _private_tmp_file(path: Path) -> bool:
    tmp = Path("/tmp").resolve()
    return (
        path.is_file()
        and path.resolve().is_relative_to(tmp)
        and not path.stat().st_mode & 0o077
        and not path.parent.stat().st_mode & 0o077
    )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            json.dump(
                value, stream, ensure_ascii=True, indent=2,
                sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    import torch

    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.chmod(temporary, 0o600)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _load_case(path: Path) -> dict[str, Any]:
    import torch

    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if (
        not isinstance(value, dict)
        or set(value) != SOURCE_CASE_FIELDS
        or value.get("schema") != SOURCE_CASE_SCHEMA
        or value.get("version") != 1
        or not isinstance(value.get("tensors"), dict)
    ):
        raise ValueError("source activation case contract differs")
    return value


def _tensor_metadata(tensor: Any) -> dict[str, Any]:
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}


def _validate_source_tensors(value: dict[str, Any]) -> dict[str, Any]:
    import torch

    tensors = value["tensors"]
    if set(tensors) != TENSOR_NAMES or not all(
        isinstance(tensor, torch.Tensor) for tensor in tensors.values()
    ):
        raise ValueError("source activation tensor set differs")
    query = tensors["query"]
    key = tensors["key"]
    val = tensors["value"]
    key_cache = tensors["key_cache"]
    value_cache = tensors["value_cache"]
    block_table = tensors["block_table"]
    query_len = query.shape[0] if query.ndim == 3 else -1
    context_len = value.get("context_tokens")
    scale = value.get("scale")
    valid = (
        isinstance(context_len, int)
        and not isinstance(context_len, bool)
        and context_len >= 0
        and context_len % BLOCK_SIZE == 0
        and isinstance(scale, (int, float))
        and not isinstance(scale, bool)
        and math.isfinite(float(scale))
        and math.isclose(
            float(scale), HEAD_DIM ** -0.5, rel_tol=0.0, abs_tol=1e-12)
        and tuple(query.shape) == (query_len, SOURCE_QUERY_HEADS, HEAD_DIM)
        and tuple(key.shape) == (query_len, SOURCE_KV_HEADS, HEAD_DIM)
        and tuple(val.shape) == (query_len, SOURCE_KV_HEADS, HEAD_DIM)
        and key_cache.ndim == 5
        and tuple(key_cache.shape[1:]) == (SOURCE_KV_HEADS, 32, 16, 8)
        and value_cache.ndim == 4
        and tuple(value_cache.shape[1:]) == (
            SOURCE_KV_HEADS, HEAD_DIM, BLOCK_SIZE)
        and key_cache.shape[0] == value_cache.shape[0]
        and tuple(block_table.shape) == (context_len // BLOCK_SIZE,)
        and all(
            tensor.dtype == torch.float16
            for tensor in (query, key, val, key_cache, value_cache)
        )
        and block_table.dtype == torch.int32
        and all(tensor.device.type == "cpu" for tensor in tensors.values())
        and all(tensor.is_contiguous() for tensor in tensors.values())
        and 16 < query_len <= 8192
        and context_len + query_len <= 262144
    )
    if not valid:
        raise ValueError("source TP1 activation shape or dtype differs")
    if block_table.numel():
        compact = sorted(
            int(item) for item in torch.unique(block_table).tolist())
        if compact != list(range(key_cache.shape[0])):
            raise ValueError("source compact block mapping differs")
    return tensors


def derive(
    source_manifest: Path,
    output_dir: Path,
    *,
    expected_source_revision: str,
    expected_runtime_identity: str,
) -> dict[str, Any]:
    source_manifest = source_manifest.resolve(strict=True)
    output_dir = output_dir.resolve()
    if not _private_tmp_file(source_manifest):
        raise ValueError("source manifest must be a private file under /tmp")
    if (
        output_dir == Path("/tmp")
        or not output_dir.is_relative_to(Path("/tmp"))
        or output_dir.exists()
    ):
        raise ValueError("output must be a new private directory under /tmp")
    if not _hex(expected_source_revision, 40):
        raise ValueError("expected source revision is invalid")
    if not expected_runtime_identity:
        raise ValueError("expected runtime identity is empty")

    manifest = json.loads(source_manifest.read_text(encoding="ascii"))
    if (
        not isinstance(manifest, dict)
        or set(manifest) != SOURCE_MANIFEST_FIELDS
        or manifest.get("schema") != SOURCE_BANK_SCHEMA
        or manifest.get("version") != 1
        or manifest.get("rank") != 0
        or manifest.get("source_revision") != expected_source_revision
        or manifest.get("runtime_identity") != expected_runtime_identity
        or manifest.get("producer") != "baseline-pytorch-fallback"
        or manifest.get("synthetic_prompt_attestation")
        != "synthetic-exact-prompt-v1"
        or manifest.get("selection") != FROZEN_SELECTION
        or manifest.get("privacy") != SOURCE_PRIVACY
        or not isinstance(manifest.get("records"), list)
        or manifest.get("record_count") != len(manifest["records"])
        or not manifest["records"]
    ):
        raise ValueError("source TP1 activation manifest differs")

    output_dir.mkdir(mode=0o700, parents=True)
    os.chmod(output_dir, 0o700)
    derived_records = []
    seen_cells: set[tuple[int, int]] = set()
    for source_record in manifest["records"]:
        if (
            not isinstance(source_record, dict)
            or set(source_record) != SOURCE_RECORD_FIELDS
        ):
            raise ValueError("source activation record contract differs")
        filename = source_record.get("file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("source activation filename is invalid")
        source_case = source_manifest.parent / filename
        if (
            not _private_tmp_file(source_case)
            or source_case.stat().st_size != source_record.get("size_bytes")
            or sha256_file(source_case) != source_record.get("sha256")
        ):
            raise ValueError("source activation case identity differs")
        value = _load_case(source_case)
        cell = (value.get("bucket"), value.get("call_ordinal"))
        if (
            value.get("rank") != 0
            or value.get("context_tokens") != source_record.get(
                "context_tokens")
            or cell != (
                source_record.get("bucket_min_context_tokens"),
                source_record.get("call_ordinal"),
            )
            or cell in seen_cells
            or value.get("context_tokens") != cell[0]
        ):
            raise ValueError("source activation metadata differs")
        seen_cells.add(cell)
        tensors = _validate_source_tensors(value)
        if source_record.get("tensors") != {
            name: _tensor_metadata(tensors[name]) for name in sorted(tensors)
        }:
            raise ValueError("source activation tensor metadata differs")
        if (
            source_record.get("query_length") != tensors["query"].shape[0]
            or source_record.get("compact_physical_blocks")
            != tensors["key_cache"].shape[0]
            or source_record.get("logical_blocks")
            != tensors["block_table"].numel()
        ):
            raise ValueError("source activation record dimensions differ")

        derived_tensors = {
            "query": tensors["query"][:, :TARGET_QUERY_HEADS].contiguous(),
            "key": tensors["key"][:, :TARGET_KV_HEADS].contiguous(),
            "value": tensors["value"][:, :TARGET_KV_HEADS].contiguous(),
            "key_cache": tensors["key_cache"][
                :, :TARGET_KV_HEADS].contiguous(),
            "value_cache": tensors["value_cache"][
                :, :TARGET_KV_HEADS].contiguous(),
            "block_table": tensors["block_table"].clone().contiguous(),
        }
        derived_name = (
            f"logical-rank-0.bucket-{cell[0]}.ordinal-{cell[1]}."
            f"ctx-{value['context_tokens']}.q-{derived_tensors['query'].shape[0]}.pt"
        )
        destination = output_dir / derived_name
        source_sha = source_record["sha256"]
        _atomic_torch_save(destination, {
            "schema": DERIVED_CASE_SCHEMA,
            "version": 1,
            "context_tokens": value["context_tokens"],
            "scale": float(value["scale"]),
            "logical_tp_rank": 0,
            "source_case_sha256": source_sha,
            "derivation": DERIVATION,
            "tensors": derived_tensors,
        })
        derived_records.append({
            "bucket_min_context_tokens": cell[0],
            "call_ordinal": cell[1],
            "context_tokens": value["context_tokens"],
            "query_length": int(derived_tensors["query"].shape[0]),
            "file": derived_name,
            "sha256": sha256_file(destination),
            "size_bytes": destination.stat().st_size,
            "source_case_sha256": source_sha,
            "compact_physical_blocks": int(
                derived_tensors["key_cache"].shape[0]),
            "logical_blocks": int(derived_tensors["block_table"].numel()),
            "tensors": {
                name: _tensor_metadata(derived_tensors[name])
                for name in sorted(derived_tensors)
            },
        })

    expected_cells = {
        (bucket, ordinal)
        for bucket in FROZEN_SELECTION["context_buckets"]
        for ordinal in FROZEN_SELECTION["full_attention_call_ordinals"]
    }
    if seen_cells != expected_cells:
        raise ValueError("source activation bank does not cover frozen cells")
    derived_manifest = {
        "schema": DERIVED_BANK_SCHEMA,
        "version": 1,
        "run_id": f"{manifest['run_id']}-tp4-rank0",
        "logical_tp_rank": 0,
        "source_revision": expected_source_revision,
        "runtime_identity": expected_runtime_identity,
        "producer": "tp1-real-weight-contiguous-head-slice",
        "selection": FROZEN_SELECTION,
        "derivation": DERIVATION,
        "source_manifest": {
            "file": source_manifest.name,
            "sha256": sha256_file(source_manifest),
            "source_logical_tp_rank": 0,
            "source_tensor_parallel_size": 1,
        },
        "record_count": len(derived_records),
        "records": sorted(
            derived_records,
            key=lambda row: (
                row["bucket_min_context_tokens"], row["call_ordinal"]),
        ),
        "privacy": SOURCE_PRIVACY,
        "authorization": {
            "operator_screen_only": True,
            "tp4_activation_capture_claim": False,
            "tp4_service_authorized": False,
            "main_or_yaml_change_authorized": False,
        },
    }
    manifest_path = output_dir / "logical-rank-0.manifest.json"
    _atomic_json(manifest_path, derived_manifest)
    return {
        "qualified": True,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "record_count": len(derived_records),
        "source_manifest_sha256": derived_manifest["source_manifest"][
            "sha256"],
        "authorization": derived_manifest["authorization"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--expected-runtime-identity", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = derive(
        args.source_manifest,
        args.output_dir,
        expected_source_revision=args.expected_source_revision,
        expected_runtime_identity=args.expected_runtime_identity,
    )
    _atomic_json(args.report, result)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
