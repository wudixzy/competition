#!/usr/bin/env python3
"""Replay private real-activation cases against one fused-prefill artifact."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Callable


REPORT_SCHEMA = "bi100-fused-prefill-activation-replay-v1"
BANK_SCHEMA = "bi100-fused-prefill-activation-bank-v1"
CASE_SCHEMA = "bi100-fused-prefill-activation-case-v1"
EXTENSION_MODULE_NAME = "corex_fused_paged_prefill"
RELATIVE_L2_LIMIT = 1.0e-5
ERROR_MULTIPLIER = 2.0
RATIO_FLOOR = 1.0e-12
LSE_RELATIVE_L2_LIMIT = 1.0e-5
WARMUPS = 1
TRIALS = 3
BLOCK_SIZE = 16
BLOCKS_PER_TILE = 32
TILE_TOKENS = BLOCK_SIZE * BLOCKS_PER_TILE
HEAD_DIM = 256
NUM_QUERY_HEADS = 4
NUM_KV_HEADS = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_l2(actual: Any, expected: Any) -> float:
    difference = (actual.float() - expected.float()).norm().item()
    denominator = expected.float().norm().item()
    if denominator == 0:
        return 0.0 if difference == 0 else math.inf
    return difference / denominator


def _update_online(
    scores: Any,
    value: Any,
    running_max: Any,
    running_sum: Any,
    running_output: Any,
) -> None:
    import torch

    block_max = scores.amax(dim=-1)
    new_max = torch.maximum(running_max, block_max)
    correction = torch.exp(running_max - new_max)
    probabilities = scores.sub(new_max.unsqueeze(-1)).exp_()
    running_sum.mul_(correction).add_(probabilities.sum(dim=-1))
    running_output.mul_(correction.unsqueeze(-1)).add_(
        torch.matmul(probabilities, value))
    running_max.copy_(new_max)


def reference_forward(
    query: Any,
    key_new: Any,
    value_new: Any,
    key_cache: Any,
    value_cache: Any,
    block_table: Any,
    context_len: int,
    scale: float,
) -> tuple[Any, Any]:
    """Match the production K-major FP32 online-softmax partitioning."""
    import torch

    query_len = query.shape[0]
    query_fp32 = (
        query.permute(1, 0, 2).float().mul(scale).unsqueeze(0))
    running_max = torch.full(
        (1, NUM_QUERY_HEADS, query_len),
        float("-inf"),
        dtype=torch.float32,
        device=query.device,
    )
    running_sum = torch.zeros_like(running_max)
    running_output = torch.zeros(
        (1, NUM_QUERY_HEADS, query_len, HEAD_DIM),
        dtype=torch.float32,
        device=query.device,
    )

    for token_start in range(0, context_len, TILE_TOKENS):
        token_end = min(token_start + TILE_TOKENS, context_len)
        first_block = token_start // BLOCK_SIZE
        last_block = (token_end + BLOCK_SIZE - 1) // BLOCK_SIZE
        block_ids = block_table[first_block:last_block]
        key = (
            key_cache[block_ids]
            .permute(0, 3, 1, 2, 4)
            .contiguous()
            .view(-1, NUM_KV_HEADS, HEAD_DIM)
        )[:token_end - token_start]
        value = (
            value_cache[block_ids]
            .permute(0, 3, 1, 2)
            .contiguous()
            .view(-1, NUM_KV_HEADS, HEAD_DIM)
        )[:token_end - token_start]
        key_matrix = (
            key.permute(1, 0, 2).unsqueeze(1).transpose(-1, -2).float())
        value_matrix = value.permute(1, 0, 2).unsqueeze(1).float()
        _update_online(
            torch.matmul(query_fp32, key_matrix),
            value_matrix,
            running_max,
            running_sum,
            running_output,
        )

    key_positions = torch.arange(query_len, device=query.device)
    query_positions = torch.arange(query_len, device=query.device)
    for key_start in range(0, query_len, TILE_TOKENS):
        key_end = min(key_start + TILE_TOKENS, query_len)
        key_matrix = (
            key_new[key_start:key_end]
            .permute(1, 0, 2)
            .unsqueeze(1)
            .transpose(-1, -2)
            .float()
        )
        value_matrix = (
            value_new[key_start:key_end]
            .permute(1, 0, 2)
            .unsqueeze(1)
            .float()
        )
        scores = torch.matmul(query_fp32, key_matrix)
        mask = (
            key_positions[key_start:key_end].unsqueeze(0)
            > query_positions.unsqueeze(1)
        )
        scores.masked_fill_(
            mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        _update_online(
            scores,
            value_matrix,
            running_max,
            running_sum,
            running_output,
        )

    output_fp32 = (
        running_output.div(running_sum.unsqueeze(-1))
        .squeeze(0)
        .permute(1, 0, 2)
        .contiguous()
    )
    lse = (
        running_max.add(torch.log(running_sum))
        .squeeze(0)
        .transpose(0, 1)
        .contiguous()
    )
    return output_fp32, lse


def _load_extension(path: Path, expected_sha256: str) -> tuple[Any, dict]:
    path = path.resolve(strict=True)
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError("candidate extension SHA-256 mismatch")
    spec = importlib.util.spec_from_file_location(
        EXTENSION_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot create candidate extension loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "forward", None)):
        raise RuntimeError("candidate extension lacks callable forward")
    return module, {
        "path": str(path),
        "sha256": actual_sha256,
        "size_bytes": path.stat().st_size,
    }


def _load_case(path: Path) -> dict[str, Any]:
    import torch

    try:
        value = torch.load(
            path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if (
        not isinstance(value, dict)
        or value.get("schema") != CASE_SCHEMA
        or value.get("version") != 1
        or not isinstance(value.get("tensors"), dict)
    ):
        raise ValueError("activation case contract differs")
    return value


def validate_bank(
    manifest_path: Path,
    *,
    expected_capture_source_revision: str,
    expected_runtime_identity: str,
) -> tuple[dict[str, Any], list[tuple[dict, Path]]]:
    manifest_path = manifest_path.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != BANK_SCHEMA
        or manifest.get("version") != 1
        or manifest.get("source_revision")
        != expected_capture_source_revision
        or manifest.get("runtime_identity") != expected_runtime_identity
        or manifest.get("producer") != "baseline-pytorch-fallback"
        or manifest.get("synthetic_prompt_attestation")
        != "synthetic-exact-prompt-v1"
        or manifest.get("privacy", {}).get(
            "raw_activation_files_may_be_committed") is not False
    ):
        raise ValueError("activation bank identity differs")
    records = manifest.get("records")
    if (
        not isinstance(records, list)
        or not records
        or manifest.get("record_count") != len(records)
    ):
        raise ValueError("activation bank records are missing")
    resolved = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("activation bank record is malformed")
        filename = record.get("file")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
        ):
            raise ValueError("activation bank filename is invalid")
        path = manifest_path.parent / filename
        if (
            not path.is_file()
            or path.stat().st_size != record.get("size_bytes")
            or sha256_file(path) != record.get("sha256")
        ):
            raise ValueError("activation bank case identity differs")
        resolved.append((record, path))
    return manifest, resolved


def _measure(
    function: Callable[[], tuple[Any, Any]],
) -> tuple[dict[str, Any], tuple[Any, Any]]:
    import torch

    for _ in range(WARMUPS):
        warm = function()
        torch.cuda.synchronize()
        del warm
    trials = []
    result = None
    for _ in range(TRIALS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function()
        end.record()
        torch.cuda.synchronize()
        trials.append(float(start.elapsed_time(end)))
    assert result is not None
    return {
        "warmups": WARMUPS,
        "trials": TRIALS,
        "cuda_trials_ms": trials,
        "cuda_median_ms": statistics.median(trials),
    }, result


def calibrated_metrics(
    candidate: Any,
    reference_fp32: Any,
) -> dict[str, Any]:
    import torch

    rounded = reference_fp32.to(candidate.dtype)
    candidate_fp32 = candidate.float()
    rounded_fp32 = rounded.float()
    candidate_to_fp32_l2 = relative_l2(
        candidate_fp32, reference_fp32)
    rounded_to_fp32_l2 = relative_l2(
        rounded_fp32, reference_fp32)
    candidate_to_fp32_max = float(
        (candidate_fp32 - reference_fp32).abs().max().item())
    rounded_to_fp32_max = float(
        (rounded_fp32 - reference_fp32).abs().max().item())
    relative = relative_l2(candidate, rounded)
    maximum = float(
        (candidate.float() - rounded.float()).abs().max().item())
    finite = bool(
        torch.isfinite(candidate).all().item()
        and torch.isfinite(reference_fp32).all().item()
    )
    qualified = bool(
        finite
        and relative <= RELATIVE_L2_LIMIT
        and candidate_to_fp32_l2
        <= ERROR_MULTIPLIER * rounded_to_fp32_l2 + RATIO_FLOOR
        and candidate_to_fp32_max
        <= ERROR_MULTIPLIER * rounded_to_fp32_max + RATIO_FLOOR
    )
    return {
        "finite": finite,
        "candidate_vs_rounded_relative_l2": relative,
        "candidate_vs_rounded_max_abs_diagnostic": maximum,
        "candidate_to_fp32_relative_l2": candidate_to_fp32_l2,
        "candidate_to_fp32_max_abs": candidate_to_fp32_max,
        "rounded_to_fp32_relative_l2": rounded_to_fp32_l2,
        "rounded_to_fp32_max_abs": rounded_to_fp32_max,
        "qualified": qualified,
    }


def _validate_case_tensors(value: dict[str, Any]) -> tuple[Any, ...]:
    import torch

    tensors = value["tensors"]
    required = {
        "query", "key", "value", "key_cache", "value_cache", "block_table",
    }
    if set(tensors) != required:
        raise ValueError("activation case tensor set differs")
    query = tensors["query"]
    key = tensors["key"]
    val = tensors["value"]
    key_cache = tensors["key_cache"]
    value_cache = tensors["value_cache"]
    block_table = tensors["block_table"]
    query_len = query.shape[0]
    context_len = value.get("context_tokens")
    expected = (
        tuple(query.shape) == (query_len, 4, 256)
        and tuple(key.shape) == (query_len, 1, 256)
        and tuple(val.shape) == (query_len, 1, 256)
        and tuple(key_cache.shape[1:]) == (1, 32, 16, 8)
        and tuple(value_cache.shape[1:]) == (1, 256, 16)
        and key_cache.shape[0] == value_cache.shape[0]
        and block_table.ndim == 1
        and block_table.numel() == context_len // BLOCK_SIZE
        and all(
            tensor.dtype == torch.float16
            for tensor in (query, key, val, key_cache, value_cache)
        )
        and block_table.dtype == torch.int32
        and 16 < query_len <= 8192
        and isinstance(context_len, int)
        and context_len >= 0
        and context_len % BLOCK_SIZE == 0
        and context_len + query_len <= 262144
    )
    if not expected:
        raise ValueError("activation case tensor shape or dtype differs")
    if block_table.numel():
        minimum = int(block_table.min().item())
        maximum = int(block_table.max().item())
        if minimum < 0 or maximum >= key_cache.shape[0]:
            raise ValueError("activation case block table is invalid")
    return query, key, val, key_cache, value_cache, block_table


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("replay requires exactly one visible CoreX GPU")
    manifest, cases = validate_bank(
        args.bank_manifest,
        expected_capture_source_revision=(
            args.capture_source_revision),
        expected_runtime_identity=args.runtime_identity,
    )
    extension, artifact = _load_extension(
        args.candidate_extension,
        args.expected_candidate_sha256,
    )
    records = []
    for bank_record, path in cases:
        loaded_started = time.monotonic()
        value = _load_case(path)
        if (
            value.get("rank") != manifest.get("rank")
            or value.get("bucket")
            != bank_record.get("bucket_min_context_tokens")
            or value.get("call_ordinal") != bank_record.get("call_ordinal")
            or value.get("context_tokens")
            != bank_record.get("context_tokens")
        ):
            raise ValueError("activation case metadata differs from manifest")
        tensors = _validate_case_tensors(value)
        query, key, val, key_cache, value_cache, block_table = (
            tensor.cuda() for tensor in tensors
        )
        torch.cuda.synchronize()
        load_elapsed_s = time.monotonic() - loaded_started
        context_len = value["context_tokens"]
        scale = float(value["scale"])
        reference_call = lambda: reference_forward(
            query, key, val, key_cache, value_cache, block_table,
            context_len, scale)
        candidate_call = lambda: tuple(extension.forward(
            query, key, val, key_cache, value_cache, block_table,
            context_len, scale))
        reference_timing, reference = _measure(reference_call)
        candidate_timing, candidate = _measure(candidate_call)
        reference_output, reference_lse = reference
        candidate_output, candidate_lse = candidate
        numeric = calibrated_metrics(candidate_output, reference_output)
        lse_finite = bool(torch.isfinite(candidate_lse).all().item())
        lse_relative = relative_l2(candidate_lse, reference_lse)
        numeric["lse_finite"] = lse_finite
        numeric["lse_relative_l2"] = lse_relative
        numeric["qualified"] = bool(
            numeric["qualified"]
            and lse_finite
            and lse_relative <= LSE_RELATIVE_L2_LIMIT)
        records.append({
            "rank": manifest["rank"],
            "bucket_min_context_tokens": bank_record[
                "bucket_min_context_tokens"],
            "call_ordinal": bank_record["call_ordinal"],
            "context_tokens": context_len,
            "query_length": int(query.shape[0]),
            "case_sha256": bank_record["sha256"],
            "load_elapsed_s": load_elapsed_s,
            "reference_timing": reference_timing,
            "candidate_timing": candidate_timing,
            "candidate_speedup": (
                reference_timing["cuda_median_ms"]
                / candidate_timing["cuda_median_ms"]),
            "numeric": numeric,
        })
        del (
            query, key, val, key_cache, value_cache, block_table,
            reference, candidate,
        )
        torch.cuda.empty_cache()
    return {
        "schema": REPORT_SCHEMA,
        "version": 1,
        "capture_source_revision": args.capture_source_revision,
        "candidate_source_revision": args.candidate_source_revision,
        "runtime_identity": args.runtime_identity,
        "instance": args.instance,
        "visible_physical_gpu": args.visible_physical_gpu,
        "rank": manifest["rank"],
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "bank": {
            "manifest": str(args.bank_manifest.resolve()),
            "manifest_sha256": sha256_file(args.bank_manifest),
            "run_id": manifest["run_id"],
            "record_count": len(cases),
        },
        "candidate_extension": artifact,
        "records": records,
        "all_numeric_qualified": all(
            record["numeric"]["qualified"] for record in records),
        "privacy": {
            "raw_tensors_persisted_in_report": False,
            "prompts_persisted_in_report": False,
            "model_outputs_persisted_in_report": False,
            "credentials_persisted_in_report": False,
        },
        "authorization": {
            "short_tp4_authorized": False,
            "main_or_yaml_change_authorized": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--candidate-extension", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--capture-source-revision", required=True)
    parser.add_argument("--candidate-source-revision", required=True)
    parser.add_argument("--runtime-identity", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--visible-physical-gpu", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="ascii",
    )
    print(json.dumps({
        "rank": report["rank"],
        "records": len(report["records"]),
        "all_numeric_qualified": report["all_numeric_qualified"],
        "median_speedup": statistics.median(
            record["candidate_speedup"]
            for record in report["records"]),
    }, ensure_ascii=True, sort_keys=True))
    return 0 if report["all_numeric_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
