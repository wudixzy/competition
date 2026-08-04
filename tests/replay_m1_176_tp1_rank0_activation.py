#!/usr/bin/env python3
"""Replay a TP1-derived logical TP4 rank-0 bank on one physical BI100."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any

import replay_fused_prefill_activation as base


REPORT_SCHEMA = "bi100-m1-176-tp1-derived-rank0-replay-v1"
BANK_SCHEMA = "bi100-m1-176-tp1-derived-tp4-rank0-bank-v1"
CASE_SCHEMA = "bi100-m1-176-tp1-derived-tp4-rank0-case-v1"
MINIMUM_SPEEDUP = 1.10
LSE_RELATIVE_L2_LIMIT = 1.0e-5
DERIVATION = {
    "source_tensor_parallel_size": 1,
    "target_tensor_parallel_size": 4,
    "target_logical_tp_rank": 0,
    "query_head_indices": [0, 1, 2, 3],
    "key_value_head_indices": [0],
    "projection_partition": "contiguous",
}
FROZEN_SELECTION = {
    "context_buckets": [24576, 57344, 122880],
    "full_attention_call_ordinals": [0],
}
PRIVACY = {
    "raw_activation_files_private": True,
    "raw_activation_files_may_be_committed": False,
    "contains_prompts": False,
    "contains_model_outputs": False,
    "contains_token_ids": False,
    "contains_credentials": False,
}
AUTHORIZATION = {
    "operator_screen_only": True,
    "tp4_activation_capture_claim": False,
    "tp4_service_authorized": False,
    "main_or_yaml_change_authorized": False,
}
BANK_FIELDS = {
    "schema", "version", "run_id", "logical_tp_rank", "source_revision",
    "runtime_identity", "producer", "selection", "derivation",
    "source_manifest", "record_count", "records", "privacy",
    "authorization",
}
RECORD_FIELDS = {
    "bucket_min_context_tokens", "call_ordinal", "context_tokens",
    "query_length", "file", "sha256", "size_bytes", "source_case_sha256",
    "compact_physical_blocks", "logical_blocks", "tensors",
}
CASE_FIELDS = {
    "schema", "version", "context_tokens", "scale", "logical_tp_rank",
    "source_case_sha256", "derivation", "tensors",
}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
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


def _load_case(path: Path) -> dict[str, Any]:
    import torch

    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if (
        not isinstance(value, dict)
        or set(value) != CASE_FIELDS
        or value.get("schema") != CASE_SCHEMA
        or value.get("version") != 1
        or value.get("logical_tp_rank") != 0
        or value.get("derivation") != DERIVATION
    ):
        raise ValueError("derived activation case contract differs")
    return value


def _load_bank(
    path: Path,
    *,
    source_revision: str,
    runtime_identity: str,
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], Path]]]:
    path = path.resolve(strict=True)
    if not _private_tmp_file(path):
        raise ValueError("derived bank must be private and under /tmp")
    manifest = json.loads(path.read_text(encoding="ascii"))
    if (
        not isinstance(manifest, dict)
        or set(manifest) != BANK_FIELDS
        or manifest.get("schema") != BANK_SCHEMA
        or manifest.get("version") != 1
        or manifest.get("logical_tp_rank") != 0
        or manifest.get("source_revision") != source_revision
        or manifest.get("runtime_identity") != runtime_identity
        or manifest.get("producer")
        != "tp1-real-weight-contiguous-head-slice"
        or manifest.get("selection") != FROZEN_SELECTION
        or manifest.get("derivation") != DERIVATION
        or manifest.get("privacy") != PRIVACY
        or not isinstance(manifest.get("records"), list)
        or not manifest["records"]
        or manifest.get("record_count") != len(manifest["records"])
        or manifest.get("authorization") != AUTHORIZATION
    ):
        raise ValueError("derived activation bank contract differs")
    source_manifest = manifest.get("source_manifest")
    if (
        not isinstance(source_manifest, dict)
        or set(source_manifest) != {
            "file", "sha256", "source_logical_tp_rank",
            "source_tensor_parallel_size",
        }
        or not isinstance(source_manifest.get("file"), str)
        or Path(source_manifest["file"]).name != source_manifest["file"]
        or not _hex(source_manifest.get("sha256"), 64)
        or source_manifest.get("source_logical_tp_rank") != 0
        or source_manifest.get("source_tensor_parallel_size") != 1
    ):
        raise ValueError("derived activation source lineage differs")
    cases = []
    seen_cells: set[tuple[int, int]] = set()
    seen_sources: set[str] = set()
    for record in manifest["records"]:
        if not isinstance(record, dict) or set(record) != RECORD_FIELDS:
            raise ValueError("derived activation record contract differs")
        filename = record.get("file")
        cell = (
            record.get("bucket_min_context_tokens"),
            record.get("call_ordinal"),
        )
        case_path = path.parent / filename if isinstance(filename, str) else path
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or cell in seen_cells
            or not _hex(record.get("sha256"), 64)
            or not _hex(record.get("source_case_sha256"), 64)
            or record["source_case_sha256"] in seen_sources
            or not _private_tmp_file(case_path)
            or case_path.stat().st_size != record.get("size_bytes")
            or base.sha256_file(case_path) != record["sha256"]
        ):
            raise ValueError("derived activation case identity differs")
        bucket, ordinal = cell
        if (
            bucket not in FROZEN_SELECTION["context_buckets"]
            or ordinal not in FROZEN_SELECTION[
                "full_attention_call_ordinals"]
            or record.get("context_tokens") != bucket
            or record.get("logical_blocks") != bucket // base.BLOCK_SIZE
        ):
            raise ValueError("derived activation cell differs")
        seen_cells.add(cell)
        seen_sources.add(record["source_case_sha256"])
        cases.append((record, case_path))
    expected_cells = {
        (bucket, ordinal)
        for bucket in FROZEN_SELECTION["context_buckets"]
        for ordinal in FROZEN_SELECTION["full_attention_call_ordinals"]
    }
    if seen_cells != expected_cells:
        raise ValueError("derived activation coverage differs")
    return manifest, cases


def _numeric(output: Any, lse: Any, reference: tuple[Any, Any]) -> dict[str, Any]:
    import torch

    reference_output, reference_lse = reference
    result = base.calibrated_metrics(output, reference_output)
    output_calibrated_qualified = bool(
        result["finite"]
        and result["candidate_to_fp32_relative_l2"]
        <= base.ERROR_MULTIPLIER
        * result["rounded_to_fp32_relative_l2"] + base.RATIO_FLOOR
        and result["candidate_to_fp32_max_abs"]
        <= base.ERROR_MULTIPLIER
        * result["rounded_to_fp32_max_abs"] + base.RATIO_FLOOR
    )
    result.update({
        "candidate_vs_rounded_is_diagnostic_only": True,
        "output_calibrated_qualified": output_calibrated_qualified,
        "candidate_lse_finite": bool(torch.isfinite(lse).all().item()),
        "reference_lse_finite": bool(
            torch.isfinite(reference_lse).all().item()),
        "lse_relative_l2": base.relative_l2(lse, reference_lse),
    })
    result["qualified"] = bool(
        output_calibrated_qualified
        and result["candidate_lse_finite"]
        and result["reference_lse_finite"]
        and result["lse_relative_l2"] <= LSE_RELATIVE_L2_LIMIT
    )
    return result


def _balanced_measure(
    baseline_call: Any,
    candidate_call: Any,
) -> tuple[dict[str, Any], tuple[Any, Any], tuple[Any, Any]]:
    baseline_forward, baseline_result = base._measure(baseline_call)
    candidate_forward, candidate_result = base._measure(candidate_call)
    candidate_reverse, candidate_repeat = base._measure(candidate_call)
    baseline_reverse, baseline_repeat = base._measure(baseline_call)
    forward_speedup = (
        baseline_forward["cuda_median_ms"]
        / candidate_forward["cuda_median_ms"])
    reverse_speedup = (
        baseline_reverse["cuda_median_ms"]
        / candidate_reverse["cuda_median_ms"])
    return ({
        "baseline_forward": baseline_forward,
        "candidate_forward": candidate_forward,
        "candidate_reverse": candidate_reverse,
        "baseline_reverse": baseline_reverse,
        "forward_speedup": forward_speedup,
        "reverse_speedup": reverse_speedup,
        "order_balanced_geometric_speedup": math.sqrt(
            forward_speedup * reverse_speedup),
        "baseline_repeat_exact": {
            "output": bool(
                (baseline_result[0] == baseline_repeat[0]).all().item()),
            "lse": bool(
                (baseline_result[1] == baseline_repeat[1]).all().item()),
        },
        "candidate_repeat_exact": {
            "output": bool(
                (candidate_result[0] == candidate_repeat[0]).all().item()),
            "lse": bool(
                (candidate_result[1] == candidate_repeat[1]).all().item()),
        },
    }, baseline_result, candidate_result)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("M1-176 replay requires one visible CoreX GPU")
    for revision in (
        args.capture_source_revision,
        args.baseline_source_revision,
        args.candidate_source_revision,
    ):
        if not _hex(revision, 40):
            raise ValueError("source revision identity is invalid")
    if (
        args.visible_physical_gpu not in {0, 1, 2, 3}
        or isinstance(args.visible_physical_gpu, bool)
        or not args.runtime_identity
        or not args.instance
    ):
        raise ValueError("runtime invocation identity is invalid")
    manifest, cases = _load_bank(
        args.bank_manifest,
        source_revision=args.capture_source_revision,
        runtime_identity=args.runtime_identity,
    )
    baseline_path = args.baseline_extension.resolve(strict=True)
    candidate_path = args.candidate_extension.resolve(strict=True)
    if (
        not _private_tmp_file(baseline_path)
        or not _private_tmp_file(candidate_path)
        or baseline_path == candidate_path
    ):
        raise ValueError(
            "baseline and candidate must be distinct private /tmp artifacts")
    baseline, baseline_artifact = base._load_extension(
        baseline_path, args.expected_baseline_sha256)
    candidate, candidate_artifact = base._load_extension(
        candidate_path, args.expected_candidate_sha256)

    records = []
    for bank_record, path in cases:
        value = _load_case(path)
        if (
            value.get("source_case_sha256")
            != bank_record.get("source_case_sha256")
            or value.get("context_tokens")
            != bank_record.get("context_tokens")
        ):
            raise ValueError("derived case lineage differs")
        tensors_cpu = base._validate_case_tensors(value)
        if (
            bank_record.get("query_length") != tensors_cpu[0].shape[0]
            or bank_record.get("compact_physical_blocks")
            != tensors_cpu[3].shape[0]
            or bank_record.get("logical_blocks")
            != tensors_cpu[5].numel()
            or bank_record.get("tensors") != {
                name: {
                    "shape": list(value["tensors"][name].shape),
                    "dtype": str(value["tensors"][name].dtype),
                }
                for name in sorted(value["tensors"])
            }
        ):
            raise ValueError("derived case metadata differs")
        tensors = tuple(tensor.cuda() for tensor in tensors_cpu)
        torch.cuda.synchronize()
        context_len = value["context_tokens"]
        scale = float(value["scale"])
        reference_call = lambda: base.reference_forward(
            *tensors, context_len, scale)
        baseline_call = lambda: base._candidate_forward(
            baseline, tensors, context_len, scale)
        candidate_call = lambda: base._candidate_forward(
            candidate, tensors, context_len, scale)
        reference_timing, reference = base._measure(reference_call)
        timing, baseline_result, candidate_result = _balanced_measure(
            baseline_call, candidate_call)
        baseline_numeric = _numeric(*baseline_result, reference)
        candidate_numeric = _numeric(*candidate_result, reference)
        candidate_vs_baseline = {
            "output_relative_l2": base.relative_l2(
                candidate_result[0], baseline_result[0]),
            "output_max_abs": float(
                (candidate_result[0].float() - baseline_result[0].float())
                .abs().max().item()),
            "lse_relative_l2": base.relative_l2(
                candidate_result[1], baseline_result[1]),
        }
        repeat_exact = all(
            timing[name][kind]
            for name in ("baseline_repeat_exact", "candidate_repeat_exact")
            for kind in ("output", "lse")
        )
        qualified = bool(
            baseline_numeric["qualified"]
            and candidate_numeric["qualified"]
            and repeat_exact
            and timing["order_balanced_geometric_speedup"]
            >= MINIMUM_SPEEDUP
        )
        records.append({
            "logical_tp_rank": 0,
            "visible_physical_gpu": args.visible_physical_gpu,
            "bucket_min_context_tokens": bank_record[
                "bucket_min_context_tokens"],
            "call_ordinal": bank_record["call_ordinal"],
            "context_tokens": context_len,
            "query_length": int(tensors[0].shape[0]),
            "derived_case_sha256": bank_record["sha256"],
            "source_case_sha256": bank_record["source_case_sha256"],
            "reference_timing": reference_timing,
            "timing": timing,
            "baseline_numeric": baseline_numeric,
            "candidate_numeric": candidate_numeric,
            "candidate_vs_baseline": candidate_vs_baseline,
            "qualified": qualified,
        })
        del tensors, reference, baseline_result, candidate_result
        torch.cuda.empty_cache()

    report = {
        "schema": REPORT_SCHEMA,
        "version": 1,
        "capture_source_revision": args.capture_source_revision,
        "baseline_source_revision": args.baseline_source_revision,
        "candidate_source_revision": args.candidate_source_revision,
        "runtime_identity": args.runtime_identity,
        "instance": args.instance,
        "visible_physical_gpu": args.visible_physical_gpu,
        "logical_tp_rank": 0,
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "bank": {
            "manifest": str(args.bank_manifest.resolve()),
            "manifest_sha256": base.sha256_file(args.bank_manifest),
            "run_id": manifest["run_id"],
            "record_count": len(cases),
            "producer": manifest["producer"],
        },
        "baseline_extension": baseline_artifact,
        "candidate_extension": candidate_artifact,
        "minimum_speedup": MINIMUM_SPEEDUP,
        "records": records,
        "all_qualified": all(record["qualified"] for record in records),
        "privacy": {
            "raw_tensors_persisted_in_report": False,
            "prompts_persisted_in_report": False,
            "model_outputs_persisted_in_report": False,
            "credentials_persisted_in_report": False,
        },
        "authorization": {
            "real_weight_tp1_operator_screen_only": True,
            "tp4_activation_capture_claim": False,
            "tp4_service_authorized": False,
            "model_quality_evaluated": False,
            "main_or_yaml_change_authorized": False,
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--baseline-extension", type=Path, required=True)
    parser.add_argument("--expected-baseline-sha256", required=True)
    parser.add_argument("--candidate-extension", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--capture-source-revision", required=True)
    parser.add_argument("--baseline-source-revision", required=True)
    parser.add_argument("--candidate-source-revision", required=True)
    parser.add_argument("--runtime-identity", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--visible-physical-gpu", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    _atomic_json(args.out, report)
    speedups = [
        record["timing"]["order_balanced_geometric_speedup"]
        for record in report["records"]
    ]
    print(json.dumps({
        "all_qualified": report["all_qualified"],
        "record_count": len(report["records"]),
        "median_order_balanced_speedup": statistics.median(speedups),
    }, ensure_ascii=True, sort_keys=True, allow_nan=False))
    return 0 if report["all_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
