#!/usr/bin/env python3
"""Replay M1-109 on two retained real-activation cells for one TP4 rank."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import replay_fused_prefill_activation as base
import replay_m1_176_tp1_rank0_activation as m176


SCHEMA = "bi100-m1-181-m1-109-rank-replay-v1"
CONTEXTS = (57344, 122880)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="ascii") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _numeric(candidate: tuple[Any, Any], reference: tuple[Any, Any],
             rounded: tuple[Any, Any]) -> dict[str, Any]:
    import torch

    output, lse = candidate
    reference_output, reference_lse = reference
    rounded_output, rounded_lse = rounded
    candidate_relative = base.relative_l2(output, reference_output)
    rounded_relative = base.relative_l2(rounded_output, reference_output)
    candidate_max_abs = float(
        (output.float() - reference_output.float()).abs().max().item())
    rounded_max_abs = float(
        (rounded_output.float() - reference_output.float()).abs().max().item())
    candidate_lse_error = base.relative_l2(lse, reference_lse)
    rounded_lse_error = base.relative_l2(rounded_lse, reference_lse)
    relative_ratio = candidate_relative / max(
        rounded_relative, m176.RATIO_FLOOR)
    max_abs_ratio = candidate_max_abs / max(
        rounded_max_abs, m176.RATIO_FLOOR)
    lse_limit = max(m176.LSE_RELATIVE_L2_FLOOR,
                    m176.ERROR_MULTIPLIER * rounded_lse_error)
    finite = all(bool(torch.isfinite(value).all().item()) for value in (
        output, lse, reference_output, reference_lse,
        rounded_output, rounded_lse))
    qualified = bool(
        finite
        and relative_ratio <= m176.ERROR_MULTIPLIER
        and max_abs_ratio <= m176.ERROR_MULTIPLIER
        and candidate_lse_error <= lse_limit)
    return {
        "all_finite": finite,
        "candidate_to_fp32_relative_l2": candidate_relative,
        "rounded_to_fp32_relative_l2": rounded_relative,
        "relative_l2_error_ratio": relative_ratio,
        "candidate_to_fp32_max_abs": candidate_max_abs,
        "rounded_to_fp32_max_abs": rounded_max_abs,
        "maximum_absolute_error_ratio": max_abs_ratio,
        "candidate_lse_relative_l2": candidate_lse_error,
        "rounded_lse_relative_l2": rounded_lse_error,
        "attention_lse_limit": lse_limit,
        "candidate_vs_rounded_relative_l2": base.relative_l2(
            output, rounded_output),
        "candidate_vs_rounded_max_abs": float(
            (output.float() - rounded_output.float()).abs().max().item()),
        "candidate_vs_rounded_role": "diagnostic_only",
        "ratio_denominator_floor": m176.RATIO_FLOOR,
        "error_ratio_limit": m176.ERROR_MULTIPLIER,
        "qualified": qualified,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("one visible CoreX GPU is required")
    manifest_raw = json.loads(args.bank_manifest.read_text(encoding="ascii"))
    manifest, cases = m176._load_bank(
        args.bank_manifest,
        source_revision=manifest_raw["source_revision"],
        runtime_identity=manifest_raw["runtime_identity"],
        logical_rank=args.logical_rank)
    extension, extension_identity = m176._load_extension(
        args.extension, args.extension_sha256, "corex_fused_paged_prefill")
    selected = [(record, path) for record, path in cases
                if record["context_tokens"] in CONTEXTS]
    if tuple(record["context_tokens"] for record, _ in selected) != CONTEXTS:
        raise ValueError("retained 65K/131K activation cells are incomplete")

    records = []
    started = time.monotonic()
    for record, path in selected:
        value = m176._load_case(path)
        tensors_cpu = base._validate_case_tensors(value)
        tensors = tuple(tensor.cuda() for tensor in tensors_cpu)
        context_len = value["context_tokens"]
        scale = float(value["scale"])
        reference = base.reference_forward(*tensors, context_len, scale)
        rounded = (reference[0].to(tensors[0].dtype), reference[1])
        first = base._candidate_forward(extension, tensors, context_len, scale)
        second = base._candidate_forward(extension, tensors, context_len, scale)
        torch.cuda.synchronize()
        repeat_exact = {
            "output": bool(torch.equal(first[0], second[0])),
            "lse": bool(torch.equal(first[1], second[1])),
        }
        numeric = _numeric(first, reference, rounded)
        qualified = numeric["qualified"] and all(repeat_exact.values())
        records.append({
            "context_tokens": context_len,
            "total_attention_tokens": context_len + int(tensors[0].shape[0]),
            "query_length": int(tensors[0].shape[0]),
            "numeric": numeric,
            "repeat_exact": repeat_exact,
            "qualified": qualified,
        })
        del tensors, reference, rounded, first, second
        torch.cuda.empty_cache()
    return {
        "schema": SCHEMA,
        "version": 1,
        "logical_tp_rank": args.logical_rank,
        "physical_gpu": args.physical_gpu,
        "device_name": torch.cuda.get_device_name(0),
        "runtime_identity": manifest["runtime_identity"],
        "capture_source_revision": manifest["source_revision"],
        "extension": extension_identity,
        "records": records,
        "all_qualified": all(record["qualified"] for record in records),
        "wall_s": time.monotonic() - started,
        "privacy": {
            "contains_prompts": False,
            "contains_model_outputs": False,
            "contains_tensor_values": False,
            "contains_token_ids": False,
            "contains_credentials": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--extension-sha256", required=True)
    parser.add_argument("--logical-rank", type=int, choices=range(4), required=True)
    parser.add_argument("--physical-gpu", type=int, choices=range(4), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args)
        _atomic_json(args.out, result)
        return 0 if result["all_qualified"] else 1
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"M1-181 rank replay invalid: {type(exc).__name__}: {exc}",
              file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
