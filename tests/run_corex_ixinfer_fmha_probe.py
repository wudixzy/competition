#!/usr/bin/env python3
"""Run one isolated CoreX ixinfer FMHA capability and numerical cell."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any


SCHEMA = "bi100-m1-160-ixinfer-fmha-probe-v1"
SEED = 20260730


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_extension(path: Path, expected_sha256: str) -> Any:
    resolved = path.resolve(strict=True)
    actual_sha256 = _sha256(resolved)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "extension SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    spec = importlib.util.spec_from_file_location(
        "corex_ixinfer_fmha_probe", resolved
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension from {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "forward", None)):
        raise RuntimeError("extension does not expose forward")
    return module


def _reference(query: Any, key: Any, value: Any, causal: bool) -> Any:
    import torch

    query_float = query.float()
    key_float = key.float()
    value_float = value.float()
    query_heads = query_float.size(2)
    kv_heads = key_float.size(2)
    repeats = query_heads // kv_heads
    key_float = key_float.repeat_interleave(repeats, dim=2)
    value_float = value_float.repeat_interleave(repeats, dim=2)
    scores = torch.einsum(
        "bqhd,bkhd->bhqk", query_float, key_float
    ) * (query.size(-1) ** -0.5)
    if causal:
        query_length = query.size(1)
        key_length = key.size(1)
        query_positions = torch.arange(
            query_length, device=query.device
        ).unsqueeze(1)
        key_positions = torch.arange(
            key_length, device=query.device
        ).unsqueeze(0)
        allowed = key_positions <= key_length - query_length + query_positions
        scores = scores.masked_fill(
            ~allowed.unsqueeze(0).unsqueeze(0), float("-inf")
        )
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.einsum(
        "bhqk,bkhd->bqhd", probabilities, value_float
    )
    return output.to(query.dtype)


def _relative_l2(actual: Any, expected: Any) -> float:
    import torch

    difference = (actual.float() - expected.float()).norm()
    denominator = expected.float().norm().clamp_min(
        torch.finfo(torch.float32).tiny
    )
    return float((difference / denominator).item())


def _timed_call(function: Any, trials: int) -> tuple[list[float], Any]:
    import torch

    for _ in range(2):
        output = function()
        torch.cuda.synchronize()
    timings = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function()
        end.record()
        torch.cuda.synchronize()
        timings.append(float(start.elapsed_time(end)))
    return timings, output


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("probe requires one visible healthy BI100")
    if args.layout != "bshd":
        raise RuntimeError("the first bounded probe supports only BSHD")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    extension = _load_extension(args.extension, args.expected_sha256)
    query = (
        torch.randn(
            1, args.query_length, args.query_heads, args.head_size,
            device="cuda", dtype=torch.float32,
        )
        .mul_(args.input_scale)
        .half()
    )
    key = (
        torch.randn(
            1, args.key_length, args.kv_heads, args.head_size,
            device="cuda", dtype=torch.float32,
        )
        .mul_(args.input_scale)
        .half()
    )
    value = (
        torch.randn(
            1, args.key_length, args.kv_heads, args.head_size,
            device="cuda", dtype=torch.float32,
        )
        .mul_(args.input_scale)
        .half()
    )
    call = lambda: extension.forward(query, key, value, args.causal, 1)
    started = time.monotonic()
    timings, actual = _timed_call(call, args.trials)
    reference = _reference(query, key, value, args.causal)
    difference = actual.float() - reference.float()
    result = {
        "schema": SCHEMA,
        "version": 1,
        "source_revision": args.source_revision,
        "runtime_identity": args.runtime_identity,
        "instance": args.instance,
        "visible_physical_gpu": args.visible_physical_gpu,
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "extension": {
            "path": str(args.extension.resolve(strict=True)),
            "sha256": args.expected_sha256,
        },
        "shape": {
            "batch": 1,
            "query_length": args.query_length,
            "key_length": args.key_length,
            "query_heads": args.query_heads,
            "kv_heads": args.kv_heads,
            "head_size": args.head_size,
            "layout": args.layout,
            "causal": args.causal,
        },
        "seed": SEED,
        "input_scale": args.input_scale,
        "trials": args.trials,
        "timing": {
            "cuda_trials_ms": timings,
            "cuda_median_ms": statistics.median(timings),
        },
        "numerical": {
            "finite": bool(torch.isfinite(actual).all().item()),
            "relative_l2": _relative_l2(actual, reference),
            "max_abs": float(difference.abs().max().item()),
            "candidate_norm": float(actual.float().norm().item()),
            "reference_norm": float(reference.float().norm().item()),
        },
        "elapsed_s": time.monotonic() - started,
        "authorization": {
            "capability_probe_only": True,
            "runtime_overlay_authorized": False,
            "tp4_service_authorized": False,
            "main_or_yaml_change_authorized": False,
        },
    }
    result["qualified"] = (
        result["numerical"]["finite"]
        and math.isfinite(result["numerical"]["relative_l2"])
        and result["numerical"]["relative_l2"] <= 1e-5
        and result["numerical"]["max_abs"] <= 1e-3
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--runtime-identity", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--visible-physical-gpu", required=True, type=int)
    parser.add_argument("--query-length", type=int, default=16)
    parser.add_argument("--key-length", type=int, default=32)
    parser.add_argument("--query-heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--head-size", type=int, default=256)
    parser.add_argument("--layout", choices=("bshd",), default="bshd")
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction)
    parser.set_defaults(causal=True)
    parser.add_argument("--input-scale", type=float, default=0.1)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            result,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "qualified": result["qualified"],
                "shape": result["shape"],
                "timing": result["timing"],
                "numerical": result["numerical"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
