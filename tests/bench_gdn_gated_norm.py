#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from ixformer.functions.rms_norm import rms_norm as ixformer_rms_norm
from vllm.model_executor.layers.linear import ReplicatedLinear


Case = Callable[[], torch.Tensor]


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def measure(case: Case, warmup: int, iterations: int,
            repeats: int) -> dict[str, object]:
    for _ in range(warmup):
        case()
    torch.cuda.synchronize()
    trials = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(iterations):
            case()
        torch.cuda.synchronize()
        trials.append((time.perf_counter() - started) * 1000.0 / iterations)
    return {
        "median_ms": statistics.median(trials),
        "p10_ms": percentile(trials, 10),
        "p90_ms": percentile(trials, 90),
        "trials_ms": trials,
    }


def difference(actual: torch.Tensor,
               expected: torch.Tensor) -> dict[str, object]:
    delta = (actual.float() - expected.float()).abs()
    return {
        "exact": bool(torch.equal(actual, expected)),
        "close": bool(torch.allclose(actual, expected,
                                     rtol=1e-4, atol=1e-5)),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "dtype": str(actual.dtype),
        "shape": list(actual.shape),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    rows, head_dim, local_value_dim, hidden = 8, 128, 1024, 2048
    epsilon = 1e-6
    runtime_dtype = torch.float16

    core = torch.randn(
        (rows, head_dim), device=device, generator=generator,
        dtype=torch.float32)
    gate = torch.randn(
        (rows, head_dim), device=device, generator=generator,
        dtype=runtime_dtype)
    weight = torch.randn(
        (head_dim,), device=device, generator=generator,
        dtype=runtime_dtype)
    output_weight = torch.randn(
        (hidden, local_value_dim), device=device, generator=generator,
        dtype=runtime_dtype) * 0.02
    out_proj = ReplicatedLinear(
        local_value_dim, hidden, bias=False, params_dtype=runtime_dtype,
        quant_config=None, prefix="probe.gdn_out_proj").to(device)
    out_proj.weight_loader(out_proj.weight, output_weight)

    def reference_norm() -> torch.Tensor:
        hs = core.float()
        variance = hs.pow(2).mean(-1, keepdim=True)
        hs = hs * torch.rsqrt(variance + epsilon)
        hs = weight * hs
        return (hs * F.silu(gate.float())).to(runtime_dtype)

    def ixformer_fp32_norm() -> torch.Tensor:
        normalized = ixformer_rms_norm(
            core, weight.float(), output=None, eps=epsilon)
        return (normalized * F.silu(gate.float())).to(runtime_dtype)

    def ixformer_fp16_norm() -> torch.Tensor:
        normalized = ixformer_rms_norm(
            core.to(runtime_dtype), weight, output=None, eps=epsilon)
        return (normalized.float() * F.silu(gate.float())).to(runtime_dtype)

    variants: dict[str, Case] = {
        "reference": reference_norm,
        "ixformer_fp32": ixformer_fp32_norm,
        "ixformer_fp16": ixformer_fp16_norm,
    }
    reference_output = reference_norm()
    reference_tail, _ = out_proj(reference_output.reshape(1, -1))
    report_variants: dict[str, object] = {}
    valid_variants: dict[str, Case] = {"reference": reference_norm}
    for name, case in variants.items():
        try:
            output = case()
            tail, _ = out_proj(output.reshape(1, -1))
            torch.cuda.synchronize()
            report_variants[name] = {
                "norm": difference(output, reference_output),
                "tail": difference(tail, reference_tail),
            }
            valid_variants[name] = case
        except Exception as exc:  # noqa: BLE001 - capability probe.
            report_variants[name] = {"error": repr(exc)}

    timings: dict[str, object] = {}
    for name, norm_case in valid_variants.items():
        def tail_case(current: Case = norm_case) -> torch.Tensor:
            output, _ = out_proj(current().reshape(1, -1))
            return output

        timings[f"{name}_norm"] = measure(
            norm_case, args.warmup, args.iterations, args.repeats)
        timings[f"{name}_tail"] = measure(
            tail_case, args.warmup, args.iterations, args.repeats)

    baseline_norm = timings["reference_norm"]["median_ms"]
    baseline_tail = timings["reference_tail"]["median_ms"]
    for name in valid_variants:
        timings[f"{name}_norm"]["speedup_vs_reference"] = (
            baseline_norm / timings[f"{name}_norm"]["median_ms"])
        timings[f"{name}_tail"]["speedup_vs_reference"] = (
            baseline_tail / timings[f"{name}_tail"]["median_ms"])

    report = {
        "device": torch.cuda.get_device_name(device),
        "config": {
            "rows": rows, "head_dim": head_dim,
            "local_value_dim": local_value_dim, "hidden": hidden,
            "runtime_dtype": str(runtime_dtype), "epsilon": epsilon,
            "warmup": args.warmup, "iterations": args.iterations,
            "repeats": args.repeats, "seed": args.seed},
        "variants": report_variants,
        "timings": timings,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
