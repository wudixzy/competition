#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch


Case = Callable[[], torch.Tensor]


def load_extension(path: Path):
    spec = importlib.util.spec_from_file_location("corex_gdn_recurrent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def l2norm(value: torch.Tensor) -> torch.Tensor:
    return value * torch.rsqrt((value * value).sum(-1, keepdim=True) + 1e-6)


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def measure(case: Case, warmup: int, iterations: int,
            repeats: int) -> list[float]:
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
    return trials


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--sustained-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    extension = load_extension(args.extension)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    batch, heads, dim = 1, 12, 128

    raw_q = torch.randn((batch, heads, dim), device=device,
                        generator=generator, dtype=torch.float16)
    raw_k = torch.randn((batch, heads, dim), device=device,
                        generator=generator, dtype=torch.float16)
    query = l2norm(raw_q).float() * (dim ** -0.5)
    key = l2norm(raw_k).float()
    value = torch.randn((batch, heads, dim), device=device,
                        generator=generator, dtype=torch.float16).float()
    decay = torch.full((batch, heads), 0.98, device=device)
    beta = torch.full((batch, heads), 0.5, device=device)
    initial_state = torch.randn(
        (batch, heads, dim, dim), device=device, generator=generator) * 0.01

    def reference(state: torch.Tensor) -> torch.Tensor:
        state.mul_(decay[:, :, None, None])
        flat = state.view(-1, dim, dim)
        bh = flat.shape[0]
        memory = torch.bmm(key.view(bh, 1, dim), flat).view(
            batch, heads, dim)
        delta = (value - memory) * beta[:, :, None]
        flat.baddbmm_(key.view(bh, dim, 1), delta.view(bh, 1, dim))
        return torch.bmm(query.view(bh, 1, dim), flat).view(
            batch, heads, dim)

    reference_state = initial_state.clone()
    reference_output = reference(reference_state)
    candidate_state = initial_state.clone()
    candidate_output = extension.recurrent_update(
        candidate_state, query, key, value, decay, beta)
    torch.cuda.synchronize()

    one_step = {
        "output_max_abs": float(
            (candidate_output - reference_output).abs().max()),
        "state_max_abs": float((candidate_state - reference_state).abs().max()),
        "output_exact": bool(torch.equal(candidate_output, reference_output)),
        "state_exact": bool(torch.equal(candidate_state, reference_state)),
        "finite": bool(torch.isfinite(candidate_output).all()
                       and torch.isfinite(candidate_state).all()),
    }

    sustained_reference_state = initial_state.clone()
    sustained_candidate_state = initial_state.clone()
    for _ in range(args.sustained_steps):
        sustained_reference_output = reference(sustained_reference_state)
        sustained_candidate_output = extension.recurrent_update(
            sustained_candidate_state, query, key, value, decay, beta)
    torch.cuda.synchronize()
    sustained = {
        "steps": args.sustained_steps,
        "output_max_abs": float((sustained_candidate_output
                                 - sustained_reference_output).abs().max()),
        "state_max_abs": float((sustained_candidate_state
                                - sustained_reference_state).abs().max()),
        "finite": bool(torch.isfinite(sustained_candidate_output).all()
                       and torch.isfinite(sustained_candidate_state).all()),
    }

    reference_timing_state = initial_state.clone()
    candidate_timing_state = initial_state.clone()

    def reference_case() -> torch.Tensor:
        return reference(reference_timing_state)

    def candidate_case() -> torch.Tensor:
        return extension.recurrent_update(
            candidate_timing_state, query, key, value, decay, beta)

    results = {}
    for name, case in {
            "reference": reference_case,
            "candidate": candidate_case}.items():
        trials = measure(case, args.warmup, args.iterations, args.repeats)
        results[name] = {
            "median_ms": statistics.median(trials),
            "p10_ms": percentile(trials, 10),
            "p90_ms": percentile(trials, 90),
            "trials_ms": trials,
        }
    baseline = results["reference"]["median_ms"]
    for result in results.values():
        result["speedup_vs_reference"] = baseline / result["median_ms"]

    report = {
        "device": torch.cuda.get_device_name(device),
        "config": vars(args) | {
            "extension": str(args.extension), "out": str(args.out)},
        "one_step": one_step,
        "sustained": sustained,
        "results": results,
    }
    report["ok"] = bool(one_step["finite"] and sustained["finite"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
