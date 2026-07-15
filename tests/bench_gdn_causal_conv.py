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
import torch.nn.functional as F


Case = Callable[[], torch.Tensor]


def load_extension(path: Path):
    spec = importlib.util.spec_from_file_location("corex_gdn_causal_conv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(hidden: torch.Tensor, state: torch.Tensor,
              weight: torch.Tensor) -> torch.Tensor:
    combined = torch.cat([state, hidden], dim=-1).to(weight.dtype)
    state.copy_(combined[:, :, -3:])
    output = F.conv1d(
        combined, weight.unsqueeze(1), padding=0, groups=weight.shape[0])
    return F.silu(output[:, :, -hidden.shape[-1]:]).to(hidden.dtype)


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


def differences(candidate: torch.Tensor,
                baseline: torch.Tensor) -> dict[str, object]:
    delta = (candidate - baseline).abs()
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.float().mean()),
        "exact": bool(torch.equal(candidate, baseline)),
        "close": bool(torch.allclose(candidate, baseline,
                                     rtol=1e-3, atol=1e-3)),
        "finite": bool(torch.isfinite(candidate).all()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--sequence-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    extension = load_extension(args.extension)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    # Checkpoint TP4 rank shape: (2048 q + 2048 k + 4096 v) / 4.
    batch, channels, state_len = 1, 2048, 3

    hidden = torch.randn(
        (batch, channels, 1), device=device, generator=generator,
        dtype=torch.float16)
    weight = (torch.randn(
        (channels, state_len + 1), device=device, generator=generator,
        dtype=torch.float16) * 0.05).contiguous()
    initial_state = torch.randn(
        (batch, channels, state_len), device=device, generator=generator,
        dtype=torch.float16).float()

    reference_state = initial_state.clone()
    reference_output = reference(hidden, reference_state, weight)
    candidate_state = initial_state.clone()
    candidate_output = extension.causal_conv_update(
        candidate_state, hidden, weight)
    torch.cuda.synchronize()
    one_step = {
        "output": differences(candidate_output, reference_output),
        "state": differences(candidate_state, reference_state),
    }

    sequence_hidden = torch.randn(
        (args.sequence_steps, batch, channels, 1), device=device,
        generator=generator, dtype=torch.float16)
    sequence_reference_state = initial_state.clone()
    sequence_candidate_state = initial_state.clone()
    for step in range(args.sequence_steps):
        sequence_reference_output = reference(
            sequence_hidden[step], sequence_reference_state, weight)
        sequence_candidate_output = extension.causal_conv_update(
            sequence_candidate_state, sequence_hidden[step], weight)
    torch.cuda.synchronize()
    random_sequence = {
        "steps": args.sequence_steps,
        "output": differences(
            sequence_candidate_output, sequence_reference_output),
        "state": differences(
            sequence_candidate_state, sequence_reference_state),
    }

    reference_timing_state = initial_state.clone()
    candidate_timing_state = initial_state.clone()

    def reference_case() -> torch.Tensor:
        return reference(hidden, reference_timing_state, weight)

    def candidate_case() -> torch.Tensor:
        return extension.causal_conv_update(
            candidate_timing_state, hidden, weight)

    results = {}
    for name, case in {"reference": reference_case,
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
        "shape": {
            "batch": batch, "channels": channels,
            "state_len": state_len, "input_dtype": "float16",
            "state_dtype": "float32"},
        "config": {
            "warmup": args.warmup, "iterations": args.iterations,
            "repeats": args.repeats, "sequence_steps": args.sequence_steps,
            "seed": args.seed},
        "one_step": one_step,
        "random_sequence": random_sequence,
        "results": results,
    }
    report["ok"] = bool(
        one_step["output"]["close"] and one_step["state"]["exact"]
        and random_sequence["output"]["close"]
        and random_sequence["state"]["exact"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
