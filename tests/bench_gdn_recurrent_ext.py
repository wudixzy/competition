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
    batch, key_heads, heads, dim = 1, 4, 12, 128

    raw_q = torch.randn((batch, key_heads, dim), device=device,
                        generator=generator, dtype=torch.float16)
    raw_k = torch.randn((batch, key_heads, dim), device=device,
                        generator=generator, dtype=torch.float16)
    value = torch.randn((batch, heads, dim), device=device,
                        generator=generator, dtype=torch.float16).float()
    decay = torch.full((batch, heads), 0.98, device=device)
    beta = torch.full((batch, heads), 0.5, device=device)
    initial_state = torch.randn(
        (batch, heads, dim, dim), device=device, generator=generator) * 0.01
    candidate_output_workspace = torch.empty_like(value)

    def prepare(current_raw_q: torch.Tensor,
                current_raw_k: torch.Tensor) -> tuple[torch.Tensor,
                                                       torch.Tensor]:
        ratio = heads // key_heads
        current_query = current_raw_q.repeat_interleave(ratio, dim=1)
        current_key = current_raw_k.repeat_interleave(ratio, dim=1)
        return (l2norm(current_query).float() * (dim ** -0.5),
                l2norm(current_key).float())

    def candidate(state: torch.Tensor, current_raw_q: torch.Tensor,
                  current_raw_k: torch.Tensor, current_value: torch.Tensor,
                  current_decay: torch.Tensor,
                  current_beta: torch.Tensor) -> torch.Tensor:
        extension.prep_recurrent_update_out(
            state, current_raw_q, current_raw_k, current_value,
            current_decay, current_beta, candidate_output_workspace)
        return candidate_output_workspace

    def reference(state: torch.Tensor, current_raw_q: torch.Tensor,
                  current_raw_k: torch.Tensor, current_value: torch.Tensor,
                  current_decay: torch.Tensor,
                  current_beta: torch.Tensor) -> torch.Tensor:
        current_query, current_key = prepare(current_raw_q, current_raw_k)
        state.mul_(current_decay[:, :, None, None])
        flat = state.view(-1, dim, dim)
        bh = flat.shape[0]
        memory = torch.bmm(current_key.view(bh, 1, dim), flat).view(
            batch, heads, dim)
        delta = (current_value - memory) * current_beta[:, :, None]
        flat.baddbmm_(
            current_key.view(bh, dim, 1), delta.view(bh, 1, dim))
        return torch.bmm(current_query.view(bh, 1, dim), flat).view(
            batch, heads, dim)

    reference_state = initial_state.clone()
    reference_output = reference(
        reference_state, raw_q, raw_k, value, decay, beta)
    candidate_state = initial_state.clone()
    candidate_output = candidate(
        candidate_state, raw_q, raw_k, value, decay, beta)
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
        sustained_reference_output = reference(
            sustained_reference_state, raw_q, raw_k, value, decay, beta)
        sustained_candidate_output = candidate(
            sustained_candidate_state, raw_q, raw_k, value, decay, beta)
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

    sequence_q = torch.randn(
        (args.sustained_steps, batch, key_heads, dim), device=device,
        generator=generator, dtype=torch.float16)
    sequence_k = torch.randn(
        (args.sustained_steps, batch, key_heads, dim), device=device,
        generator=generator, dtype=torch.float16)
    sequence_v = torch.randn(
        (args.sustained_steps, batch, heads, dim), device=device,
        generator=generator, dtype=torch.float16).float()
    sequence_decay = 0.95 + 0.049 * torch.rand(
        (args.sustained_steps, batch, heads), device=device,
        generator=generator)
    sequence_beta = torch.sigmoid(torch.randn(
        (args.sustained_steps, batch, heads), device=device,
        generator=generator))
    sequence_reference_state = initial_state.clone()
    sequence_candidate_state = initial_state.clone()
    for step in range(args.sustained_steps):
        step_query = sequence_q[step]
        step_key = sequence_k[step]
        step_value = sequence_v[step]
        step_decay = sequence_decay[step]
        step_beta = sequence_beta[step]

        sequence_reference_output = reference(
            sequence_reference_state, step_query, step_key, step_value,
            step_decay, step_beta)

        sequence_candidate_output = candidate(
            sequence_candidate_state, step_query, step_key, step_value,
            step_decay, step_beta)
    torch.cuda.synchronize()
    sequence_output_diff = (
        sequence_candidate_output - sequence_reference_output).abs()
    sequence_state_diff = (
        sequence_candidate_state - sequence_reference_state).abs()
    random_sequence = {
        "steps": args.sustained_steps,
        "output_max_abs": float(sequence_output_diff.max()),
        "output_mean_abs": float(sequence_output_diff.mean()),
        "state_max_abs": float(sequence_state_diff.max()),
        "state_mean_abs": float(sequence_state_diff.mean()),
        "output_close": bool(torch.allclose(
            sequence_candidate_output, sequence_reference_output,
            rtol=1e-4, atol=1e-5)),
        "state_close": bool(torch.allclose(
            sequence_candidate_state, sequence_reference_state,
            rtol=1e-4, atol=1e-5)),
        "finite": bool(torch.isfinite(sequence_candidate_output).all()
                       and torch.isfinite(sequence_candidate_state).all()),
    }

    reference_timing_state = initial_state.clone()
    candidate_timing_state = initial_state.clone()

    def reference_case() -> torch.Tensor:
        return reference(
            reference_timing_state, raw_q, raw_k, value, decay, beta)

    def candidate_case() -> torch.Tensor:
        return candidate(
            candidate_timing_state, raw_q, raw_k, value, decay, beta)

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
        "random_sequence": random_sequence,
        "results": results,
    }
    report["ok"] = bool(
        one_step["finite"] and sustained["finite"]
        and random_sequence["finite"]
        and random_sequence["output_close"]
        and random_sequence["state_close"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
