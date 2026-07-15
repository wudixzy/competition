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
    spec = importlib.util.spec_from_file_location(
        "corex_gdn_packed_decode", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def l2norm(value: torch.Tensor) -> torch.Tensor:
    return value * torch.rsqrt(
        (value * value).sum(-1, keepdim=True) + 1e-6)


def measure(case: Case, reset: Callable[[], None], warmup: int,
            iterations: int, repeats: int) -> list[float]:
    reset()
    for _ in range(warmup):
        case()
    torch.cuda.synchronize()
    trials = []
    for _ in range(repeats):
        reset()
        torch.cuda.synchronize()
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
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--sequence-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    extension = load_extension(args.extension)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    batch, key_heads, value_heads, dim = 1, 4, 8, 128
    mixed_dim = (2 * key_heads + value_heads) * dim

    mixed_qkv = torch.randn(
        (batch, mixed_dim), device=device, generator=generator,
        dtype=torch.float16)
    beta_input = torch.randn(
        (batch, value_heads), device=device, generator=generator,
        dtype=torch.float16)
    decay_input = torch.randn(
        (batch, value_heads), device=device, generator=generator,
        dtype=torch.float16)
    a_log = torch.randn(
        (value_heads,), device=device, generator=generator,
        dtype=torch.float16) * 0.1
    dt_bias = torch.randn(
        (value_heads,), device=device, generator=generator,
        dtype=torch.float16) * 0.1
    initial_state = torch.randn(
        (batch, value_heads, dim, dim), device=device,
        generator=generator, dtype=torch.float32) * 0.01

    def reference(state: torch.Tensor, mixed: torch.Tensor,
                  beta_raw: torch.Tensor,
                  decay_raw: torch.Tensor) -> torch.Tensor:
        query = mixed[:, :key_heads * dim].view(batch, key_heads, dim)
        key = mixed[:, key_heads * dim:2 * key_heads * dim].view(
            batch, key_heads, dim)
        value = mixed[:, 2 * key_heads * dim:].view(
            batch, value_heads, dim).float()
        query = l2norm(query).repeat_interleave(2, dim=1).float()
        query.mul_(dim ** -0.5)
        key = l2norm(key).repeat_interleave(2, dim=1).float()
        beta = torch.sigmoid(beta_raw.float()).half().float()
        decay = torch.exp(
            -torch.exp(a_log.float())
            * F.softplus(decay_raw.float() + dt_bias.float()))

        state.mul_(decay[:, :, None, None])
        flat = state.view(-1, dim, dim)
        batch_heads = flat.shape[0]
        memory = torch.bmm(
            key.view(batch_heads, 1, dim), flat).view(
                batch, value_heads, dim)
        delta = (value - memory) * beta[:, :, None]
        flat.baddbmm_(
            key.view(batch_heads, dim, 1),
            delta.view(batch_heads, 1, dim))
        return torch.bmm(
            query.view(batch_heads, 1, dim), flat).view(
                batch, value_heads, dim)

    def candidate(state: torch.Tensor, mixed: torch.Tensor,
                  beta_raw: torch.Tensor,
                  decay_raw: torch.Tensor) -> torch.Tensor:
        return extension.packed_decode(
            state, mixed, beta_raw, decay_raw, a_log, dt_bias)

    reference_state = initial_state.clone()
    candidate_state = initial_state.clone()
    reference_output = reference(
        reference_state, mixed_qkv, beta_input, decay_input)
    candidate_output = candidate(
        candidate_state, mixed_qkv, beta_input, decay_input)
    torch.cuda.synchronize()
    one_step = {
        "output_max_abs": float(
            (candidate_output - reference_output).abs().max()),
        "output_mean_abs": float(
            (candidate_output - reference_output).abs().mean()),
        "state_max_abs": float(
            (candidate_state - reference_state).abs().max()),
        "state_mean_abs": float(
            (candidate_state - reference_state).abs().mean()),
        "finite": bool(
            torch.isfinite(candidate_output).all()
            and torch.isfinite(candidate_state).all()),
    }

    sequence_mixed = torch.randn(
        (args.sequence_steps, batch, mixed_dim), device=device,
        generator=generator, dtype=torch.float16)
    sequence_beta = torch.randn(
        (args.sequence_steps, batch, value_heads), device=device,
        generator=generator, dtype=torch.float16)
    sequence_decay = torch.randn(
        (args.sequence_steps, batch, value_heads), device=device,
        generator=generator, dtype=torch.float16)
    sequence_reference_state = initial_state.clone()
    sequence_candidate_state = initial_state.clone()
    output_max_abs = 0.0
    output_abs_sum = 0.0
    output_elements = 0
    finite_steps = 0
    for step in range(args.sequence_steps):
        reference_output = reference(
            sequence_reference_state, sequence_mixed[step],
            sequence_beta[step], sequence_decay[step])
        candidate_output = candidate(
            sequence_candidate_state, sequence_mixed[step],
            sequence_beta[step], sequence_decay[step])
        difference = (candidate_output - reference_output).abs()
        output_max_abs = max(output_max_abs, float(difference.max()))
        output_abs_sum += float(difference.sum())
        output_elements += difference.numel()
        if (torch.isfinite(candidate_output).all()
                and torch.isfinite(sequence_candidate_state).all()):
            finite_steps += 1
    torch.cuda.synchronize()
    state_difference = (
        sequence_candidate_state - sequence_reference_state).abs()
    random_sequence = {
        "steps": args.sequence_steps,
        "finite_steps": finite_steps,
        "output_max_abs": output_max_abs,
        "output_mean_abs": output_abs_sum / output_elements,
        "state_max_abs": float(state_difference.max()),
        "state_mean_abs": float(state_difference.mean()),
        "finite": finite_steps == args.sequence_steps,
    }

    reference_timing_state = initial_state.clone()
    candidate_timing_state = initial_state.clone()

    def reset_reference() -> None:
        reference_timing_state.copy_(initial_state)

    def reset_candidate() -> None:
        candidate_timing_state.copy_(initial_state)

    def reference_case() -> torch.Tensor:
        return reference(reference_timing_state, mixed_qkv,
                         beta_input, decay_input)

    def candidate_case() -> torch.Tensor:
        return candidate(candidate_timing_state, mixed_qkv,
                         beta_input, decay_input)

    reference_trials = measure(
        reference_case, reset_reference, args.warmup,
        args.iterations, args.repeats)
    candidate_trials = measure(
        candidate_case, reset_candidate, args.warmup,
        args.iterations, args.repeats)
    reference_median = statistics.median(reference_trials)
    candidate_median = statistics.median(candidate_trials)
    performance = {
        "reference_median_ms": reference_median,
        "candidate_median_ms": candidate_median,
        "speedup": reference_median / candidate_median,
        "reference_trials_ms": reference_trials,
        "candidate_trials_ms": candidate_trials,
    }

    report = {
        "device": torch.cuda.get_device_name(device),
        "config": {
            "extension": str(args.extension),
            "device": args.device,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "sequence_steps": args.sequence_steps,
            "seed": args.seed,
            "shape": [batch, key_heads, value_heads, dim],
        },
        "one_step": one_step,
        "random_sequence": random_sequence,
        "performance": performance,
    }
    report["ok"] = bool(one_step["finite"] and random_sequence["finite"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
