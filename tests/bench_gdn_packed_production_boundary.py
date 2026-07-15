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


def load_extension(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
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
        trials.append((time.perf_counter() - started) * 1000 / iterations)
    return trials


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed-extension", type=Path, required=True)
    parser.add_argument("--beta-decay-extension", type=Path, required=True)
    parser.add_argument("--qk-map-extension", type=Path, required=True)
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
    packed = load_extension(
        "corex_gdn_packed_decode", args.packed_extension)
    beta_decay = load_extension(
        "corex_gdn_beta_decay", args.beta_decay_extension)
    qk_map = load_extension("corex_gdn_qk_map", args.qk_map_extension)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    batch, key_heads, value_heads, dim = 1, 4, 8, 128
    mixed_dim = (2 * key_heads + value_heads) * dim

    mixed = torch.randn(
        (batch, mixed_dim), device=device, generator=generator,
        dtype=torch.float16)
    beta_raw = torch.randn(
        (batch, value_heads), device=device, generator=generator,
        dtype=torch.float16)
    decay_raw = torch.randn(
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

    def current(state: torch.Tensor, current_mixed: torch.Tensor,
                current_beta: torch.Tensor,
                current_decay: torch.Tensor) -> torch.Tensor:
        query = current_mixed[:, :key_heads * dim].view(
            batch, key_heads, dim)
        key = current_mixed[:, key_heads * dim:2 * key_heads * dim].view(
            batch, key_heads, dim)
        value = current_mixed[:, 2 * key_heads * dim:].view(
            batch, value_heads, dim).float()
        mapped = qk_map.qk_map(
            l2norm(query), l2norm(key), value_heads)
        query, key = mapped[0], mapped[1]
        prepared = beta_decay.beta_decay(
            current_beta, current_decay, a_log, dt_bias)
        beta, decay = prepared[0], prepared[1]
        state.mul_(decay[:, :, None, None])
        flat = state.view(-1, dim, dim)
        bh = flat.shape[0]
        memory = torch.bmm(key.view(bh, 1, dim), flat).view(
            batch, value_heads, dim)
        delta = (value - memory) * beta[:, :, None]
        flat.baddbmm_(
            key.view(bh, dim, 1), delta.view(bh, 1, dim))
        return torch.bmm(query.view(bh, 1, dim), flat).view(
            batch, value_heads, dim)

    def candidate(state: torch.Tensor, current_mixed: torch.Tensor,
                  current_beta: torch.Tensor,
                  current_decay: torch.Tensor) -> torch.Tensor:
        return packed.packed_decode(
            state, current_mixed, current_beta, current_decay,
            a_log, dt_bias)

    sequence_mixed = torch.randn(
        (args.sequence_steps, batch, mixed_dim), device=device,
        generator=generator, dtype=torch.float16)
    sequence_beta = torch.randn(
        (args.sequence_steps, batch, value_heads), device=device,
        generator=generator, dtype=torch.float16)
    sequence_decay = torch.randn(
        (args.sequence_steps, batch, value_heads), device=device,
        generator=generator, dtype=torch.float16)
    current_state = initial_state.clone()
    candidate_state = initial_state.clone()
    output_max_abs = 0.0
    output_abs_sum = 0.0
    output_elements = 0
    finite_steps = 0
    for step in range(args.sequence_steps):
        expected = current(
            current_state, sequence_mixed[step], sequence_beta[step],
            sequence_decay[step])
        actual = candidate(
            candidate_state, sequence_mixed[step], sequence_beta[step],
            sequence_decay[step])
        difference = (actual - expected).abs()
        output_max_abs = max(output_max_abs, float(difference.max()))
        output_abs_sum += float(difference.sum())
        output_elements += difference.numel()
        if (torch.isfinite(actual).all()
                and torch.isfinite(candidate_state).all()):
            finite_steps += 1
    torch.cuda.synchronize()
    state_difference = (candidate_state - current_state).abs()

    current_timing_state = initial_state.clone()
    candidate_timing_state = initial_state.clone()

    def reset_current() -> None:
        current_timing_state.copy_(initial_state)

    def reset_candidate() -> None:
        candidate_timing_state.copy_(initial_state)

    current_trials = measure(
        lambda: current(current_timing_state, mixed, beta_raw, decay_raw),
        reset_current, args.warmup, args.iterations, args.repeats)
    candidate_trials = measure(
        lambda: candidate(
            candidate_timing_state, mixed, beta_raw, decay_raw),
        reset_candidate, args.warmup, args.iterations, args.repeats)
    current_median = statistics.median(current_trials)
    candidate_median = statistics.median(candidate_trials)
    report = {
        "device": torch.cuda.get_device_name(device),
        "config": vars(args) | {"shape": [1, 4, 8, 128]},
        "sequence": {
            "steps": args.sequence_steps,
            "finite_steps": finite_steps,
            "output_max_abs": output_max_abs,
            "output_mean_abs": output_abs_sum / output_elements,
            "state_max_abs": float(state_difference.max()),
            "state_mean_abs": float(state_difference.mean()),
        },
        "performance": {
            "current_median_ms": current_median,
            "candidate_median_ms": candidate_median,
            "speedup": current_median / candidate_median,
            "current_trials_ms": current_trials,
            "candidate_trials_ms": candidate_trials,
        },
    }
    report["ok"] = bool(
        finite_steps == args.sequence_steps
        and candidate_median <= 0.110
        and current_median / candidate_median >= 1.5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
