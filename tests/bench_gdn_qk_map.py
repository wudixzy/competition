#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch


def load_extension(path: Path):
    spec = importlib.util.spec_from_file_location("corex_gdn_qk_map", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def l2norm(value: torch.Tensor) -> torch.Tensor:
    return value * torch.rsqrt(
        (value * value).sum(dim=-1, keepdim=True) + 1e-6)


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def measure(case, warmup: int, iterations: int, repeats: int) -> dict:
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


def differences(actual: torch.Tensor,
                expected: torch.Tensor) -> dict[str, object]:
    delta = (actual - expected).abs()
    return {
        "exact": bool(torch.equal(actual, expected)),
        "max_abs": float(delta.max()),
        "finite": bool(torch.isfinite(actual).all()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--random-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    extension = load_extension(args.extension)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    batch, key_heads, value_heads, dim = 1, 4, 8, 128
    ratio = value_heads // key_heads
    scale = dim ** -0.5
    mixed = torch.randn(
        (batch, 1, 2048), device=device, generator=generator,
        dtype=torch.float16)
    raw_q, raw_k, raw_value = torch.split(mixed, [512, 512, 1024], dim=-1)
    raw_q = raw_q.reshape(batch, key_heads, dim)
    raw_k = raw_k.reshape(batch, key_heads, dim)
    value = raw_value.reshape(batch, value_heads, dim).float()
    decay = torch.full(
        (batch, value_heads), 0.98, device=device)
    beta = torch.full(
        (batch, value_heads), 0.5, device=device)
    initial_state = torch.randn(
        (batch, value_heads, dim, dim), device=device,
        generator=generator) * 0.01

    def baseline_prep(q: torch.Tensor, k: torch.Tensor):
        q = q.repeat_interleave(ratio, dim=1)
        k = k.repeat_interleave(ratio, dim=1)
        return l2norm(q).float() * scale, l2norm(k).float()

    def candidate_prep(q: torch.Tensor, k: torch.Tensor):
        return extension.qk_map(l2norm(q), l2norm(k), value_heads).unbind(0)

    def recurrent(state: torch.Tensor, q_t: torch.Tensor,
                  k_t: torch.Tensor, current_value: torch.Tensor,
                  current_decay: torch.Tensor,
                  current_beta: torch.Tensor) -> torch.Tensor:
        state.mul_(current_decay[:, :, None, None])
        flat = state.view(-1, dim, dim)
        bh = flat.shape[0]
        memory = torch.bmm(k_t.view(bh, 1, dim), flat).view(
            batch, value_heads, dim)
        delta = (current_value - memory) * current_beta[:, :, None]
        flat.baddbmm_(k_t.view(bh, dim, 1), delta.view(bh, 1, dim))
        return torch.bmm(q_t.view(bh, 1, dim), flat).view(
            batch, value_heads, dim)

    expected_q, expected_k = baseline_prep(raw_q, raw_k)
    actual_q, actual_k = candidate_prep(raw_q, raw_k)
    expected_state = initial_state.clone()
    actual_state = initial_state.clone()
    expected_output = recurrent(
        expected_state, expected_q, expected_k, value, decay, beta)
    actual_output = recurrent(
        actual_state, actual_q, actual_k, value, decay, beta)
    one_step = {
        "query": differences(actual_q, expected_q),
        "key": differences(actual_k, expected_k),
        "output": differences(actual_output, expected_output),
        "state": differences(actual_state, expected_state),
    }

    sequence_expected_state = initial_state.clone()
    sequence_actual_state = initial_state.clone()
    exact_steps = 0
    max_output_abs = 0.0
    max_state_abs = 0.0
    for _ in range(args.random_steps):
        step_q = torch.randn(
            raw_q.shape, device=device, generator=generator,
            dtype=torch.float16)
        step_k = torch.randn(
            raw_k.shape, device=device, generator=generator,
            dtype=torch.float16)
        step_v = torch.randn(
            value.shape, device=device, generator=generator)
        step_decay = 0.95 + 0.049 * torch.rand(
            decay.shape, device=device, generator=generator)
        step_beta = torch.rand(
            beta.shape, device=device, generator=generator)
        step_expected_q, step_expected_k = baseline_prep(step_q, step_k)
        step_actual_q, step_actual_k = candidate_prep(step_q, step_k)
        step_expected = recurrent(
            sequence_expected_state, step_expected_q, step_expected_k,
            step_v, step_decay, step_beta)
        step_actual = recurrent(
            sequence_actual_state, step_actual_q, step_actual_k,
            step_v, step_decay, step_beta)
        exact_steps += int(
            torch.equal(step_actual_q, step_expected_q)
            and torch.equal(step_actual_k, step_expected_k)
            and torch.equal(step_actual, step_expected)
            and torch.equal(sequence_actual_state, sequence_expected_state))
        max_output_abs = max(max_output_abs, float(
            (step_actual - step_expected).abs().max()))
        max_state_abs = max(max_state_abs, float(
            (sequence_actual_state - sequence_expected_state).abs().max()))

    baseline_timing_state = initial_state.clone()
    candidate_timing_state = initial_state.clone()

    def baseline_full():
        q_t, k_t = baseline_prep(raw_q, raw_k)
        return recurrent(
            baseline_timing_state, q_t, k_t, value, decay, beta)

    def candidate_full():
        q_t, k_t = candidate_prep(raw_q, raw_k)
        return recurrent(
            candidate_timing_state, q_t, k_t, value, decay, beta)

    timings = {
        "baseline_prep": measure(
            lambda: baseline_prep(raw_q, raw_k)[0],
            args.warmup, args.iterations, args.repeats),
        "candidate_prep": measure(
            lambda: candidate_prep(raw_q, raw_k)[0],
            args.warmup, args.iterations, args.repeats),
        "baseline_full": measure(
            baseline_full, args.warmup, args.iterations, args.repeats),
        "candidate_full": measure(
            candidate_full, args.warmup, args.iterations, args.repeats),
    }
    timings["candidate_prep"]["speedup_vs_baseline"] = (
        timings["baseline_prep"]["median_ms"]
        / timings["candidate_prep"]["median_ms"])
    timings["candidate_full"]["speedup_vs_baseline"] = (
        timings["baseline_full"]["median_ms"]
        / timings["candidate_full"]["median_ms"])

    report = {
        "device": torch.cuda.get_device_name(device),
        "shape": {
            "batch": batch, "key_heads": key_heads,
            "value_heads": value_heads, "head_dim": dim,
        },
        "layout": {
            "query_contiguous": raw_q.is_contiguous(),
            "key_contiguous": raw_k.is_contiguous(),
        },
        "config": {
            "warmup": args.warmup, "iterations": args.iterations,
            "repeats": args.repeats, "random_steps": args.random_steps,
            "seed": args.seed,
        },
        "one_step": one_step,
        "random": {
            "steps": args.random_steps, "exact_steps": exact_steps,
            "max_output_abs": max_output_abs,
            "max_state_abs": max_state_abs,
        },
        "timings": timings,
    }
    report["ok"] = bool(
        all(check["exact"] for check in one_step.values())
        and exact_steps == args.random_steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
