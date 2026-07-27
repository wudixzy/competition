#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F


FIXED_SEEDS = (20260716, 20260727)


def load_extension(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    delta = (actual.float() - expected.float()).abs()
    squared_error = float(delta.square().sum())
    squared_reference = float(expected.float().square().sum())
    return {
        "exact": bool(torch.equal(actual, expected)),
        "finite": bool(torch.isfinite(actual).all()),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "relative_l2": math.sqrt(
            squared_error / max(squared_reference, 1.0e-30)),
    }


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
    ordered = sorted(trials)
    return {
        "median_ms": statistics.median(trials),
        "p10_ms": ordered[max(0, int(0.1 * (len(ordered) - 1)))],
        "p90_ms": ordered[min(
            len(ordered) - 1, int(0.9 * (len(ordered) - 1)))],
        "trials_ms": trials,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-extension", type=Path, required=True)
    parser.add_argument("--direct-extension", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--sequence-steps", type=int, default=500)
    parser.add_argument("--sequence-seed", type=int, default=20260728)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    candidate = load_extension(
        "corex_moe_pairwise_w13", args.candidate_extension)
    direct = load_extension(
        "corex_moe_direct_routed", args.direct_extension)

    experts, top_k, hidden, rows = 256, 8, 2048, 256
    dtype = torch.float16
    fixed = {}
    timing_state = None
    for seed in FIXED_SEEDS:
        generator = torch.Generator(device=device).manual_seed(seed)
        states = torch.randn(
            (1, hidden), device=device, dtype=dtype, generator=generator)
        logits = torch.randn(
            (1, experts), device=device, dtype=dtype, generator=generator)
        w13 = torch.randn(
            (experts, rows, hidden),
            device=device,
            dtype=dtype,
            generator=generator,
        ) * 0.02
        ids = torch.topk(logits.float(), top_k, dim=-1).indices[0]
        selected = w13[ids]
        reference = F.linear(
            states, selected.reshape(-1, hidden)).view(top_k, rows)
        fixed[str(seed)] = {
            "direct": compare(direct.w13(states, w13, ids), reference),
            "pairwise": compare(candidate.w13(states, w13, ids), reference),
        }
        if seed == FIXED_SEEDS[-1]:
            timing_state = (states, logits, w13, ids, selected)

    assert timing_state is not None
    states, logits, w13, ids, selected = timing_state

    def reference_fixed():
        return F.linear(
            states, selected.reshape(-1, hidden)).view(top_k, rows)

    def reference_routed():
        routed_ids = torch.topk(logits.float(), top_k, dim=-1).indices[0]
        routed_weights = w13[routed_ids]
        return F.linear(
            states, routed_weights.reshape(-1, hidden)).view(top_k, rows)

    def candidate_routed():
        routed_ids = torch.topk(logits.float(), top_k, dim=-1).indices[0]
        return candidate.w13(states, w13, routed_ids)

    cases = {
        "reference_fixed": reference_fixed,
        "direct_fixed": lambda: direct.w13(states, w13, ids),
        "pairwise_fixed": lambda: candidate.w13(states, w13, ids),
        "reference_routed": reference_routed,
        "pairwise_routed": candidate_routed,
    }
    timings = {
        name: measure(case, args.warmup, args.iterations, args.repeats)
        for name, case in cases.items()
    }
    timings["pairwise_fixed"]["speedup_vs_reference"] = (
        timings["reference_fixed"]["median_ms"]
        / timings["pairwise_fixed"]["median_ms"])
    timings["pairwise_routed"]["speedup_vs_reference"] = (
        timings["reference_routed"]["median_ms"]
        / timings["pairwise_routed"]["median_ms"])

    generator = torch.Generator(
        device=device).manual_seed(args.sequence_seed)
    sequence_w13 = torch.randn(
        (experts, rows, hidden),
        device=device,
        dtype=dtype,
        generator=generator,
    ) * 0.02
    names = ("direct", "pairwise")
    squared_error = {name: 0.0 for name in names}
    squared_reference = {name: 0.0 for name in names}
    max_abs = {name: 0.0 for name in names}
    max_step_relative_l2 = {name: 0.0 for name in names}
    finite_steps = {name: 0 for name in names}
    exact_steps = {name: 0 for name in names}
    for _ in range(args.sequence_steps):
        step_states = torch.randn(
            (1, hidden), device=device, dtype=dtype, generator=generator)
        step_logits = torch.randn(
            (1, experts), device=device, dtype=dtype, generator=generator)
        step_ids = torch.topk(
            step_logits.float(), top_k, dim=-1).indices[0]
        step_selected = sequence_w13[step_ids]
        expected = F.linear(
            step_states,
            step_selected.reshape(-1, hidden),
        ).view(top_k, rows)
        outputs = {
            "direct": direct.w13(step_states, sequence_w13, step_ids),
            "pairwise": candidate.w13(
                step_states, sequence_w13, step_ids),
        }
        for name, actual in outputs.items():
            delta = (actual.float() - expected.float()).abs()
            error = float(delta.square().sum())
            reference = float(expected.float().square().sum())
            relative_l2 = math.sqrt(error / max(reference, 1.0e-30))
            squared_error[name] += error
            squared_reference[name] += reference
            max_abs[name] = max(max_abs[name], float(delta.max()))
            max_step_relative_l2[name] = max(
                max_step_relative_l2[name], relative_l2)
            finite_steps[name] += int(torch.isfinite(actual).all())
            exact_steps[name] += int(torch.equal(actual, expected))

    report = {
        "schema": "bi100-moe-pairwise-w13-v1",
        "shape": {
            "experts": experts,
            "top_k": top_k,
            "hidden": hidden,
            "rows_per_expert": rows,
            "dtype": str(dtype),
        },
        "config": {
            "device": args.device,
            "fixed_seeds": list(FIXED_SEEDS),
            "sequence_seed": args.sequence_seed,
            "sequence_steps": args.sequence_steps,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
        },
        "fixed": fixed,
        "sequence": {
            name: {
                "steps": args.sequence_steps,
                "finite_steps": finite_steps[name],
                "exact_steps": exact_steps[name],
                "max_abs": max_abs[name],
                "relative_l2": math.sqrt(
                    squared_error[name]
                    / max(squared_reference[name], 1.0e-30)),
                "max_step_relative_l2": max_step_relative_l2[name],
            }
            for name in names
        },
        "timings": timings,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
