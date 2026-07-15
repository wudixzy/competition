#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.activation import SiluAndMul


def load_extension(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        "p90_ms": ordered[min(len(ordered) - 1,
                               int(0.9 * (len(ordered) - 1)))],
        "trials_ms": trials,
    }


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    delta = (actual.float() - expected.float()).abs()
    denominator = expected.float().abs().clamp_min(1.0e-6)
    return {
        "exact": bool(torch.equal(actual, expected)),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "max_rel": float((delta / denominator).max()),
        "finite": bool(torch.isfinite(actual).all()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-extension", type=Path, required=True)
    parser.add_argument("--gather-extension", type=Path, required=True)
    parser.add_argument("--reduce-extension", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--sequence-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    direct = load_extension(
        "corex_moe_direct_routed", args.direct_extension)
    gather = load_extension(
        "corex_moe_weight_gather", args.gather_extension)
    reducer = load_extension(
        "corex_moe_exact_reduce", args.reduce_extension)
    activation = SiluAndMul()

    experts, top_k, hidden, intermediate = 256, 8, 2048, 128
    dtype = torch.float16
    generator = torch.Generator(device=device).manual_seed(args.seed)
    hidden_states = torch.randn(
        (1, hidden), device=device, dtype=dtype, generator=generator)
    router_logits = torch.randn(
        (1, experts), device=device, dtype=dtype, generator=generator)
    w13 = torch.randn(
        (experts, 2 * intermediate, hidden), device=device, dtype=dtype,
        generator=generator) * 0.02
    w2 = torch.randn(
        (experts, hidden, intermediate), device=device, dtype=dtype,
        generator=generator) * 0.02

    def route(logits: torch.Tensor):
        selected, ids = torch.topk(logits.float(), top_k, dim=-1)
        weights = torch.softmax(selected, dim=-1)[0].to(dtype)
        return weights, ids[0]

    def baseline_from_route(
        states: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor,
    ):
        selected_w13, selected_w2 = gather.gather(w13, w2, ids)
        gate_up = F.linear(
            states, selected_w13.reshape(-1, hidden)).view(top_k, -1)
        activated = activation(gate_up)
        expert_output = torch.bmm(
            selected_w2, activated.unsqueeze(-1)).squeeze(-1)
        return reducer.serial_float(expert_output, weights)

    def staged_from_route(
        states: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor,
    ):
        gate_up = direct.w13(states, w13, ids)
        activated = activation(gate_up)
        return direct.w2_reduce(activated, w2, ids, weights)

    def fused_from_route(
        states: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor,
    ):
        activated = direct.w13_silu(states, w13, ids)
        return direct.w2_reduce(activated, w2, ids, weights)

    weights, ids = route(router_logits)
    selected_w13, selected_w2 = gather.gather(w13, w2, ids)
    reference_gate = F.linear(
        hidden_states, selected_w13.reshape(-1, hidden)).view(top_k, -1)
    reference_activation = activation(reference_gate)
    reference_expert = torch.bmm(
        selected_w2, reference_activation.unsqueeze(-1)).squeeze(-1)
    reference = reducer.serial_float(reference_expert, weights)
    direct_gate = direct.w13(hidden_states, w13, ids)
    direct_activation = direct.w13_silu(hidden_states, w13, ids)
    direct_tail = direct.w2_reduce(
        reference_activation, w2, ids, weights)
    staged = staged_from_route(hidden_states, weights, ids)
    fused = fused_from_route(hidden_states, weights, ids)

    cases = {
        "baseline_gather": lambda: gather.gather(w13, w2, ids),
        "baseline_w13": lambda: F.linear(
            hidden_states, selected_w13.reshape(-1, hidden)),
        "baseline_activation": lambda: activation(reference_gate),
        "baseline_w2": lambda: torch.bmm(
            selected_w2, reference_activation.unsqueeze(-1)).squeeze(-1),
        "baseline_reduce": lambda: reducer.serial_float(
            reference_expert, weights),
        "direct_w13": lambda: direct.w13(hidden_states, w13, ids),
        "direct_w13_silu": lambda: direct.w13_silu(
            hidden_states, w13, ids),
        "direct_w2_reduce": lambda: direct.w2_reduce(
            reference_activation, w2, ids, weights),
        "baseline_fixed": lambda: baseline_from_route(
            hidden_states, weights, ids),
        "staged_fixed": lambda: staged_from_route(
            hidden_states, weights, ids),
        "fused_fixed": lambda: fused_from_route(
            hidden_states, weights, ids),
        "baseline_routed": lambda: baseline_from_route(
            hidden_states, *route(router_logits)),
        "staged_routed": lambda: staged_from_route(
            hidden_states, *route(router_logits)),
        "fused_routed": lambda: fused_from_route(
            hidden_states, *route(router_logits)),
    }
    timings = {
        name: measure(case, args.warmup, args.iterations, args.repeats)
        for name, case in cases.items()
    }
    for candidate in ("staged_fixed", "fused_fixed"):
        timings[candidate]["speedup_vs_baseline"] = (
            timings["baseline_fixed"]["median_ms"]
            / timings[candidate]["median_ms"])
    for candidate in ("staged_routed", "fused_routed"):
        timings[candidate]["speedup_vs_baseline"] = (
            timings["baseline_routed"]["median_ms"]
            / timings[candidate]["median_ms"])

    max_abs = {"staged": 0.0, "fused": 0.0}
    mean_abs = {"staged": [], "fused": []}
    exact_steps = {"staged": 0, "fused": 0}
    finite_steps = {"staged": 0, "fused": 0}
    for _ in range(args.sequence_steps):
        step_hidden = torch.randn(
            (1, hidden), device=device, dtype=dtype, generator=generator)
        step_logits = torch.randn(
            (1, experts), device=device, dtype=dtype, generator=generator)
        step_weights, step_ids = route(step_logits)
        expected = baseline_from_route(step_hidden, step_weights, step_ids)
        candidates = {
            "staged": staged_from_route(
                step_hidden, step_weights, step_ids),
            "fused": fused_from_route(
                step_hidden, step_weights, step_ids),
        }
        for name, actual in candidates.items():
            delta = (actual.float() - expected.float()).abs()
            max_abs[name] = max(max_abs[name], float(delta.max()))
            mean_abs[name].append(float(delta.mean()))
            exact_steps[name] += int(torch.equal(actual, expected))
            finite_steps[name] += int(torch.isfinite(actual).all())

    report = {
        "device": torch.cuda.get_device_name(device),
        "shape": {
            "experts": experts,
            "top_k": top_k,
            "hidden": hidden,
            "intermediate": intermediate,
            "dtype": str(dtype),
        },
        "config": {
            "device": args.device,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "sequence_steps": args.sequence_steps,
            "seed": args.seed,
        },
        "numerics": {
            "direct_w13": compare(direct_gate, reference_gate),
            "direct_w13_silu": compare(
                direct_activation, reference_activation),
            "direct_w2_reduce": compare(direct_tail, reference),
            "staged": compare(staged, reference),
            "fused": compare(fused, reference),
        },
        "sequence": {
            name: {
                "steps": args.sequence_steps,
                "exact_steps": exact_steps[name],
                "finite_steps": finite_steps[name],
                "max_abs": max_abs[name],
                "mean_abs": statistics.mean(mean_abs[name]),
                "p99_mean_abs": sorted(mean_abs[name])[
                    min(len(mean_abs[name]) - 1,
                        int(0.99 * (len(mean_abs[name]) - 1)))],
            }
            for name in ("staged", "fused")
        },
        "timings": timings,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
