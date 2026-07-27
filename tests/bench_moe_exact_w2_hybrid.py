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
        "p90_ms": ordered[min(
            len(ordered) - 1, int(0.9 * (len(ordered) - 1)))],
        "trials_ms": trials,
    }


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    delta = (actual_fp32 - expected_fp32).abs()
    squared_error = float(delta.square().sum())
    squared_reference = float(expected_fp32.square().sum())
    return {
        "exact": bool(torch.equal(actual, expected)),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "relative_l2": math.sqrt(
            squared_error / max(squared_reference, 1.0e-30)),
        "finite": bool(torch.isfinite(actual).all()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-extension", type=Path, required=True)
    parser.add_argument("--gather-extension", type=Path, required=True)
    parser.add_argument("--reduce-extension", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--sequence-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260727)
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
    if not hasattr(gather, "gather_w2"):
        raise RuntimeError("gather extension does not expose gather_w2")

    experts, top_k, hidden, intermediate = 256, 8, 2048, 128
    dtype = torch.float16
    generator = torch.Generator(device=device).manual_seed(args.seed)
    activation = SiluAndMul()
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
    ) -> torch.Tensor:
        selected_w13, selected_w2 = gather.gather(w13, w2, ids)
        gate_up = F.linear(
            states, selected_w13.reshape(-1, hidden)).view(top_k, -1)
        activated = activation(gate_up)
        expert_output = torch.bmm(
            selected_w2, activated.unsqueeze(-1)).squeeze(-1)
        return reducer.serial_float(expert_output, weights)

    def hybrid_from_route(
        states: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor,
    ) -> torch.Tensor:
        gate_up = direct.w13(states, w13, ids)
        activated = activation(gate_up)
        selected_w2 = gather.gather_w2(w2, ids)
        expert_output = torch.bmm(
            selected_w2, activated.unsqueeze(-1)).squeeze(-1)
        return reducer.serial_float(expert_output, weights)

    weights, ids = route(router_logits)
    reference = baseline_from_route(hidden_states, weights, ids)
    candidate = hybrid_from_route(hidden_states, weights, ids)
    selected_w13, selected_w2 = gather.gather(w13, w2, ids)
    reference_gate = F.linear(
        hidden_states, selected_w13.reshape(-1, hidden)).view(top_k, -1)
    direct_gate = direct.w13(hidden_states, w13, ids)
    w2_only = gather.gather_w2(w2, ids)

    cases = {
        "baseline_gather": lambda: gather.gather(w13, w2, ids),
        "w2_only_gather": lambda: gather.gather_w2(w2, ids),
        "baseline_fixed": lambda: baseline_from_route(
            hidden_states, weights, ids),
        "hybrid_fixed": lambda: hybrid_from_route(
            hidden_states, weights, ids),
        "baseline_routed": lambda: baseline_from_route(
            hidden_states, *route(router_logits)),
        "hybrid_routed": lambda: hybrid_from_route(
            hidden_states, *route(router_logits)),
    }
    timings = {
        name: measure(case, args.warmup, args.iterations, args.repeats)
        for name, case in cases.items()
    }
    timings["hybrid_fixed"]["speedup_vs_baseline"] = (
        timings["baseline_fixed"]["median_ms"]
        / timings["hybrid_fixed"]["median_ms"])
    timings["hybrid_routed"]["speedup_vs_baseline"] = (
        timings["baseline_routed"]["median_ms"]
        / timings["hybrid_routed"]["median_ms"])

    squared_error = 0.0
    squared_reference = 0.0
    max_abs = 0.0
    mean_abs = []
    exact_steps = 0
    finite_steps = 0
    max_step_relative_l2 = 0.0
    for _ in range(args.sequence_steps):
        step_hidden = torch.randn(
            (1, hidden), device=device, dtype=dtype, generator=generator)
        step_logits = torch.randn(
            (1, experts), device=device, dtype=dtype, generator=generator)
        step_weights, step_ids = route(step_logits)
        expected = baseline_from_route(
            step_hidden, step_weights, step_ids)
        actual = hybrid_from_route(
            step_hidden, step_weights, step_ids)
        delta = (actual.float() - expected.float()).abs()
        step_squared_error = float(delta.square().sum())
        step_squared_reference = float(expected.float().square().sum())
        step_relative_l2 = math.sqrt(
            step_squared_error / max(step_squared_reference, 1.0e-30))
        squared_error += step_squared_error
        squared_reference += step_squared_reference
        max_abs = max(max_abs, float(delta.max()))
        mean_abs.append(float(delta.mean()))
        exact_steps += int(torch.equal(actual, expected))
        finite_steps += int(torch.isfinite(actual).all())
        max_step_relative_l2 = max(
            max_step_relative_l2, step_relative_l2)

    report = {
        "schema": "bi100-moe-exact-w2-hybrid-v1",
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
        "checks": {
            "selected_w2_exact": bool(torch.equal(w2_only, selected_w2)),
            "direct_w13": compare(direct_gate, reference_gate),
            "hybrid": compare(candidate, reference),
        },
        "sequence": {
            "steps": args.sequence_steps,
            "exact_steps": exact_steps,
            "finite_steps": finite_steps,
            "max_abs": max_abs,
            "mean_abs": statistics.mean(mean_abs),
            "relative_l2": math.sqrt(
                squared_error / max(squared_reference, 1.0e-30)),
            "max_step_relative_l2": max_step_relative_l2,
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
