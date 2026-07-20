#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import statistics
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.activation import SiluAndMul


MAX_ABS_LIMIT = 0.0001220703125
MEAN_ABS_LIMIT = 6.8e-6
MIN_SPEEDUP = {2: 1.25, 8: 1.5, 16: 1.5}
MIN_PROJECTED_WARM_TTFT_GAIN = 0.05
PROJECTION_GATE_TOKENS = {8, 16}


def load_extension(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_production_method(path: Path, namespace: dict):
    tree = ast.parse(path.read_text(), filename=str(path))
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "Qwen3_5MoeSparseBlock")
    method = next(
        node for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_pure_pytorch_experts")
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["_pure_pytorch_experts"]


def measure_once(case, iterations: int) -> float:
    started = time.perf_counter()
    for _ in range(iterations):
        case()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0 / iterations


def measure_pair(baseline, candidate, warmup: int, iterations: int,
                 repeats: int) -> dict:
    for _ in range(warmup):
        baseline()
        candidate()
    torch.cuda.synchronize()

    baseline_trials = []
    candidate_trials = []
    for repeat in range(repeats):
        if repeat % 2 == 0:
            baseline_trials.append(measure_once(baseline, iterations))
            candidate_trials.append(measure_once(candidate, iterations))
        else:
            candidate_trials.append(measure_once(candidate, iterations))
            baseline_trials.append(measure_once(baseline, iterations))

    baseline_median = statistics.median(baseline_trials)
    candidate_median = statistics.median(candidate_trials)
    return {
        "baseline": {
            "median_ms": baseline_median,
            "trials_ms": baseline_trials,
        },
        "candidate": {
            "median_ms": candidate_median,
            "trials_ms": candidate_trials,
        },
        "speedup": baseline_median / candidate_median,
        "saved_ms_per_layer": baseline_median - candidate_median,
    }


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual_float = actual.float()
    expected_float = expected.float()
    delta = (actual_float - expected_float).abs()
    relative_l2 = float(
        torch.linalg.vector_norm(actual_float - expected_float)
        / torch.linalg.vector_norm(expected_float).clamp_min(1.0e-12))
    return {
        "exact": bool(torch.equal(actual, expected)),
        "finite": bool(torch.isfinite(actual).all()),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "relative_l2": relative_l2,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--direct-extension", type=Path, required=True)
    parser.add_argument(
        "--candidate-mode", choices=("direct", "sorted-half"),
        default="direct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--sequence-steps", type=int, default=40)
    parser.add_argument("--moe-layers", type=int, default=40)
    parser.add_argument("--warm-ttft-ms", type=float, default=1443.85)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    direct = load_extension(
        "corex_moe_direct_routed", args.direct_extension)
    namespace = {
        "torch": torch,
        "F": F,
        "_corex_moe_direct_routed": direct,
        "_corex_moe_weight_gather": None,
        "_corex_moe_exact_reduce": None,
        "_USE_COREX_MOE_DIRECT_ROUTED": True,
        "_USE_COREX_MOE_WEIGHT_GATHER": False,
        "_USE_COREX_MOE_EXACT_REDUCE": False,
        "_USE_FUSED_MOE_ACTIVATION": True,
    }
    production = load_production_method(args.model_source, namespace)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    experts, top_k, hidden, intermediate = 256, 8, 2048, 128
    dtype = torch.float16
    w13 = torch.randn(
        (experts, 2 * intermediate, hidden), device=device, dtype=dtype,
        generator=generator) * 0.02
    w2 = torch.randn(
        (experts, hidden, intermediate), device=device, dtype=dtype,
        generator=generator) * 0.02
    activation = SiluAndMul()
    fake_self = types.SimpleNamespace(
        top_k=top_k,
        experts=types.SimpleNamespace(w13_weight=w13, w2_weight=w2),
        act_fn=activation,
    )

    def reference(states: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        return production(fake_self, states, logits)

    def direct_loop(states: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        selected, expert_ids = torch.topk(logits.float(), top_k, dim=-1)
        route_weights = torch.softmax(selected, dim=-1).to(dtype)
        if args.candidate_mode == "sorted-half":
            order = torch.argsort(expert_ids, dim=-1, stable=True)
            expert_ids = torch.gather(expert_ids, 1, order)
            route_weights = torch.gather(route_weights, 1, order)
        outputs = []
        for token in range(states.shape[0]):
            gate_up = direct.w13(
                states[token:token + 1], w13, expert_ids[token])
            activated = activation(gate_up)
            if args.candidate_mode == "sorted-half":
                outputs.append(direct.w2_reduce_serial_half(
                    activated, w2, expert_ids[token], route_weights[token]))
            else:
                outputs.append(direct.w2_reduce(
                    activated, w2, expert_ids[token], route_weights[token]))
        return torch.cat(outputs, dim=0)

    cases = {}
    all_numerics_pass = True
    all_speed_pass = True
    all_projection_pass = True
    for tokens in sorted(MIN_SPEEDUP):
        fixed_states = torch.randn(
            (tokens, hidden), device=device, dtype=dtype,
            generator=generator)
        fixed_logits = torch.randn(
            (tokens, experts), device=device, dtype=dtype,
            generator=generator)
        fixed_reference = reference(fixed_states, fixed_logits)
        fixed_candidate = direct_loop(fixed_states, fixed_logits)
        fixed_numerics = compare(fixed_candidate, fixed_reference)

        sequence_max_abs = 0.0
        sequence_mean_abs = []
        sequence_max_relative_l2 = 0.0
        finite_steps = 0
        for _ in range(args.sequence_steps):
            states = torch.randn(
                (tokens, hidden), device=device, dtype=dtype,
                generator=generator)
            logits = torch.randn(
                (tokens, experts), device=device, dtype=dtype,
                generator=generator)
            comparison = compare(
                direct_loop(states, logits), reference(states, logits))
            finite_steps += int(comparison["finite"])
            sequence_max_abs = max(
                sequence_max_abs, comparison["max_abs"])
            sequence_mean_abs.append(comparison["mean_abs"])
            sequence_max_relative_l2 = max(
                sequence_max_relative_l2, comparison["relative_l2"])

        timing = measure_pair(
            lambda: reference(fixed_states, fixed_logits),
            lambda: direct_loop(fixed_states, fixed_logits),
            args.warmup,
            args.iterations,
            args.repeats,
        )
        projected_saved_ms = (
            timing["saved_ms_per_layer"] * args.moe_layers)
        projected_gain = projected_saved_ms / args.warm_ttft_ms
        numerics_pass = (
            fixed_numerics["finite"]
            and finite_steps == args.sequence_steps
            and sequence_max_abs <= MAX_ABS_LIMIT
            and statistics.mean(sequence_mean_abs) <= MEAN_ABS_LIMIT)
        speed_pass = timing["speedup"] >= MIN_SPEEDUP[tokens]
        projection_pass = projected_gain >= MIN_PROJECTED_WARM_TTFT_GAIN
        all_numerics_pass &= numerics_pass
        all_speed_pass &= speed_pass
        if tokens in PROJECTION_GATE_TOKENS:
            all_projection_pass &= projection_pass
        cases[str(tokens)] = {
            "fixed_numerics": fixed_numerics,
            "sequence": {
                "steps": args.sequence_steps,
                "finite_steps": finite_steps,
                "max_abs": sequence_max_abs,
                "mean_abs": statistics.mean(sequence_mean_abs),
                "max_relative_l2": sequence_max_relative_l2,
            },
            "timing": timing,
            "projection": {
                "moe_layers": args.moe_layers,
                "warm_ttft_ms": args.warm_ttft_ms,
                "saved_ms": projected_saved_ms,
                "relative_gain": projected_gain,
            },
            "gates": {
                "numerics": numerics_pass,
                "speed": speed_pass,
                "projection_required": tokens in PROJECTION_GATE_TOKENS,
                "projection": projection_pass,
            },
        }

    report = {
        "device": torch.cuda.get_device_name(device),
        "shape": {
            "experts": experts,
            "top_k": top_k,
            "hidden": hidden,
            "intermediate": intermediate,
            "dtype": str(dtype),
            "tokens": sorted(MIN_SPEEDUP),
        },
        "config": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "sequence_steps": args.sequence_steps,
            "seed": args.seed,
            "candidate_mode": args.candidate_mode,
        },
        "limits": {
            "max_abs": MAX_ABS_LIMIT,
            "mean_abs": MEAN_ABS_LIMIT,
            "relative_l2": "diagnostic_only",
            "min_speedup": MIN_SPEEDUP,
            "min_projected_warm_ttft_gain": MIN_PROJECTED_WARM_TTFT_GAIN,
            "projection_gate_tokens": sorted(PROJECTION_GATE_TOKENS),
        },
        "cases": cases,
        "qualification": {
            "numerics": all_numerics_pass,
            "speed": all_speed_pass,
            "projection": all_projection_pass,
            "passed": (
                all_numerics_pass
                and all_speed_pass
                and all_projection_pass),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["qualification"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
