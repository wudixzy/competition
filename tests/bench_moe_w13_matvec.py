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
    return {"median_ms": statistics.median(trials), "trials_ms": trials}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--linear-extension", type=Path, required=True)
    parser.add_argument("--gather-extension", type=Path, required=True)
    parser.add_argument("--reduce-extension", type=Path, required=True)
    parser.add_argument(
        "--modes",
        default="0,1,32,64,128,256,1032,1064,1128,1256,"
                "2032,2064,2128,2256")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--sequence-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    linear = load_extension("corex_moe_w13_matvec", args.linear_extension)
    gather = load_extension(
        "corex_moe_weight_gather", args.gather_extension)
    reducer = load_extension("corex_moe_exact_reduce", args.reduce_extension)
    modes = [int(value) for value in args.modes.split(",")]
    generator = torch.Generator(device=device).manual_seed(args.seed)
    experts, top_k, hidden, intermediate = 256, 8, 2048, 128
    dtype = torch.float16
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
    activation = SiluAndMul()

    def route(logits):
        selected, ids = torch.topk(logits.float(), top_k, dim=-1)
        weights = torch.softmax(selected, dim=-1)[0].to(dtype)
        return weights, ids[0]

    def finish(gate_up, selected_w2, weights):
        activated = activation(gate_up)
        expert_output = torch.bmm(
            selected_w2, activated.unsqueeze(-1)).squeeze(-1)
        return reducer.serial_float(expert_output, weights)

    weights, ids = route(router_logits)
    selected_w13, selected_w2 = gather.gather(w13, w2, ids)
    flat_w13 = selected_w13.reshape(-1, hidden)
    expected_gate = F.linear(hidden_states, flat_w13).view(top_k, -1)
    expected = finish(expected_gate, selected_w2, weights)
    baseline_linear = measure(
        lambda: F.linear(hidden_states, flat_w13),
        args.warmup, args.iterations, args.repeats)

    def baseline_full():
        current_w13, current_w2 = gather.gather(w13, w2, ids)
        gate_up = F.linear(
            hidden_states, current_w13.reshape(-1, hidden)).view(top_k, -1)
        return finish(gate_up, current_w2, weights)

    baseline_full_timing = measure(
        baseline_full, args.warmup, args.iterations, args.repeats)
    results = {}
    for mode in modes:
        actual_gate = linear.linear(
            hidden_states, flat_w13, mode).view(top_k, -1)
        actual = finish(actual_gate, selected_w2, weights)
        linear_timing = measure(
            lambda current=mode: linear.linear(
                hidden_states, flat_w13, current),
            args.warmup, args.iterations, args.repeats)

        def candidate_full(current=mode):
            current_w13, current_w2 = gather.gather(w13, w2, ids)
            gate_up = linear.linear(
                hidden_states, current_w13.reshape(-1, hidden),
                current).view(top_k, -1)
            return finish(gate_up, current_w2, weights)

        full_timing = measure(
            candidate_full, args.warmup, args.iterations, args.repeats)
        linear_timing["speedup_vs_baseline"] = (
            baseline_linear["median_ms"] / linear_timing["median_ms"])
        full_timing["speedup_vs_baseline"] = (
            baseline_full_timing["median_ms"] / full_timing["median_ms"])
        results[str(mode)] = {
            "gate_exact": bool(torch.equal(actual_gate, expected_gate)),
            "gate_max_abs": float((
                actual_gate.float() - expected_gate.float()).abs().max()),
            "output_exact": bool(torch.equal(actual, expected)),
            "max_abs": float((actual.float() - expected.float()).abs().max()),
            "linear": linear_timing,
            "full": full_timing,
        }

    exact_modes = [
        mode for mode in modes
        if results[str(mode)]["gate_exact"]
        and results[str(mode)]["output_exact"]
    ]
    best_mode = min(
        exact_modes,
        key=lambda mode: results[str(mode)]["full"]["median_ms"],
    ) if exact_modes else None

    sequence = None
    if best_mode is not None:
        exact_steps = 0
        max_abs = 0.0
        for _ in range(args.sequence_steps):
            step_hidden = torch.randn(
                (1, hidden), device=device, dtype=dtype, generator=generator)
            step_logits = torch.randn(
                (1, experts), device=device, dtype=dtype, generator=generator)
            step_weights, step_ids = route(step_logits)
            step_w13, step_w2 = gather.gather(w13, w2, step_ids)
            reference_gate = F.linear(
                step_hidden, step_w13.reshape(-1, hidden)).view(top_k, -1)
            candidate_gate = linear.linear(
                step_hidden, step_w13.reshape(-1, hidden),
                best_mode).view(top_k, -1)
            reference_output = finish(reference_gate, step_w2, step_weights)
            candidate_output = finish(
                candidate_gate, step_w2, step_weights)
            exact_steps += int(
                torch.equal(candidate_gate, reference_gate)
                and torch.equal(candidate_output, reference_output))
            max_abs = max(
                max_abs,
                float((candidate_output.float()
                       - reference_output.float()).abs().max()),
            )
        sequence = {
            "exact_steps": exact_steps,
            "steps": args.sequence_steps,
            "max_abs": max_abs,
        }

    report = {
        "device": torch.cuda.get_device_name(device),
        "config": vars(args) | {
            "linear_extension": str(args.linear_extension),
            "gather_extension": str(args.gather_extension),
            "reduce_extension": str(args.reduce_extension),
            "out": str(args.out),
        },
        "baseline": {
            "linear": baseline_linear,
            "full": baseline_full_timing,
        },
        "best_exact_mode": best_mode,
        "sequence": sequence,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
