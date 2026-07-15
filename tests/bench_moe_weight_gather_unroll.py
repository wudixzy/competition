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
    parser.add_argument("--baseline-extension", type=Path, required=True)
    parser.add_argument("--candidate-extension", type=Path, required=True)
    parser.add_argument("--reduce-extension", type=Path, required=True)
    parser.add_argument("--unroll", default="1,2,4,8")
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
    baseline = load_extension(
        "corex_moe_weight_gather", args.baseline_extension)
    candidate = load_extension(
        "corex_moe_weight_gather_unroll", args.candidate_extension)
    reducer = load_extension("corex_moe_exact_reduce", args.reduce_extension)
    unroll_values = [int(value) for value in args.unroll.split(",")]
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

    def finish(selected_w13, selected_w2, weights):
        gate_up = F.linear(
            hidden_states, selected_w13.reshape(-1, hidden)).view(top_k, -1)
        activated = activation(gate_up)
        expert_output = torch.bmm(
            selected_w2, activated.unsqueeze(-1)).squeeze(-1)
        return reducer.serial_float(expert_output, weights)

    weights, ids = route(router_logits)
    baseline_w13, baseline_w2 = baseline.gather(w13, w2, ids)
    reference = finish(baseline_w13, baseline_w2, weights)
    results = {
        "baseline_gather": measure(
            lambda: baseline.gather(w13, w2, ids),
            args.warmup, args.iterations, args.repeats),
        "baseline_full": measure(
            lambda: finish(*baseline.gather(w13, w2, ids), weights),
            args.warmup, args.iterations, args.repeats),
        "unroll": {},
    }
    for value in unroll_values:
        selected_w13, selected_w2 = candidate.gather(w13, w2, ids, value)
        actual = finish(selected_w13, selected_w2, weights)
        gather_timing = measure(
            lambda u=value: candidate.gather(w13, w2, ids, u),
            args.warmup, args.iterations, args.repeats)
        full_timing = measure(
            lambda u=value: finish(
                *candidate.gather(w13, w2, ids, u), weights),
            args.warmup, args.iterations, args.repeats)
        gather_timing["speedup_vs_baseline"] = (
            results["baseline_gather"]["median_ms"] /
            gather_timing["median_ms"])
        full_timing["speedup_vs_baseline"] = (
            results["baseline_full"]["median_ms"] /
            full_timing["median_ms"])
        results["unroll"][str(value)] = {
            "w13_exact": bool(torch.equal(selected_w13, baseline_w13)),
            "w2_exact": bool(torch.equal(selected_w2, baseline_w2)),
            "output_exact": bool(torch.equal(actual, reference)),
            "max_abs": float((actual.float() - reference.float()).abs().max()),
            "gather": gather_timing,
            "full": full_timing,
        }

    best = min(
        unroll_values,
        key=lambda value: results["unroll"][str(value)]["full"]["median_ms"],
    )

    def baseline_routed():
        routed_weights, routed_ids = route(router_logits)
        return finish(
            *baseline.gather(w13, w2, routed_ids), routed_weights)

    def candidate_routed():
        routed_weights, routed_ids = route(router_logits)
        return finish(
            *candidate.gather(w13, w2, routed_ids, best), routed_weights)

    routed = {
        "baseline": measure(
            baseline_routed, args.warmup, args.iterations, args.repeats),
        "candidate": measure(
            candidate_routed, args.warmup, args.iterations, args.repeats),
    }
    routed["candidate"]["speedup_vs_baseline"] = (
        routed["baseline"]["median_ms"] /
        routed["candidate"]["median_ms"])

    exact_steps = 0
    max_abs = 0.0
    for _ in range(args.sequence_steps):
        step_logits = torch.randn(
            (1, experts), device=device, dtype=dtype, generator=generator)
        step_weights, step_ids = route(step_logits)
        expected = finish(*baseline.gather(w13, w2, step_ids), step_weights)
        actual = finish(
            *candidate.gather(w13, w2, step_ids, best), step_weights)
        exact_steps += int(torch.equal(actual, expected))
        max_abs = max(
            max_abs, float((actual.float() - expected.float()).abs().max()))

    report = {
        "device": torch.cuda.get_device_name(device),
        "config": vars(args) | {
            "baseline_extension": str(args.baseline_extension),
            "candidate_extension": str(args.candidate_extension),
            "reduce_extension": str(args.reduce_extension),
            "out": str(args.out),
        },
        "best_unroll": best,
        "sequence": {
            "exact_steps": exact_steps,
            "steps": args.sequence_steps,
            "max_abs": max_abs,
        },
        "routed": routed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
