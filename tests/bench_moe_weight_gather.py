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
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--reduce-extension", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grid-caps", default="128,256,512,1024,2048,8192")
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
    extension = load_extension("corex_moe_weight_gather", args.extension)
    reducer = load_extension("corex_moe_exact_reduce", args.reduce_extension)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    experts, top_k, hidden, intermediate = 256, 8, 2048, 128
    dtype = torch.float16
    grid_caps = [int(value) for value in args.grid_caps.split(",")]
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
        selected_logits, ids = torch.topk(logits.float(), top_k, dim=-1)
        weights = torch.softmax(selected_logits, dim=-1)[0].to(dtype)
        return weights, ids[0]

    def finish(selected_w13, selected_w2, weights):
        gate_up = F.linear(
            hidden_states, selected_w13.reshape(-1, hidden)).view(top_k, -1)
        activated = activation(gate_up)
        expert_output = torch.bmm(
            selected_w2, activated.unsqueeze(-1)).squeeze(-1)
        return reducer.serial_float(expert_output, weights)

    fixed_weights, fixed_ids = route(router_logits)
    native_w13 = w13[fixed_ids]
    native_w2 = w2[fixed_ids]
    reference = finish(native_w13, native_w2, fixed_weights)

    results = {
        "native_gather": measure(
            lambda: (w13[fixed_ids], w2[fixed_ids]),
            args.warmup, args.iterations, args.repeats),
        "native_full": measure(
            lambda: finish(w13[fixed_ids], w2[fixed_ids], fixed_weights),
            args.warmup, args.iterations, args.repeats),
        "grid": {},
        "grid_half2": {},
    }
    for grid_cap in grid_caps:
        selected_w13, selected_w2 = extension.gather(
            w13, w2, fixed_ids, grid_cap)
        actual = finish(selected_w13, selected_w2, fixed_weights)
        gather_timing = measure(
            lambda cap=grid_cap: extension.gather(w13, w2, fixed_ids, cap),
            args.warmup, args.iterations, args.repeats)
        full_timing = measure(
            lambda cap=grid_cap: finish(
                *extension.gather(w13, w2, fixed_ids, cap), fixed_weights),
            args.warmup, args.iterations, args.repeats)
        gather_timing["speedup_vs_native"] = (
            results["native_gather"]["median_ms"] /
            gather_timing["median_ms"])
        full_timing["speedup_vs_native"] = (
            results["native_full"]["median_ms"] /
            full_timing["median_ms"])
        results["grid"][str(grid_cap)] = {
            "w13_exact": bool(torch.equal(selected_w13, native_w13)),
            "w2_exact": bool(torch.equal(selected_w2, native_w2)),
            "output_exact": bool(torch.equal(actual, reference)),
            "max_abs": float((actual.float() - reference.float()).abs().max()),
            "gather": gather_timing,
            "full": full_timing,
        }

        half2_w13, half2_w2 = extension.gather_half2(
            w13, w2, fixed_ids, grid_cap)
        half2_output = finish(half2_w13, half2_w2, fixed_weights)
        half2_gather_timing = measure(
            lambda cap=grid_cap: extension.gather_half2(
                w13, w2, fixed_ids, cap),
            args.warmup, args.iterations, args.repeats)
        half2_full_timing = measure(
            lambda cap=grid_cap: finish(
                *extension.gather_half2(w13, w2, fixed_ids, cap),
                fixed_weights),
            args.warmup, args.iterations, args.repeats)
        half2_gather_timing["speedup_vs_native"] = (
            results["native_gather"]["median_ms"] /
            half2_gather_timing["median_ms"])
        half2_full_timing["speedup_vs_native"] = (
            results["native_full"]["median_ms"] /
            half2_full_timing["median_ms"])
        results["grid_half2"][str(grid_cap)] = {
            "w13_exact": bool(torch.equal(half2_w13, native_w13)),
            "w2_exact": bool(torch.equal(half2_w2, native_w2)),
            "output_exact": bool(torch.equal(half2_output, reference)),
            "max_abs": float((
                half2_output.float() - reference.float()).abs().max()),
            "gather": half2_gather_timing,
            "full": half2_full_timing,
        }

    best_cap = min(grid_caps, key=lambda cap: results["grid_half2"][str(cap)][
        "full"]["median_ms"])

    def native_routed_full():
        weights, ids = route(router_logits)
        return finish(w13[ids], w2[ids], weights)

    def half2_routed_full():
        weights, ids = route(router_logits)
        selected_w13, selected_w2 = extension.gather_half2(
            w13, w2, ids, best_cap)
        return finish(selected_w13, selected_w2, weights)

    routed_timings = {
        "native": measure(
            native_routed_full, args.warmup, args.iterations, args.repeats),
        "half2": measure(
            half2_routed_full, args.warmup, args.iterations, args.repeats),
    }
    routed_timings["half2"]["speedup_vs_native"] = (
        routed_timings["native"]["median_ms"] /
        routed_timings["half2"]["median_ms"])
    exact_steps = 0
    max_abs = 0.0
    for _ in range(args.sequence_steps):
        step_logits = torch.randn(
            (1, experts), device=device, dtype=dtype, generator=generator)
        weights, ids = route(step_logits)
        expected = finish(w13[ids], w2[ids], weights)
        selected_w13, selected_w2 = extension.gather_half2(
            w13, w2, ids, best_cap)
        actual = finish(selected_w13, selected_w2, weights)
        exact_steps += int(torch.equal(actual, expected))
        max_abs = max(max_abs, float((
            actual.float() - expected.float()).abs().max()))

    report = {
        "device": torch.cuda.get_device_name(device),
        "config": vars(args) | {
            "extension": str(args.extension),
            "reduce_extension": str(args.reduce_extension),
            "out": str(args.out),
        },
        "best_cap": best_cap,
        "sequence": {
            "exact_steps": exact_steps,
            "steps": args.sequence_steps,
            "max_abs": max_abs,
        },
        "routed_timings": routed_timings,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
