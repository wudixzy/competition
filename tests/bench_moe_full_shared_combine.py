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
    parser.add_argument("--gather-extension", type=Path, required=True)
    parser.add_argument("--reduce-extension", type=Path, required=True)
    parser.add_argument("--combine-extension", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--random-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    gather = load_extension(
        "corex_moe_weight_gather", args.gather_extension)
    reducer = load_extension("corex_moe_exact_reduce", args.reduce_extension)
    combine = load_extension(
        "corex_moe_shared_combine", args.combine_extension)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    experts, top_k = 256, 8
    hidden, routed_intermediate, shared_intermediate = 2048, 128, 128
    dtype = torch.float16
    router_shared_weight = torch.randn(
        (experts + 1, hidden), device=device, dtype=dtype,
        generator=generator) * 0.02
    w13 = torch.randn(
        (experts, 2 * routed_intermediate, hidden), device=device,
        dtype=dtype, generator=generator) * 0.02
    w2 = torch.randn(
        (experts, hidden, routed_intermediate), device=device,
        dtype=dtype, generator=generator) * 0.02
    shared_gate_up_weight = torch.randn(
        (2 * shared_intermediate, hidden), device=device, dtype=dtype,
        generator=generator) * 0.02
    shared_down_weight = torch.randn(
        (hidden, shared_intermediate), device=device, dtype=dtype,
        generator=generator) * 0.02
    activation = SiluAndMul()

    def common(current_hidden: torch.Tensor):
        router_and_gate = F.linear(current_hidden, router_shared_weight)
        router_logits = router_and_gate[..., :experts]
        gate_score = router_and_gate[..., experts:]
        topk_logits, topk_ids = torch.topk(
            router_logits.float(), top_k, dim=-1)
        topk_weights = torch.softmax(topk_logits, dim=-1)[0].to(dtype)
        selected_w13, selected_w2 = gather.gather(
            w13, w2, topk_ids[0])
        gate_up = F.linear(
            current_hidden,
            selected_w13.reshape(-1, hidden)).view(top_k, -1)
        routed_activation = activation(gate_up)
        expert_out = torch.bmm(
            selected_w2, routed_activation.unsqueeze(-1)).squeeze(-1)
        routed_out = reducer.serial_float(expert_out, topk_weights)

        shared_gate_up = F.linear(current_hidden, shared_gate_up_weight)
        shared_activation = activation(shared_gate_up)
        shared_out = F.linear(shared_activation, shared_down_weight)
        return routed_out, shared_out, gate_score

    def reference(current_hidden: torch.Tensor):
        routed_out, shared_out, gate_score = common(current_hidden)
        return routed_out + shared_out * torch.sigmoid(gate_score)

    def candidate(current_hidden: torch.Tensor):
        routed_out, shared_out, gate_score = common(current_hidden)
        return combine.shared_combine(routed_out, shared_out, gate_score)

    hidden_states = torch.randn(
        (1, hidden), device=device, dtype=dtype, generator=generator)
    expected = reference(hidden_states)
    actual = candidate(hidden_states)
    exact_steps = 0
    max_abs = 0.0
    for _ in range(args.random_steps):
        step_hidden = torch.randn(
            hidden_states.shape, device=device, dtype=dtype,
            generator=generator)
        step_expected = reference(step_hidden)
        step_actual = candidate(step_hidden)
        exact_steps += int(torch.equal(step_actual, step_expected))
        max_abs = max(max_abs, float(
            (step_actual - step_expected).abs().max()))

    timings = {
        "reference_full": measure(
            lambda: reference(hidden_states), args.warmup,
            args.iterations, args.repeats),
        "candidate_full": measure(
            lambda: candidate(hidden_states), args.warmup,
            args.iterations, args.repeats),
    }
    reference_ms = timings["reference_full"]["median_ms"]
    candidate_ms = timings["candidate_full"]["median_ms"]
    report = {
        "device": torch.cuda.get_device_name(device),
        "shape": {
            "experts": experts,
            "top_k": top_k,
            "hidden": hidden,
            "routed_intermediate_tp4": routed_intermediate,
            "shared_intermediate_tp4": shared_intermediate,
        },
        "config": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "random_steps": args.random_steps,
            "seed": args.seed,
        },
        "one_step": {
            "exact": bool(torch.equal(actual, expected)),
            "max_abs": float((actual - expected).abs().max()),
        },
        "random": {
            "steps": args.random_steps,
            "exact_steps": exact_steps,
            "max_abs": max_abs,
        },
        "timings": timings,
        "speedup": reference_ms / candidate_ms,
        "saving_ms_per_layer": reference_ms - candidate_ms,
        "projected_saving_ms_40_layers": 40.0 * (
            reference_ms - candidate_ms),
    }
    report["ok"] = bool(
        report["one_step"]["exact"]
        and exact_steps == args.random_steps
        and report["speedup"] > 1.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
