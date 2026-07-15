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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    gather = load_extension(
        "corex_moe_weight_gather", args.gather_extension)
    reducer = load_extension("corex_moe_exact_reduce", args.reduce_extension)
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

    def route():
        selected, ids = torch.topk(router_logits.float(), top_k, dim=-1)
        weights = torch.softmax(selected, dim=-1)[0].to(dtype)
        return weights, ids[0]

    weights, ids = route()
    selected_w13, selected_w2 = gather.gather(w13, w2, ids)
    gate_up = F.linear(
        hidden_states, selected_w13.reshape(-1, hidden)).view(top_k, -1)
    activated = activation(gate_up)
    expert_output = torch.bmm(
        selected_w2, activated.unsqueeze(-1)).squeeze(-1)

    def w13_linear():
        return F.linear(
            hidden_states, selected_w13.reshape(-1, hidden)).view(top_k, -1)

    def w2_bmm():
        return torch.bmm(
            selected_w2, activated.unsqueeze(-1)).squeeze(-1)

    def compute():
        current_gate_up = F.linear(
            hidden_states, selected_w13.reshape(-1, hidden)).view(top_k, -1)
        current_activation = activation(current_gate_up)
        current_output = torch.bmm(
            selected_w2, current_activation.unsqueeze(-1)).squeeze(-1)
        return reducer.serial_float(current_output, weights)

    def fixed_full():
        current_w13, current_w2 = gather.gather(w13, w2, ids)
        current_gate_up = F.linear(
            hidden_states, current_w13.reshape(-1, hidden)).view(top_k, -1)
        current_activation = activation(current_gate_up)
        current_output = torch.bmm(
            current_w2, current_activation.unsqueeze(-1)).squeeze(-1)
        return reducer.serial_float(current_output, weights)

    def routed_full():
        current_weights, current_ids = route()
        current_w13, current_w2 = gather.gather(w13, w2, current_ids)
        current_gate_up = F.linear(
            hidden_states, current_w13.reshape(-1, hidden)).view(top_k, -1)
        current_activation = activation(current_gate_up)
        current_output = torch.bmm(
            current_w2, current_activation.unsqueeze(-1)).squeeze(-1)
        return reducer.serial_float(current_output, current_weights)

    cases = {
        "route": route,
        "gather": lambda: gather.gather(w13, w2, ids),
        "w13_linear": w13_linear,
        "activation": lambda: activation(gate_up),
        "w2_bmm": w2_bmm,
        "reduce": lambda: reducer.serial_float(expert_output, weights),
        "compute": compute,
        "fixed_full": fixed_full,
        "routed_full": routed_full,
    }
    results = {
        name: measure(case, args.warmup, args.iterations, args.repeats)
        for name, case in cases.items()
    }
    routed_ms = results["routed_full"]["median_ms"]
    for value in results.values():
        value["share_of_routed"] = value["median_ms"] / routed_ms

    reference = fixed_full()
    candidate = routed_full()
    report = {
        "device": torch.cuda.get_device_name(device),
        "config": vars(args) | {
            "gather_extension": str(args.gather_extension),
            "reduce_extension": str(args.reduce_extension),
            "out": str(args.out),
        },
        "exact": bool(torch.equal(candidate, reference)),
        "max_abs": float((candidate.float() - reference.float()).abs().max()),
        "results": results,
        "inferred": {
            "composition_overhead_ms": (
                results["routed_full"]["median_ms"]
                - results["route"]["median_ms"]
                - results["gather"]["median_ms"]
                - results["compute"]["median_ms"]
            ),
            "isolated_compute_sum_ms": sum(
                results[name]["median_ms"]
                for name in ("w13_linear", "activation", "w2_bmm", "reduce")
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
