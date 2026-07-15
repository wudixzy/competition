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
    parser.add_argument("--sequence-steps", type=int, default=1000)
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

    def route_baseline(logits):
        selected, ids = torch.topk(logits.float(), top_k, dim=-1)
        weights = torch.softmax(selected, dim=-1).to(dtype)
        return weights[0], ids[0]

    def route_candidate(logits):
        selected, ids = torch.topk(logits, top_k, dim=-1)
        weights = torch.softmax(selected.float(), dim=-1).to(dtype)
        return weights[0], ids[0]

    def finish(weights, ids):
        selected_w13, selected_w2 = gather.gather(w13, w2, ids)
        gate_up = F.linear(
            hidden_states, selected_w13.reshape(-1, hidden)).view(top_k, -1)
        activated = activation(gate_up)
        expert_output = torch.bmm(
            selected_w2, activated.unsqueeze(-1)).squeeze(-1)
        return reducer.serial_float(expert_output, weights)

    def run(route):
        weights, ids = route(router_logits)
        return finish(weights, ids)

    route_timings = {
        "baseline": measure(
            lambda: route_baseline(router_logits),
            args.warmup, args.iterations, args.repeats),
        "candidate": measure(
            lambda: route_candidate(router_logits),
            args.warmup, args.iterations, args.repeats),
    }
    full_timings = {
        "baseline": measure(
            lambda: run(route_baseline),
            args.warmup, args.iterations, args.repeats),
        "candidate": measure(
            lambda: run(route_candidate),
            args.warmup, args.iterations, args.repeats),
    }
    route_timings["candidate"]["speedup_vs_baseline"] = (
        route_timings["baseline"]["median_ms"] /
        route_timings["candidate"]["median_ms"])
    full_timings["candidate"]["speedup_vs_baseline"] = (
        full_timings["baseline"]["median_ms"] /
        full_timings["candidate"]["median_ms"])

    edge_cases = []
    edge_cases.append(torch.zeros(
        (1, experts), device=device, dtype=dtype))
    tied = torch.arange(experts, device=device, dtype=dtype).view(1, -1)
    tied[:, 240:] = 100
    edge_cases.append(tied)
    alternating = torch.empty(
        (1, experts), device=device, dtype=dtype)
    alternating[:, 0::2] = 32
    alternating[:, 1::2] = -32
    edge_cases.append(alternating)

    ids_exact = 0
    weights_exact = 0
    output_exact = 0
    max_abs = 0.0
    cases = edge_cases
    cases.extend(
        torch.randn(
            (1, experts), device=device, dtype=dtype, generator=generator)
        * scale
        for scale in (1.0, 8.0, 64.0)
        for _ in range(args.sequence_steps)
    )
    for logits in cases:
        baseline_weights, baseline_ids = route_baseline(logits)
        candidate_weights, candidate_ids = route_candidate(logits)
        expected = finish(baseline_weights, baseline_ids)
        actual = finish(candidate_weights, candidate_ids)
        ids_exact += int(torch.equal(candidate_ids, baseline_ids))
        weights_exact += int(torch.equal(candidate_weights, baseline_weights))
        output_exact += int(torch.equal(actual, expected))
        max_abs = max(
            max_abs, float((actual.float() - expected.float()).abs().max()))

    report = {
        "device": torch.cuda.get_device_name(device),
        "config": vars(args) | {
            "gather_extension": str(args.gather_extension),
            "reduce_extension": str(args.reduce_extension),
            "out": str(args.out),
        },
        "route": route_timings,
        "full": full_timings,
        "sequence": {
            "cases": len(cases),
            "ids_exact": ids_exact,
            "weights_exact": weights_exact,
            "output_exact": output_exact,
            "max_abs": max_abs,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
