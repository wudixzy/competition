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
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--reduce-extension", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--sequence-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    gather = load_extension("corex_moe_weight_gather", args.extension)
    reducer = load_extension("corex_moe_exact_reduce", args.reduce_extension)
    namespace = {
        "torch": torch,
        "F": F,
        "_corex_moe_weight_gather": gather,
        "_corex_moe_exact_reduce": reducer,
        "_USE_COREX_MOE_WEIGHT_GATHER": False,
        "_USE_COREX_MOE_EXACT_REDUCE": True,
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
    fake_self = types.SimpleNamespace(
        top_k=top_k,
        experts=types.SimpleNamespace(w13_weight=w13, w2_weight=w2),
        act_fn=SiluAndMul(),
    )
    hidden_states = torch.randn(
        (1, hidden), device=device, dtype=dtype, generator=generator)
    router_logits = torch.randn(
        (1, experts), device=device, dtype=dtype, generator=generator)

    def run(enabled: bool, states=hidden_states, logits=router_logits):
        namespace["_USE_COREX_MOE_WEIGHT_GATHER"] = enabled
        return production(fake_self, states, logits)

    reference = run(False)
    candidate = run(True)
    native = measure(
        lambda: run(False), args.warmup, args.iterations, args.repeats)
    corex = measure(
        lambda: run(True), args.warmup, args.iterations, args.repeats)
    corex["speedup_vs_native"] = native["median_ms"] / corex["median_ms"]

    exact_steps = 0
    max_abs = 0.0
    for _ in range(args.sequence_steps):
        step_hidden = torch.randn(
            (1, hidden), device=device, dtype=dtype, generator=generator)
        step_logits = torch.randn(
            (1, experts), device=device, dtype=dtype, generator=generator)
        expected = run(False, step_hidden, step_logits)
        actual = run(True, step_hidden, step_logits)
        exact_steps += int(torch.equal(actual, expected))
        max_abs = max(
            max_abs,
            float((actual.float() - expected.float()).abs().max()),
        )

    report = {
        "device": torch.cuda.get_device_name(device),
        "config": vars(args) | {
            "model_source": str(args.model_source),
            "extension": str(args.extension),
            "reduce_extension": str(args.reduce_extension),
            "out": str(args.out),
        },
        "extension_loaded": True,
        "exact": bool(torch.equal(candidate, reference)),
        "max_abs": float((candidate.float() - reference.float()).abs().max()),
        "sequence": {
            "exact_steps": exact_steps,
            "steps": args.sequence_steps,
            "max_abs": max_abs,
        },
        "native": native,
        "corex": corex,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
