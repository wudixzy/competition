#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F


def bench(fn, *, warmups: int, repeats: int, iterations: int) -> list[float]:
    result = None
    for _ in range(warmups):
        result = fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(iterations):
            result = fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0 / iterations)
    assert result is not None
    return samples


def summarize(samples: list[float]) -> dict[str, float | list[float]]:
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-label", required=True)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(20260715)
    device = torch.device("cuda:0")
    dtype = torch.float16
    hidden_size = 2048
    num_experts = 256
    top_k = 8

    router_weight = torch.randn(
        num_experts, hidden_size, device=device, dtype=dtype) * 0.01
    shared_gate_weight = torch.randn(
        1, hidden_size, device=device, dtype=dtype) * 0.01
    fused_weight = torch.cat([router_weight, shared_gate_weight], dim=0)

    def current_projection(x: torch.Tensor):
        return F.linear(x, router_weight), F.linear(x, shared_gate_weight)

    def fused_projection(x: torch.Tensor):
        projected = F.linear(x, fused_weight)
        return projected[..., :num_experts], projected[..., num_experts:]

    def route(router_logits: torch.Tensor, gate_score: torch.Tensor):
        routing_weights = torch.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(
            routing_weights, top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(
            dim=-1, keepdim=True)
        return topk_weights.to(dtype), topk_ids, torch.sigmoid(gate_score)

    def current_pipeline(x: torch.Tensor):
        return route(*current_projection(x))

    def fused_pipeline(x: torch.Tensor):
        return route(*fused_projection(x))

    results = {}
    for tokens, iterations in ((1, 500), (64, 50)):
        x = torch.randn(tokens, hidden_size, device=device, dtype=dtype)
        current_logits, current_gate = current_projection(x)
        fused_logits, fused_gate = fused_projection(x)
        current_weights, current_ids, current_sigmoid = current_pipeline(x)
        fused_weights, fused_ids, fused_sigmoid = fused_pipeline(x)

        parity = {
            "router_logits_max_abs": float(
                (current_logits - fused_logits).abs().max().item()),
            "shared_gate_max_abs": float(
                (current_gate - fused_gate).abs().max().item()),
            "topk_weights_max_abs": float(
                (current_weights - fused_weights).abs().max().item()),
            "shared_sigmoid_max_abs": float(
                (current_sigmoid - fused_sigmoid).abs().max().item()),
            "topk_ids_equal": bool(torch.equal(current_ids, fused_ids)),
            "all_finite": all(bool(torch.isfinite(value).all().item()) for value in (
                current_logits, current_gate, fused_logits, fused_gate,
                current_weights, fused_weights, current_sigmoid, fused_sigmoid,
            )),
        }

        timings = {}
        functions = {
            "current_projection": lambda: current_projection(x),
            "fused_projection": lambda: fused_projection(x),
            "current_pipeline": lambda: current_pipeline(x),
            "fused_pipeline": lambda: fused_pipeline(x),
        }
        for name, fn in functions.items():
            timings[name] = summarize(bench(
                fn,
                warmups=args.warmups,
                repeats=args.repeats,
                iterations=iterations,
            ))

        projection_baseline = float(timings["current_projection"]["median_ms"])
        pipeline_baseline = float(timings["current_pipeline"]["median_ms"])
        timings["fused_projection"]["speedup_vs_current"] = (
            projection_baseline
            / float(timings["fused_projection"]["median_ms"]))
        timings["fused_pipeline"]["speedup_vs_current"] = (
            pipeline_baseline
            / float(timings["fused_pipeline"]["median_ms"]))

        results[f"t{tokens}"] = {
            "iterations": iterations,
            "parity": parity,
            "timings": timings,
        }

    print(json.dumps({
        "device": args.device_label,
        "shape": {
            "hidden_size": hidden_size,
            "num_experts": num_experts,
            "top_k": top_k,
            "router_output_size": num_experts,
            "shared_gate_output_size": 1,
            "fused_output_size": num_experts + 1,
            "dtype": str(dtype),
        },
        "results": results,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
