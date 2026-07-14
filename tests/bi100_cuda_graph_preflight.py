#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.activation import SiluAndMul


def capture(call: Callable[[], torch.Tensor]) \
        -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = call()
    torch.cuda.synchronize()
    return graph, output


def elapsed_ms(call: Callable[[], None], iterations: int) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0 / iterations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--stack-layers", type=int, default=40)
    parser.add_argument("--stack-iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = torch.float16
    generator = torch.Generator(device=device).manual_seed(args.seed)

    hidden = torch.randn(
        (1, 2048), device=device, dtype=dtype, generator=generator)
    router_logits = torch.randn(
        (1, 256), device=device, dtype=dtype, generator=generator)
    w13 = torch.empty((256, 256, 2048), device=device, dtype=dtype)
    w2 = torch.empty((256, 2048, 128), device=device, dtype=dtype)
    w13.normal_(mean=0.0, std=0.02, generator=generator)
    w2.normal_(mean=0.0, std=0.02, generator=generator)
    activation = SiluAndMul()

    def moe_step() -> torch.Tensor:
        topk_logits, topk_ids = torch.topk(
            router_logits.float(), 8, dim=-1)
        weights = torch.softmax(topk_logits, dim=-1)[0].to(dtype)
        w13_sel = w13[topk_ids[0]]
        w2_sel = w2[topk_ids[0]]
        gate_up = F.linear(hidden, w13_sel.reshape(-1, 2048)).view(8, -1)
        act = activation(gate_up)
        expert_out = torch.bmm(w2_sel, act.unsqueeze(-1)).squeeze(-1)
        return (expert_out * weights.unsqueeze(-1)).sum(0, keepdim=True)

    eager_moe = moe_step()
    moe_graph, graph_moe = capture(moe_step)
    moe_graph.replay()
    torch.cuda.synchronize()
    moe_finite = bool(
        torch.isfinite(eager_moe).all() and torch.isfinite(graph_moe).all())
    moe_difference = (eager_moe.float() - graph_moe.float()).abs()
    eager_moe_ms = elapsed_ms(lambda: moe_step(), args.iterations)
    graph_moe_ms = elapsed_ms(moe_graph.replay, args.iterations)

    def moe_stack() -> torch.Tensor:
        output = None
        for _ in range(args.stack_layers):
            output = moe_step()
        assert output is not None
        return output

    eager_stack = moe_stack()
    stack_graph, graph_stack = capture(moe_stack)
    stack_graph.replay()
    torch.cuda.synchronize()
    stack_finite = bool(
        torch.isfinite(eager_stack).all()
        and torch.isfinite(graph_stack).all())
    stack_difference = (eager_stack.float() - graph_stack.float()).abs()
    eager_stack_ms = elapsed_ms(moe_stack, args.stack_iterations)
    graph_stack_ms = elapsed_ms(stack_graph.replay, args.stack_iterations)

    initial_state = torch.randn(
        (1, 12, 128, 128), device=device, dtype=torch.float32,
        generator=generator) * 0.01
    q = torch.randn(
        (1, 12, 128), device=device, dtype=torch.float32,
        generator=generator) * 0.01
    k = torch.randn(
        (1, 12, 128), device=device, dtype=torch.float32,
        generator=generator) * 0.01
    value = torch.randn(
        (1, 12, 128), device=device, dtype=torch.float32,
        generator=generator) * 0.01
    beta = torch.sigmoid(torch.randn(
        (1, 12), device=device, dtype=torch.float32, generator=generator))
    decay = torch.full((1, 12), 0.999, device=device, dtype=torch.float32)
    static_state = initial_state.clone()

    def gdn_step() -> torch.Tensor:
        static_state.mul_(decay[:, :, None, None])
        state = static_state.view(-1, 128, 128)
        key_memory = torch.bmm(k.view(-1, 1, 128), state).view(1, 12, 128)
        delta = (value - key_memory) * beta[:, :, None]
        state.baddbmm_(k.view(-1, 128, 1), delta.view(-1, 1, 128))
        return torch.bmm(q.view(-1, 1, 128), state).view(1, 12, 128)

    static_state.copy_(initial_state)
    eager_gdn = gdn_step().clone()
    eager_state = static_state.clone()
    static_state.copy_(initial_state)
    gdn_graph, graph_gdn = capture(gdn_step)
    static_state.copy_(initial_state)
    gdn_graph.replay()
    torch.cuda.synchronize()
    gdn_finite = bool(
        torch.isfinite(eager_gdn).all()
        and torch.isfinite(graph_gdn).all()
        and torch.isfinite(eager_state).all()
        and torch.isfinite(static_state).all())
    gdn_difference = (eager_gdn - graph_gdn).abs()
    state_difference = (eager_state - static_state).abs()

    report = {
        "device": torch.cuda.get_device_name(device),
        "moe": {
            "finite": moe_finite,
            "exact": bool(moe_finite and torch.equal(eager_moe, graph_moe)),
            "max_abs": float(moe_difference.max()),
            "eager_ms": eager_moe_ms,
            "graph_ms": graph_moe_ms,
            "speedup": eager_moe_ms / graph_moe_ms,
        },
        "moe_stack": {
            "layers": args.stack_layers,
            "finite": stack_finite,
            "exact": bool(
                stack_finite and torch.equal(eager_stack, graph_stack)),
            "max_abs": float(stack_difference.max()),
            "eager_ms": eager_stack_ms,
            "graph_ms": graph_stack_ms,
            "speedup": eager_stack_ms / graph_stack_ms,
        },
        "gdn_stateful": {
            "finite": gdn_finite,
            "output_exact": bool(
                gdn_finite and torch.equal(eager_gdn, graph_gdn)),
            "output_max_abs": float(gdn_difference.max()),
            "state_exact": bool(
                gdn_finite and torch.equal(eager_state, static_state)),
            "state_max_abs": float(state_difference.max()),
        },
    }
    report["ok"] = bool(
        report["moe"]["exact"]
        and report["moe_stack"]["exact"]
        and report["gdn_stateful"]["output_exact"]
        and report["gdn_stateful"]["state_exact"]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
