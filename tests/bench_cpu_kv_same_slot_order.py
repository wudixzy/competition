#!/usr/bin/env python3
"""Single-GPU data-plane gate for D2H-then-H2D reuse of one GPU slot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


NUM_ATTENTION_LAYERS = 10
NUM_BLOCKS = 2
BLOCK_SIZE = 16
LOCAL_NUM_KV_HEADS = 1
HEAD_SIZE = 256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from vllm.attention.ops.paged_attn import PagedAttention

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    shape = PagedAttention.get_kv_cache_shape(
        NUM_BLOCKS, BLOCK_SIZE, LOCAL_NUM_KV_HEADS, HEAD_SIZE)
    gpu_cache = [
        torch.empty(shape, dtype=torch.float16, device=device)
        for _ in range(NUM_ATTENTION_LAYERS)
    ]
    cpu_cache = [
        torch.empty(shape, dtype=torch.float16, device="cpu", pin_memory=True)
        for _ in range(NUM_ATTENTION_LAYERS)
    ]

    expected_victim = []
    expected_requested = []
    for layer, (gpu_layer, cpu_layer) in enumerate(
            zip(gpu_cache, cpu_cache)):
        gpu_layer.zero_()
        cpu_layer.zero_()
        gpu_layer[:, 1, :].fill_(float(100 + layer))
        cpu_layer[:, 0, :].fill_(float(200 + layer))
        expected_victim.append(gpu_layer[:, 1, :].cpu().clone())
        expected_requested.append(cpu_layer[:, 0, :].clone())
    torch.cuda.synchronize(device)

    d2h = torch.tensor([[1, 1]], dtype=torch.int64, device="cpu")
    h2d = torch.tensor([[0, 1]], dtype=torch.int64, device="cpu")

    # This is the exact ordering used by the patched Worker.execute_worker.
    for layer in range(NUM_ATTENTION_LAYERS):
        PagedAttention.swap_blocks(
            gpu_cache[layer], cpu_cache[layer], d2h)
    for layer in range(NUM_ATTENTION_LAYERS):
        PagedAttention.swap_blocks(
            cpu_cache[layer], gpu_cache[layer], h2d)
    torch.cuda.synchronize(device)

    victim_preserved = all(
        torch.equal(cpu_cache[layer][:, 1, :], expected_victim[layer])
        for layer in range(NUM_ATTENTION_LAYERS))
    requested_restored = all(
        torch.equal(gpu_cache[layer][:, 1, :].cpu(),
                    expected_requested[layer])
        for layer in range(NUM_ATTENTION_LAYERS))
    source_unchanged = all(
        torch.equal(cpu_cache[layer][:, 0, :], expected_requested[layer])
        for layer in range(NUM_ATTENTION_LAYERS))

    result = {
        "schema": "bi100-cpu-kv-same-slot-order-v1",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "shape": list(shape),
        "d2h": [[1, 1]],
        "h2d": [[0, 1]],
        "same_gpu_slot_reused": d2h[0, 0].item() == h2d[0, 1].item(),
        "victim_preserved": victim_preserved,
        "requested_restored": requested_restored,
        "inclusive_cpu_source_unchanged": source_unchanged,
    }
    result["qualified"] = all((
        result["same_gpu_slot_reused"],
        victim_preserved,
        requested_restored,
        source_unchanged,
    ))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
