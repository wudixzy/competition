#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def load_model_module():
    vllm_spec = importlib.util.find_spec("vllm")
    if vllm_spec is None or vllm_spec.submodule_search_locations is None:
        raise RuntimeError("vllm package not found")
    root = Path(next(iter(vllm_spec.submodule_search_locations)))
    path = root / "model_executor" / "models" / "qwen3_5.py"
    spec = importlib.util.spec_from_file_location("qwen3_5_qgkv_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tp-rank", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from vllm.model_executor.layers.linear import ReplicatedLinear

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    hidden_size, global_qg_dim, local_qg_dim, kv_dim = 2048, 8192, 2048, 512
    dtype = torch.float16
    module = load_model_module()
    module.get_tensor_model_parallel_world_size = lambda: 4
    module.get_tensor_model_parallel_rank = lambda: args.tp_rank

    def make_layer(output_size: int, prefix: str) -> ReplicatedLinear:
        return ReplicatedLinear(
            hidden_size, output_size, bias=False, params_dtype=dtype,
            quant_config=None, prefix=prefix).to(device)

    packed_layer = make_layer(local_qg_dim + 2 * kv_dim, "probe.qgkv")
    qg_layer = make_layer(local_qg_dim, "probe.qg")
    key_layer = make_layer(kv_dim, "probe.k")
    value_layer = make_layer(kv_dim, "probe.v")

    global_qg_weight = torch.randn(
        (global_qg_dim, hidden_size), device=device, dtype=dtype,
        generator=generator) * 0.02
    key_weight = torch.randn(
        (kv_dim, hidden_size), device=device, dtype=dtype,
        generator=generator) * 0.02
    value_weight = torch.randn(
        (kv_dim, hidden_size), device=device, dtype=dtype,
        generator=generator) * 0.02
    qg_start = args.tp_rank * local_qg_dim
    local_qg_weight = global_qg_weight[qg_start:qg_start + local_qg_dim]
    config = SimpleNamespace(
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=256,
    )
    params = {"model.layers.3.self_attn.qgkv_proj.weight":
              packed_layer.weight}
    loaded = []
    for source, weight in (("q_proj", global_qg_weight),
                           ("k_proj", key_weight),
                           ("v_proj", value_weight)):
        loaded.append(module._load_full_attention_qgkv_weight(
            params, f"model.layers.3.self_attn.{source}.weight",
            weight, config))
    qg_layer.weight_loader(qg_layer.weight, local_qg_weight)
    key_layer.weight_loader(key_layer.weight, key_weight)
    value_layer.weight_loader(value_layer.weight, value_weight)

    hidden = torch.randn(
        (1, hidden_size), device=device, dtype=dtype, generator=generator)
    projected, _ = packed_layer(hidden)
    actual_qg, actual_key, actual_value = torch.split(
        projected, (local_qg_dim, kv_dim, kv_dim), dim=-1)
    expected_qg, _ = qg_layer(hidden)
    expected_key, _ = key_layer(hidden)
    expected_value, _ = value_layer(hidden)
    torch.cuda.synchronize()

    expected_weights = (local_qg_weight, key_weight, value_weight)
    actual_weights = torch.split(
        packed_layer.weight, (local_qg_dim, kv_dim, kv_dim), dim=0)
    expected_outputs = (expected_qg, expected_key, expected_value)
    actual_outputs = (actual_qg, actual_key, actual_value)
    weight_checks = [bool(torch.equal(actual, expected))
                     for actual, expected in zip(actual_weights,
                                                 expected_weights)]
    output_checks = []
    for actual, expected in zip(actual_outputs, expected_outputs):
        difference = (actual.float() - expected.float()).abs()
        output_checks.append({
            "exact": bool(torch.equal(actual, expected)),
            "max_abs": float(difference.max()),
        })

    report = {
        "device": torch.cuda.get_device_name(device),
        "tp_rank": args.tp_rank,
        "loaded": [bool(value) for value in loaded],
        "weight_exact": weight_checks,
        "output_checks": output_checks,
    }
    report["ok"] = bool(
        all(report["loaded"])
        and all(report["weight_exact"])
        and all(check["exact"] for check in output_checks))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
