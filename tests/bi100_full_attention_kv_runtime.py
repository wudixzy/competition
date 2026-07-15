#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def load_model_module():
    vllm_spec = importlib.util.find_spec("vllm")
    if vllm_spec is None or vllm_spec.submodule_search_locations is None:
        raise RuntimeError("vllm package not found")
    root = Path(next(iter(vllm_spec.submodule_search_locations)))
    path = root / "model_executor" / "models" / "qwen3_5.py"
    spec = importlib.util.spec_from_file_location("qwen3_5_kv_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from vllm.model_executor.layers.linear import ReplicatedLinear

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    hidden_size, kv_dim = 2048, 512
    dtype = torch.float16
    module = load_model_module()
    layer = ReplicatedLinear(
        hidden_size, 2 * kv_dim, bias=False,
        params_dtype=dtype, quant_config=None,
        prefix="model.layers.3.self_attn.kv_proj").to(device)
    key_layer = ReplicatedLinear(
        hidden_size, kv_dim, bias=False,
        params_dtype=dtype, quant_config=None,
        prefix="model.layers.3.self_attn.k_proj").to(device)
    value_layer = ReplicatedLinear(
        hidden_size, kv_dim, bias=False,
        params_dtype=dtype, quant_config=None,
        prefix="model.layers.3.self_attn.v_proj").to(device)
    key_weight = torch.randn(
        (kv_dim, hidden_size), device=device, dtype=dtype,
        generator=generator) * 0.02
    value_weight = torch.randn(
        (kv_dim, hidden_size), device=device, dtype=dtype,
        generator=generator) * 0.02
    config = SimpleNamespace(num_key_value_heads=2, head_dim=256)
    params = {"model.layers.3.self_attn.kv_proj.weight": layer.weight}
    key_loaded = module._load_full_attention_kv_weight(
        params, "model.layers.3.self_attn.k_proj.weight",
        key_weight, config)
    value_loaded = module._load_full_attention_kv_weight(
        params, "model.layers.3.self_attn.v_proj.weight",
        value_weight, config)
    key_layer.weight_loader(key_layer.weight, key_weight)
    value_layer.weight_loader(value_layer.weight, value_weight)

    hidden = torch.randn(
        (1, hidden_size), device=device, dtype=dtype, generator=generator)
    merged, _ = layer(hidden)
    actual_key, actual_value = torch.split(merged, kv_dim, dim=-1)
    expected_key, _ = key_layer(hidden)
    expected_value, _ = value_layer(hidden)
    raw_key = F.linear(hidden, key_weight)
    raw_value = F.linear(hidden, value_weight)
    torch.cuda.synchronize()

    report = {
        "device": torch.cuda.get_device_name(device),
        "key_loaded": bool(key_loaded),
        "value_loaded": bool(value_loaded),
        "key_weight_exact": bool(torch.equal(layer.weight[:kv_dim],
                                              key_weight)),
        "value_weight_exact": bool(torch.equal(layer.weight[kv_dim:],
                                                value_weight)),
        "key_output_exact": bool(torch.equal(actual_key, expected_key)),
        "value_output_exact": bool(torch.equal(actual_value, expected_value)),
        "key_output_max_abs": float(
            (actual_key.float() - expected_key.float()).abs().max()),
        "value_output_max_abs": float(
            (actual_value.float() - expected_value.float()).abs().max()),
        "separate_key_vs_raw_exact": bool(torch.equal(expected_key, raw_key)),
        "separate_value_vs_raw_exact": bool(torch.equal(expected_value,
                                                          raw_value)),
    }
    report["ok"] = all(report[key] for key in (
        "key_loaded",
        "value_loaded",
        "key_weight_exact",
        "value_weight_exact",
        "key_output_exact",
        "value_output_exact",
    ))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
