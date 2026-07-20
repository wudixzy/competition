#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from test_gdn_parity import _load_production_chunk_rule


def compare_split(total: int, split: int, seed: int,
                  device: torch.device) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (1, total, 8, 128)
    query = torch.randn(
        shape, device=device, dtype=torch.float16, generator=generator)
    key = torch.randn(
        shape, device=device, dtype=torch.float16, generator=generator)
    value = torch.randn(
        shape, device=device, dtype=torch.float16, generator=generator)
    g = -torch.rand(
        (1, total, 8), device=device, dtype=torch.float32,
        generator=generator)
    beta = torch.rand(
        (1, total, 8), device=device, dtype=torch.float16,
        generator=generator)
    initial_state = torch.randn(
        (1, 8, 128, 128), device=device, dtype=torch.float32,
        generator=generator)
    rule = _load_production_chunk_rule()

    full_output, full_state = rule(
        query, key, value, g, beta, chunk_size=64,
        initial_state=initial_state, output_final_state=True,
        use_qk_l2norm_in_kernel=True)
    prefix_output, boundary_state = rule(
        query[:, :split], key[:, :split], value[:, :split],
        g[:, :split], beta[:, :split], chunk_size=64,
        initial_state=initial_state, output_final_state=True,
        use_qk_l2norm_in_kernel=True)
    suffix_output, split_state = rule(
        query[:, split:], key[:, split:], value[:, split:],
        g[:, split:], beta[:, split:], chunk_size=64,
        initial_state=boundary_state, output_final_state=True,
        use_qk_l2norm_in_kernel=True)
    split_output = torch.cat([prefix_output, suffix_output], dim=1)
    torch.cuda.synchronize(device)

    return {
        "total_tokens": total,
        "split_tokens": split,
        "native_chunk_aligned": split % 64 == 0,
        "output_equal": torch.equal(split_output, full_output),
        "state_equal": torch.equal(split_state, full_state),
        "output_max_abs": float(
            (split_output - full_output).abs().max().item()),
        "state_max_abs": float((split_state - full_state).abs().max().item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CoreX CUDA device is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    cases = (
        (193, 64),
        (193, 128),
        (2401, 2368),
        (2401, 2400),
        (4097, 4096),
    )
    results = [
        compare_split(total, split, args.seed + index, device)
        for index, (total, split) in enumerate(cases)
    ]
    native_results = [
        result for result in results if result["native_chunk_aligned"]]
    report = {
        "device": str(device),
        "seed": args.seed,
        "results": results,
        "native_chunk_exact": all(
            result["output_equal"] and result["state_equal"]
            for result in native_results),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["native_chunk_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
