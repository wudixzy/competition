#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


Case = Callable[[], torch.Tensor]


def load_extension(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def measure(case: Case, warmup: int, iterations: int,
            repeats: int) -> dict[str, object]:
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
    return {
        "median_ms": statistics.median(trials),
        "p10_ms": percentile(trials, 10),
        "p90_ms": percentile(trials, 90),
        "trials_ms": trials,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal-conv-extension", type=Path, required=True)
    parser.add_argument("--gated-norm-extension", type=Path, required=True)
    parser.add_argument("--beta-decay-extension", type=Path, required=True)
    parser.add_argument("--qk-map-extension", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    causal_conv = load_extension(
        "corex_gdn_causal_conv", args.causal_conv_extension)
    gated_norm = load_extension(
        "corex_gdn_gated_norm", args.gated_norm_extension)
    beta_decay = load_extension(
        "corex_gdn_beta_decay", args.beta_decay_extension)
    qk_map = load_extension("corex_gdn_qk_map", args.qk_map_extension)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Qwen3.6-35B-A3B checkpoint dimensions after TP=4 sharding.
    batch = 1
    hidden_size = 2048
    local_key_dim = 512
    local_value_dim = 1024
    local_key_heads = 4
    local_value_heads = 8
    head_dim = 128
    local_conv_dim = 2048
    local_projection_dim = 3088
    head_expand_ratio = 2
    epsilon = 1e-6

    hidden = torch.randn(
        (batch, hidden_size), device=device, generator=generator,
        dtype=torch.float16) * 0.05
    input_weight = torch.randn(
        (local_projection_dim, hidden_size), device=device,
        generator=generator, dtype=torch.float16) * 0.02
    output_weight = torch.randn(
        (hidden_size, local_value_dim), device=device,
        generator=generator, dtype=torch.float16) * 0.02
    conv_weight = torch.randn(
        (local_conv_dim, 4), device=device, generator=generator,
        dtype=torch.float16) * 0.05
    norm_weight = torch.randn(
        (head_dim,), device=device, generator=generator,
        dtype=torch.float16) * 0.05
    a_log = torch.zeros(
        (local_value_heads,), device=device, dtype=torch.float16)
    dt_bias = torch.zeros(
        (local_value_heads,), device=device, dtype=torch.float16)

    projected = F.linear(hidden, input_weight)
    mixed, z_all, b_all, a_all = torch.split(
        projected, [local_conv_dim, local_value_dim,
                    local_value_heads, local_value_heads], dim=-1)
    mixed = mixed.unsqueeze(-1).contiguous()
    z = z_all.reshape(batch, local_value_heads, head_dim)

    initial_conv_state = torch.randn(
        (batch, local_conv_dim, 3), device=device, generator=generator,
        dtype=torch.float16).float() * 0.05
    initial_temporal_state = torch.randn(
        (batch, local_value_heads, head_dim, head_dim), device=device,
        generator=generator, dtype=torch.float32) * 0.001
    conv_state = initial_conv_state.clone()
    temporal_state = initial_temporal_state.clone()

    mixed_conv = causal_conv.causal_conv_update(
        conv_state, mixed, conv_weight)
    q_raw, k_raw, v_raw = torch.split(
        mixed_conv.squeeze(-1).unsqueeze(1),
        [local_key_dim, local_key_dim, local_value_dim], dim=-1)
    q_raw = q_raw.reshape(batch, 1, local_key_heads, head_dim)
    k_raw = k_raw.reshape(batch, 1, local_key_heads, head_dim)
    v_raw = v_raw.reshape(batch, 1, local_value_heads, head_dim)
    beta = b_all.sigmoid().unsqueeze(1)
    decay = (-a_log.float().exp()
             * F.softplus(a_all.float() + dt_bias)).unsqueeze(1)

    def prepare_qk() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = q_raw.repeat_interleave(head_expand_ratio, dim=2)
        k = k_raw.repeat_interleave(head_expand_ratio, dim=2)
        q = q.squeeze(1)
        k = k.squeeze(1)
        q_t = (q * torch.rsqrt(
            (q * q).sum(dim=-1, keepdim=True) + 1e-6)).float()
        q_t.mul_(head_dim ** -0.5)
        k_t = (k * torch.rsqrt(
            (k * k).sum(dim=-1, keepdim=True) + 1e-6)).float()
        return q_t, k_t, v_raw.squeeze(1).float()

    q_t, k_t, v_t = prepare_qk()
    g_t = decay.squeeze(1).float().exp()
    beta_t = beta.squeeze(1).float()

    def recurrent(state: torch.Tensor) -> torch.Tensor:
        state.mul_(g_t[:, :, None, None])
        flat = state.view(-1, head_dim, head_dim)
        bh = flat.shape[0]
        memory = torch.bmm(
            k_t.view(bh, 1, head_dim), flat).view(
                batch, local_value_heads, head_dim)
        delta = (v_t - memory) * beta_t[:, :, None]
        flat.baddbmm_(
            k_t.view(bh, head_dim, 1), delta.view(bh, 1, head_dim))
        return torch.bmm(q_t.view(bh, 1, head_dim), flat).view(
            batch, local_value_heads, head_dim)

    core = recurrent(temporal_state)

    def norm(current_core: torch.Tensor = core) -> torch.Tensor:
        core_2d = current_core.reshape(-1, head_dim)
        inverse = torch.rsqrt(
            core_2d.pow(2).mean(-1, keepdim=True) + epsilon)
        return gated_norm.apply_inverse(
            core_2d, z.reshape(-1, head_dim), norm_weight, inverse)

    normed = norm()

    timing_conv_state = initial_conv_state.clone()
    timing_temporal_state = initial_temporal_state.clone()

    def input_projection() -> torch.Tensor:
        return F.linear(hidden, input_weight)

    def causal_conv_stage() -> torch.Tensor:
        return causal_conv.causal_conv_update(
            timing_conv_state, mixed, conv_weight)

    def beta_decay_stage() -> torch.Tensor:
        b_all.sigmoid()
        current_decay = (-a_log.float().exp()
                         * F.softplus(a_all.float() + dt_bias))
        return current_decay

    def beta_decay_current_stage() -> torch.Tensor:
        return beta_decay.beta_decay(b_all, a_all, a_log, dt_bias)

    def qk_prep_stage() -> torch.Tensor:
        current_q, _, _ = prepare_qk()
        return current_q

    def qk_prep_current_stage() -> torch.Tensor:
        current_q = q_raw.squeeze(1)
        current_k = k_raw.squeeze(1)
        normalized_q = current_q * torch.rsqrt(
            (current_q * current_q).sum(dim=-1, keepdim=True) + 1e-6)
        normalized_k = current_k * torch.rsqrt(
            (current_k * current_k).sum(dim=-1, keepdim=True) + 1e-6)
        return qk_map.qk_map(
            normalized_q, normalized_k, local_value_heads)

    def recurrent_stage() -> torch.Tensor:
        return recurrent(timing_temporal_state)

    def gated_norm_stage() -> torch.Tensor:
        return norm()

    def output_projection() -> torch.Tensor:
        return F.linear(normed.reshape(batch, -1), output_weight)

    reference_conv_state = initial_conv_state.clone()
    reference_temporal_state = initial_temporal_state.clone()
    current_conv_state = initial_conv_state.clone()
    current_temporal_state = initial_temporal_state.clone()

    def local_full_decode(conv_state_arg: torch.Tensor,
                          temporal_state_arg: torch.Tensor,
                          optimized: bool) -> torch.Tensor:
        current = F.linear(hidden, input_weight)
        current_mixed, current_z, current_b, current_a = torch.split(
            current, [local_conv_dim, local_value_dim,
                      local_value_heads, local_value_heads], dim=-1)
        current_mixed = causal_conv.causal_conv_update(
            conv_state_arg, current_mixed.unsqueeze(-1), conv_weight)
        current_q, current_k, current_v = torch.split(
            current_mixed.squeeze(-1).unsqueeze(1),
            [local_key_dim, local_key_dim, local_value_dim], dim=-1)
        current_q = current_q.reshape(batch, local_key_heads, head_dim)
        current_k = current_k.reshape(batch, local_key_heads, head_dim)
        current_v = current_v.reshape(
            batch, local_value_heads, head_dim).float()
        if optimized:
            normalized_q = current_q * torch.rsqrt(
                (current_q * current_q).sum(dim=-1, keepdim=True) + 1e-6)
            normalized_k = current_k * torch.rsqrt(
                (current_k * current_k).sum(dim=-1, keepdim=True) + 1e-6)
            mapped = qk_map.qk_map(
                normalized_q, normalized_k, local_value_heads)
            current_q, current_k = mapped[0], mapped[1]
            prepared = beta_decay.beta_decay(
                current_b, current_a, a_log, dt_bias)
            current_beta, current_g = prepared[0], prepared[1]
        else:
            current_q = current_q.repeat_interleave(
                head_expand_ratio, dim=1)
            current_k = current_k.repeat_interleave(
                head_expand_ratio, dim=1)
            current_q = (current_q * torch.rsqrt(
                (current_q * current_q).sum(dim=-1, keepdim=True)
                + 1e-6)).float()
            current_q.mul_(head_dim ** -0.5)
            current_k = (current_k * torch.rsqrt(
                (current_k * current_k).sum(dim=-1, keepdim=True)
                + 1e-6)).float()
            current_g = (-a_log.float().exp() * F.softplus(
                current_a.float() + dt_bias)).float().exp()
            current_beta = current_b.sigmoid().float()
        temporal_state_arg.mul_(current_g[:, :, None, None])
        flat = temporal_state_arg.view(-1, head_dim, head_dim)
        bh = flat.shape[0]
        memory = torch.bmm(
            current_k.view(bh, 1, head_dim), flat).view(
                batch, local_value_heads, head_dim)
        delta = (current_v - memory) * current_beta[:, :, None]
        flat.baddbmm_(
            current_k.view(bh, head_dim, 1),
            delta.view(bh, 1, head_dim))
        current_core = torch.bmm(
            current_q.view(bh, 1, head_dim), flat).view(-1, head_dim)
        inverse = torch.rsqrt(
            current_core.pow(2).mean(-1, keepdim=True) + epsilon)
        current_norm = gated_norm.apply_inverse(
            current_core, current_z.reshape(-1, head_dim),
            norm_weight, inverse)
        return F.linear(current_norm.reshape(batch, -1), output_weight)

    def local_full_reference() -> torch.Tensor:
        return local_full_decode(
            reference_conv_state, reference_temporal_state, False)

    def local_full_current() -> torch.Tensor:
        return local_full_decode(
            current_conv_state, current_temporal_state, True)

    check_reference_conv = initial_conv_state.clone()
    check_reference_temporal = initial_temporal_state.clone()
    check_current_conv = initial_conv_state.clone()
    check_current_temporal = initial_temporal_state.clone()
    check_reference_output = local_full_decode(
        check_reference_conv, check_reference_temporal, False)
    check_current_output = local_full_decode(
        check_current_conv, check_current_temporal, True)
    torch.cuda.synchronize()

    cases = {
        "input_projection": input_projection,
        "causal_conv": causal_conv_stage,
        "beta_decay_reference": beta_decay_stage,
        "beta_decay_current": beta_decay_current_stage,
        "qk_prep_reference": qk_prep_stage,
        "qk_prep_current": qk_prep_current_stage,
        "recurrent": recurrent_stage,
        "gated_norm": gated_norm_stage,
        "output_projection": output_projection,
        "local_full_reference": local_full_reference,
        "local_full_current": local_full_current,
    }
    timings = {
        name: measure(case, args.warmup, args.iterations, args.repeats)
        for name, case in cases.items()
    }
    common_stages = [
        "input_projection", "causal_conv", "recurrent",
        "gated_norm", "output_projection"]
    reference_stages = common_stages + [
        "beta_decay_reference", "qk_prep_reference"]
    current_stages = common_stages + [
        "beta_decay_current", "qk_prep_current"]
    reference_isolated_sum = sum(
        timings[name]["median_ms"] for name in reference_stages)
    current_isolated_sum = sum(
        timings[name]["median_ms"] for name in current_stages)
    for name in reference_stages:
        timings[name]["reference_isolated_share_pct"] = (
            timings[name]["median_ms"] / reference_isolated_sum * 100.0)
    for name in current_stages:
        timings[name]["current_isolated_share_pct"] = (
            timings[name]["median_ms"] / current_isolated_sum * 100.0)

    output = local_full_current()
    torch.cuda.synchronize()
    report = {
        "device": torch.cuda.get_device_name(device),
        "scope": "one TP4 rank; communication/all-reduce excluded",
        "shape": {
            "batch": batch,
            "hidden_size": hidden_size,
            "local_projection_dim": local_projection_dim,
            "local_conv_dim": local_conv_dim,
            "local_key_heads": local_key_heads,
            "local_value_heads": local_value_heads,
            "head_dim": head_dim,
            "local_value_dim": local_value_dim,
        },
        "config": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "checks": {
            "output_shape": list(output.shape),
            "output_finite": bool(torch.isfinite(output).all()),
            "combined_output_exact": bool(torch.equal(
                check_current_output, check_reference_output)),
            "combined_conv_state_exact": bool(torch.equal(
                check_current_conv, check_reference_conv)),
            "combined_temporal_state_exact": bool(torch.equal(
                check_current_temporal, check_reference_temporal)),
            "combined_output_max_abs": float((
                check_current_output - check_reference_output).abs().max()),
            "combined_temporal_state_max_abs": float((
                check_current_temporal
                - check_reference_temporal).abs().max()),
        },
        "reference_isolated_sum_ms": reference_isolated_sum,
        "current_isolated_sum_ms": current_isolated_sum,
        "reference_composition_overhead_ms": (
            timings["local_full_reference"]["median_ms"]
            - reference_isolated_sum),
        "current_composition_overhead_ms": (
            timings["local_full_current"]["median_ms"]
            - current_isolated_sum),
        "current_speedup_vs_reference": (
            timings["local_full_reference"]["median_ms"]
            / timings["local_full_current"]["median_ms"]),
        "current_saving_ms": (
            timings["local_full_reference"]["median_ms"]
            - timings["local_full_current"]["median_ms"]),
        "timings": timings,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if all([
        report["checks"]["output_finite"],
        report["checks"]["combined_output_exact"],
        report["checks"]["combined_conv_state_exact"],
        report["checks"]["combined_temporal_state_exact"],
    ]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
