#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F


CHUNK_SIZE = 64
NEUMANN_ORDER = 3
RESIDUAL_TERMS = 8
QUALIFICATION_LENGTH = 4096
MIN_INVERSE_SPEEDUP = 2.0
MIN_COMPLETE_SPEEDUP = 1.5
MAX_ABS_LIMIT = 1.0e-3
RELATIVE_L2_LIMIT = 1.0e-5

TensorPair = tuple[torch.Tensor, torch.Tensor]
InverseBuilder = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def l2norm(x: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def row_loop_inverse(attn: torch.Tensor,
                     identity: torch.Tensor) -> torch.Tensor:
    chunk_size = attn.shape[-1]
    for row_index in range(1, chunk_size):
        row = attn[..., row_index, :row_index].clone()
        submatrix = attn[..., :row_index, :row_index].clone()
        attn[..., row_index, :row_index] = (
            row + (row.unsqueeze(-1) * submatrix).sum(-2))
    return attn + identity


def masked_neumann_residual_inverse(
    attn: torch.Tensor,
    identity: torch.Tensor,
) -> torch.Tensor:
    """Apply the fixed N=3, S=8 construction from arXiv:2606.06034."""
    identity_batch = identity.expand_as(attn)
    power = identity_batch
    initial = identity_batch
    for _ in range(NEUMANN_ORDER):
        power = power @ attn
        initial = initial + power

    indices = torch.arange(
        attn.shape[-1], device=attn.device, dtype=torch.int64)
    diagonal_distance = indices[:, None] - indices[None, :]
    diagonal_mask = (
        (diagonal_distance >= 0)
        & (diagonal_distance <= NEUMANN_ORDER)
    )
    initial = initial * diagonal_mask.to(initial.dtype)

    system = identity_batch - attn
    residual = identity_batch - system @ initial
    correction = identity_batch
    power = identity_batch
    for _ in range(RESIDUAL_TERMS):
        power = power @ residual
        correction = correction + power
    return initial @ correction


def prepare_inverse_input(
    key: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = l2norm(key).transpose(1, 2).contiguous().to(torch.float32)
    beta = beta.transpose(1, 2).contiguous().to(torch.float32)
    g = g.transpose(1, 2).contiguous().to(torch.float32)
    seq_len = key.shape[2]
    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    key = F.pad(key, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad))
    g = F.pad(g, (0, pad))
    k_beta = key * beta.unsqueeze(-1)
    key = key.reshape(
        key.shape[0], key.shape[1], -1, chunk_size, key.shape[-1])
    k_beta = k_beta.reshape(
        k_beta.shape[0], k_beta.shape[1], -1, chunk_size,
        k_beta.shape[-1])
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size).cumsum(-1)
    decay = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float())
    upper = torch.triu(
        torch.ones(
            chunk_size, chunk_size, dtype=torch.bool, device=key.device),
        diagonal=0,
    )
    attn = -((k_beta @ key.transpose(-1, -2)) * decay).masked_fill(
        upper, 0)
    identity = torch.eye(
        chunk_size, dtype=attn.dtype, device=attn.device)
    return attn, identity


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    inverse_builder: InverseBuilder,
    chunk_size: int = CHUNK_SIZE,
    initial_state: Optional[torch.Tensor] = None,
) -> TensorPair:
    query = l2norm(query)
    key = l2norm(key)
    query, key, value, beta, g = [
        tensor.transpose(1, 2).contiguous().to(torch.float32)
        for tensor in (query, key, value, beta, g)
    ]
    batch, num_heads, seq_len, key_dim = key.shape
    value_dim = value.shape[-1]
    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad))
    key = F.pad(key, (0, 0, 0, pad))
    value = F.pad(value, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad))
    g = F.pad(g, (0, pad))
    total_len = seq_len + pad
    query = query * (query.shape[-1] ** -0.5)

    value_beta = value * beta.unsqueeze(-1)
    key_beta = key * beta.unsqueeze(-1)
    query, key, value, key_beta, value_beta = [
        tensor.reshape(
            tensor.shape[0], tensor.shape[1], -1, chunk_size,
            tensor.shape[-1])
        for tensor in (query, key, value, key_beta, value_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size).cumsum(-1)
    upper_with_diagonal = torch.triu(
        torch.ones(
            chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )
    decay = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float())
    attn = -((key_beta @ key.transpose(-1, -2)) * decay).masked_fill(
        upper_with_diagonal, 0)
    identity = torch.eye(
        chunk_size, dtype=attn.dtype, device=attn.device)
    inverse = inverse_builder(attn, identity)
    value = inverse @ value_beta
    key_cumulative_decay = inverse @ (
        key_beta * g.exp().unsqueeze(-1))

    last_state = (
        torch.zeros(
            batch, num_heads, key_dim, value_dim, dtype=value.dtype,
            device=value.device)
        if initial_state is None else initial_state.to(value)
    )
    core_output = torch.zeros_like(value)
    strict_upper = torch.triu(
        torch.ones(
            chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    for chunk_index in range(total_len // chunk_size):
        query_chunk = query[:, :, chunk_index]
        key_chunk = key[:, :, chunk_index]
        value_chunk = value[:, :, chunk_index]
        within_chunk = (
            query_chunk @ key_chunk.transpose(-1, -2)
            * decay[:, :, chunk_index]
        ).masked_fill_(strict_upper, 0)
        value_prime = key_cumulative_decay[:, :, chunk_index] @ last_state
        value_new = value_chunk - value_prime
        inter_chunk = (
            query_chunk * g[:, :, chunk_index, :, None].exp()) @ last_state
        core_output[:, :, chunk_index] = (
            inter_chunk + within_chunk @ value_new)
        last_state = (
            last_state * g[:, :, chunk_index, -1, None, None].exp()
            + (
                key_chunk
                * (
                    g[:, :, chunk_index, -1, None]
                    - g[:, :, chunk_index]
                ).exp()[..., None]
            ).transpose(-1, -2) @ value_new
        )

    core_output = core_output.reshape(batch, num_heads, -1, value_dim)
    core_output = core_output[:, :, :seq_len].transpose(1, 2).contiguous()
    return core_output, last_state


def segmented_rule(
    inputs: tuple[torch.Tensor, ...],
    initial_state: torch.Tensor,
    inverse_builder: InverseBuilder,
    segment_size: int,
) -> TensorPair:
    query, key, value, g, beta = inputs
    outputs = []
    state = initial_state
    for start in range(0, query.shape[1], segment_size):
        end = min(start + segment_size, query.shape[1])
        output, state = chunk_gated_delta_rule(
            query[:, start:end],
            key[:, start:end],
            value[:, start:end],
            g[:, start:end],
            beta[:, start:end],
            inverse_builder=inverse_builder,
            initial_state=state,
        )
        outputs.append(output)
    return torch.cat(outputs, dim=1), state


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile_value / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def measure_once(case: Callable[[], object], iterations: int) -> float:
    started = time.perf_counter()
    for _ in range(iterations):
        case()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0 / iterations


def measure_pair(
    baseline: Callable[[], object],
    candidate: Callable[[], object],
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, object]:
    for _ in range(warmup):
        baseline()
        candidate()
    torch.cuda.synchronize()
    baseline_trials = []
    candidate_trials = []
    for repeat in range(repeats):
        if repeat % 2 == 0:
            baseline_trials.append(measure_once(baseline, iterations))
            candidate_trials.append(measure_once(candidate, iterations))
        else:
            candidate_trials.append(measure_once(candidate, iterations))
            baseline_trials.append(measure_once(baseline, iterations))
    baseline_median = statistics.median(baseline_trials)
    candidate_median = statistics.median(candidate_trials)
    return {
        "baseline": {
            "median_ms": baseline_median,
            "p10_ms": percentile(baseline_trials, 10),
            "p90_ms": percentile(baseline_trials, 90),
            "trials_ms": baseline_trials,
        },
        "candidate": {
            "median_ms": candidate_median,
            "p10_ms": percentile(candidate_trials, 10),
            "p90_ms": percentile(candidate_trials, 90),
            "trials_ms": candidate_trials,
        },
        "speedup": baseline_median / candidate_median,
    }


def peak_allocated(case: Callable[[], object]) -> int:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    case()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def make_inputs(
    length: int,
    device: torch.device,
    generator: torch.Generator,
    heads: int,
    head_dim: int,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    shape = (1, length, heads, head_dim)
    query = torch.randn(
        shape, device=device, dtype=torch.float16,
        generator=generator) * 0.1
    key = torch.randn(
        shape, device=device, dtype=torch.float16,
        generator=generator) * 0.1
    value = torch.randn(
        shape, device=device, dtype=torch.float16,
        generator=generator) * 0.1
    g = -(
        torch.rand(
            (1, length, heads), device=device, dtype=torch.float32,
            generator=generator) * 0.049 + 0.001
    ).to(torch.float16)
    beta = torch.sigmoid(torch.randn(
        (1, length, heads), device=device, dtype=torch.float32,
        generator=generator)).to(torch.float16)
    initial_state = torch.randn(
        (1, heads, head_dim, head_dim), device=device,
        dtype=torch.float32, generator=generator) * 0.01
    return (query, key, value, g, beta), initial_state


def compare_tensors(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float | bool]:
    reference_float = reference.float()
    candidate_float = candidate.float()
    difference = candidate_float - reference_float
    return {
        "finite": bool(torch.isfinite(candidate_float).all()),
        "max_abs": float(difference.abs().max()),
        "mean_abs": float(difference.abs().mean()),
        "relative_l2": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(reference_float).clamp_min(1.0e-12)
        ),
    }


def compare_pairs(reference: TensorPair, candidate: TensorPair) -> dict:
    return {
        "output": compare_tensors(reference[0], candidate[0]),
        "state": compare_tensors(reference[1], candidate[1]),
    }


def parity_passes(comparison: dict[str, float | bool]) -> bool:
    return bool(
        comparison["finite"]
        and comparison["max_abs"] <= MAX_ABS_LIMIT
        and comparison["relative_l2"] <= RELATIVE_L2_LIMIT
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[64, 1024, 4096, 7800])
    parser.add_argument("--segment-size", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--stress-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if CHUNK_SIZE != 64 or NEUMANN_ORDER != 3 or RESIDUAL_TERMS != 8:
        raise RuntimeError("M1-40 fixed algorithm contract changed")
    if QUALIFICATION_LENGTH not in args.lengths:
        raise ValueError(
            f"lengths must include qualification length {QUALIFICATION_LENGTH}")
    if args.segment_size != QUALIFICATION_LENGTH:
        raise ValueError(
            f"segment-size must remain fixed at {QUALIFICATION_LENGTH}")

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    cases = {}

    for length in args.lengths:
        inputs, initial_state = make_inputs(
            length, device, generator, args.heads, args.head_dim)
        _, key, _, g, beta = inputs
        inverse_input, identity = prepare_inverse_input(key, g, beta)

        def baseline_inverse() -> torch.Tensor:
            return row_loop_inverse(inverse_input.clone(), identity)

        def candidate_inverse() -> torch.Tensor:
            return masked_neumann_residual_inverse(
                inverse_input.clone(), identity)

        def baseline_complete() -> TensorPair:
            return segmented_rule(
                inputs, initial_state, row_loop_inverse, args.segment_size)

        def candidate_complete() -> TensorPair:
            return segmented_rule(
                inputs, initial_state,
                masked_neumann_residual_inverse, args.segment_size)

        inverse_reference = baseline_inverse()
        inverse_candidate = candidate_inverse()
        complete_reference = baseline_complete()
        complete_candidate = candidate_complete()
        torch.cuda.synchronize()
        inverse_parity = compare_tensors(
            inverse_reference, inverse_candidate)
        complete_parity = compare_pairs(
            complete_reference, complete_candidate)
        inverse_timing = measure_pair(
            baseline_inverse, candidate_inverse, args.warmup,
            args.iterations, args.repeats)
        complete_timing = measure_pair(
            baseline_complete, candidate_complete, args.warmup,
            args.iterations, args.repeats)
        cases[str(length)] = {
            "inverse_parity": inverse_parity,
            "complete_parity": complete_parity,
            "inverse_timing": inverse_timing,
            "complete_timing": complete_timing,
            "peak_allocated_bytes": {
                "baseline": peak_allocated(baseline_complete),
                "candidate": peak_allocated(candidate_complete),
            },
        }

    stress_failures = []
    stress_worst = {
        "inverse_max_abs": 0.0,
        "inverse_relative_l2": 0.0,
        "output_max_abs": 0.0,
        "output_relative_l2": 0.0,
        "state_max_abs": 0.0,
        "state_relative_l2": 0.0,
    }
    for sample_index in range(args.stress_samples):
        inputs, initial_state = make_inputs(
            CHUNK_SIZE, device, generator, args.heads, args.head_dim)
        _, key, _, g, beta = inputs
        inverse_input, identity = prepare_inverse_input(key, g, beta)
        inverse_parity = compare_tensors(
            row_loop_inverse(inverse_input.clone(), identity),
            masked_neumann_residual_inverse(
                inverse_input.clone(), identity),
        )
        complete_parity = compare_pairs(
            segmented_rule(
                inputs, initial_state, row_loop_inverse,
                args.segment_size),
            segmented_rule(
                inputs, initial_state, masked_neumann_residual_inverse,
                args.segment_size),
        )
        for prefix, comparison in (
            ("inverse", inverse_parity),
            ("output", complete_parity["output"]),
            ("state", complete_parity["state"]),
        ):
            stress_worst[f"{prefix}_max_abs"] = max(
                stress_worst[f"{prefix}_max_abs"],
                comparison["max_abs"],
            )
            stress_worst[f"{prefix}_relative_l2"] = max(
                stress_worst[f"{prefix}_relative_l2"],
                comparison["relative_l2"],
            )
        if not (
            parity_passes(inverse_parity)
            and parity_passes(complete_parity["output"])
            and parity_passes(complete_parity["state"])
        ):
            stress_failures.append(sample_index)

    all_parity_pass = all(
        parity_passes(case["inverse_parity"])
        and parity_passes(case["complete_parity"]["output"])
        and parity_passes(case["complete_parity"]["state"])
        for case in cases.values()
    ) and not stress_failures
    qualification = cases[str(QUALIFICATION_LENGTH)]
    inverse_speed_pass = (
        qualification["inverse_timing"]["speedup"]
        >= MIN_INVERSE_SPEEDUP)
    complete_speed_pass = (
        qualification["complete_timing"]["speedup"]
        >= MIN_COMPLETE_SPEEDUP)
    memory_pass = all(
        case["peak_allocated_bytes"]["candidate"]
        <= case["peak_allocated_bytes"]["baseline"]
        for case in cases.values()
    )
    report = {
        "ok": bool(
            all_parity_pass
            and inverse_speed_pass
            and complete_speed_pass
            and memory_pass),
        "gates": {
            "parity_pass": all_parity_pass,
            "inverse_speed_pass": inverse_speed_pass,
            "complete_speed_pass": complete_speed_pass,
            "memory_pass": memory_pass,
        },
        "fixed_contract": {
            "chunk_size": CHUNK_SIZE,
            "neumann_order": NEUMANN_ORDER,
            "residual_terms": RESIDUAL_TERMS,
            "qualification_length": QUALIFICATION_LENGTH,
            "min_inverse_speedup": MIN_INVERSE_SPEEDUP,
            "min_complete_speedup": MIN_COMPLETE_SPEEDUP,
            "max_abs_limit": MAX_ABS_LIMIT,
            "relative_l2_limit": RELATIVE_L2_LIMIT,
        },
        "stress": {
            "samples": args.stress_samples,
            "failed_samples": stress_failures,
            "worst": stress_worst,
        },
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "config": vars(args) | {"out": str(args.out)},
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
