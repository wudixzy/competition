#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time

import ixformer
import torch
import torch.nn.functional as F


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def make_constants(device: torch.device, chunk_size: int, dtype: torch.dtype):
    ones = torch.ones(
        chunk_size, chunk_size, dtype=torch.bool, device=device)
    return (
        torch.triu(ones, diagonal=0),
        torch.eye(chunk_size, dtype=dtype, device=device),
        torch.triu(ones, diagonal=1),
    )


def chunk_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    constants,
    solve_identity: torch.Tensor | None = None,
    chunk_size: int = 64,
):
    query = l2norm(query)
    key = l2norm(key)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]
    batch, num_heads, seq_len, k_dim = key.shape
    v_dim = value.shape[-1]
    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad))
    key = F.pad(key, (0, 0, 0, pad))
    value = F.pad(value, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad))
    g = F.pad(g, (0, pad))
    total_len = seq_len + pad
    query = query * (query.shape[-1] ** -0.5)

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask_upper, identity, mask_upper2 = constants

    g = g.cumsum(dim=-1)
    decay_mask = ((
        g.unsqueeze(-1) - g.unsqueeze(-2)
    ).tril().exp().float()).tril()
    attn = -((
        k_beta @ key.transpose(-1, -2)
    ) * decay_mask).masked_fill(mask_upper, 0)
    if solve_identity is None:
        for index in range(1, chunk_size):
            row = attn[..., index, :index].clone()
            sub = attn[..., :index, :index].clone()
            attn[..., index, :index] = (
                row + (row.unsqueeze(-1) * sub).sum(-2))
        attn = attn + identity
    else:
        attn_shape = attn.shape
        flat_attn = attn.reshape(-1, chunk_size, chunk_size)
        if solve_identity.shape[0] != flat_attn.shape[0]:
            raise ValueError(
                f"solve identity batch mismatch: {solve_identity.shape[0]} "
                f"!= {flat_attn.shape[0]}")
        system = solve_identity - flat_attn
        solution = ixformer.functions.solve(system, solve_identity)
        residual = (solve_identity - system @ solution).contiguous()
        correction = ixformer.functions.solve(system, residual)
        attn = (solution + correction).reshape(attn_shape)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    last_state = initial_state.to(value)
    core_out = torch.zeros_like(value)
    for index in range(total_len // chunk_size):
        q_i, k_i, v_i = query[:, :, index], key[:, :, index], value[:, :, index]
        attn_i = (
            q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, index]
        ).masked_fill_(mask_upper2, 0)
        v_prime = k_cumdecay[:, :, index] @ last_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, index, :, None].exp()) @ last_state
        core_out[:, :, index] = attn_inter + attn_i @ v_new
        last_state = (
            last_state * g[:, :, index, -1, None, None].exp()
            + (k_i * (
                g[:, :, index, -1, None] - g[:, :, index]
            ).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    core_out = core_out.reshape(batch, num_heads, -1, v_dim)[:, :, :seq_len]
    return core_out.transpose(1, 2).contiguous(), last_state


def bench(fn, *, warmups: int, repeats: int, iterations: int):
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
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-label", required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(20260715)
    device = torch.device("cuda:0")
    dtype = torch.float16
    batch = 1
    heads = 12
    key_dim = 128
    value_dim = 128
    chunk_size = 64
    constants = make_constants(device, chunk_size, torch.float32)
    results = {}

    with torch.no_grad():
        for tokens in (64, 256):
            query = torch.randn(
                batch, tokens, heads, key_dim, device=device, dtype=dtype)
            key = torch.randn_like(query)
            value = torch.randn(
                batch, tokens, heads, value_dim, device=device, dtype=dtype)
            g = -torch.rand(
                batch, tokens, heads, device=device, dtype=torch.float32) * 0.02
            beta = torch.sigmoid(torch.randn(
                batch, tokens, heads, device=device, dtype=dtype))
            state = torch.randn(
                batch, heads, key_dim, value_dim,
                device=device, dtype=torch.float32) * 0.01

            matrix_batch = batch * heads * (
                (tokens + chunk_size - 1) // chunk_size)
            solve_identity = constants[1].expand(
                matrix_batch, -1, -1).contiguous()
            current = lambda: chunk_rule(
                query, key, value, g, beta, state, constants=constants)
            solved = lambda: chunk_rule(
                query, key, value, g, beta, state,
                constants=constants, solve_identity=solve_identity)
            current_output, current_state = current()
            solved_output, solved_state = solved()
            parity = {
                "output_max_abs": float(
                    (current_output - solved_output).abs().max().item()),
                "state_max_abs": float(
                    (current_state - solved_state).abs().max().item()),
                "output_mean_abs": float(
                    (current_output - solved_output).abs().mean().item()),
                "state_mean_abs": float(
                    (current_state - solved_state).abs().mean().item()),
                "all_finite": all(bool(torch.isfinite(tensor).all().item())
                                  for tensor in (
                                      current_output, current_state,
                                      solved_output, solved_state)),
            }
            timings = {
                "current_loop_full_rule": bench(
                    current, warmups=args.warmups,
                    repeats=args.repeats, iterations=1),
                "ixformer_solve_full_rule": bench(
                    solved, warmups=args.warmups,
                    repeats=args.repeats, iterations=1),
            }
            full_current = timings["current_loop_full_rule"]["median_ms"]
            full_solved = timings["ixformer_solve_full_rule"]["median_ms"]
            timings["ixformer_solve_full_rule"]["speedup_vs_current"] = (
                full_current / full_solved)
            results[f"t{tokens}"] = {
                "parity": parity,
                "timings": timings,
            }

    print(json.dumps({
        "device": args.device_label,
        "shape": {
            "batch": batch,
            "heads_per_tp_rank": heads,
            "key_dim": key_dim,
            "value_dim": value_dim,
            "chunk_size": chunk_size,
            "dtype": str(dtype),
        },
        "candidate": "ixformer solve((I - A), I) with one refinement",
        "results": results,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
