#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any, Callable


BLOCK_SIZE = 16
BLOCKS_PER_TILE = 32
HEAD_DIM = 256
NUM_Q_HEADS = 6
NUM_KV_HEADS = 1
WARMUP_TRIALS = 5
MEASURED_TRIALS = 7
MAX_ABS_LIMIT = 1e-3
RELATIVE_L2_LIMIT = 1e-5
MIN_SPEEDUP = 1.5

CASES = (
    ("dense_q1", 0, 1, False),
    ("dense_q8", 0, 8, False),
    ("dense_q256", 0, 256, False),
    ("dense_boundary", 240, 16, False),
    ("paged_65520_q16", 65_520, 16, False),
    ("paged_234992_q8", 234_992, 8, False),
    ("service_65k_q8192", 65_536, 8_192, False),
    ("perf_74k", 73_728, 256, True),
    ("perf_128k", 130_816, 256, True),
    ("perf_235k", 234_736, 256, True),
)


def _finite_nonnegative(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def _finite_positive(value: Any) -> bool:
    return _finite_nonnegative(value) and value > 0


def evaluate(cases: Any) -> dict[str, Any]:
    reasons = []
    if not isinstance(cases, dict):
        return {
            "qualified": False,
            "reasons": ["cases must be an object"],
        }
    for name, ctx_len, q_len, performance_case in CASES:
        case = cases.get(name)
        if not isinstance(case, dict):
            reasons.append(f"missing case {name}")
            continue
        for field, expected in (
            ("ctx_len", ctx_len),
            ("q_len", q_len),
            ("total_kv_len", ctx_len + q_len),
        ):
            if (not isinstance(case.get(field), int)
                    or isinstance(case.get(field), bool)
                    or case[field] != expected):
                reasons.append(
                    f"case {name} {field} must equal {expected}")
        if ctx_len > BLOCK_SIZE and case.get(
                "physical_block_permutation") is not True:
            reasons.append(
                f"case {name} did not use a non-identity block table")
        if case.get("finite") is not True:
            reasons.append(f"case {name} is not finite")
        for field, limit in (
            ("output_max_abs", MAX_ABS_LIMIT),
            ("output_relative_l2", RELATIVE_L2_LIMIT),
            ("lse_relative_l2", RELATIVE_L2_LIMIT),
        ):
            value = case.get(field)
            if not _finite_nonnegative(value):
                reasons.append(f"case {name} has invalid {field}")
            elif value > limit + 1e-12:
                reasons.append(
                    f"case {name} {field} {value:.9g} exceeds {limit:.9g}")
        if performance_case:
            timings = {}
            for field in ("reference_median_ms", "candidate_median_ms"):
                value = case.get(field)
                if not _finite_positive(value):
                    reasons.append(f"case {name} has invalid {field}")
                else:
                    timings[field] = float(value)
            trials = {}
            for field in ("reference_trials_ms", "candidate_trials_ms"):
                values = case.get(field)
                if (not isinstance(values, list)
                        or len(values) != MEASURED_TRIALS
                        or not all(_finite_positive(value) for value in values)):
                    reasons.append(
                        f"case {name} must contain {MEASURED_TRIALS} "
                        f"positive {field}")
                else:
                    trials[field] = [float(value) for value in values]
            for prefix in ("reference", "candidate"):
                median_field = f"{prefix}_median_ms"
                trials_field = f"{prefix}_trials_ms"
                if median_field in timings and trials_field in trials:
                    measured_median = statistics.median(trials[trials_field])
                    if not math.isclose(
                            timings[median_field], measured_median,
                            rel_tol=1e-9, abs_tol=1e-12):
                        reasons.append(
                            f"case {name} {median_field} does not match trials")
            speedup = case.get("speedup")
            if (not isinstance(speedup, (int, float))
                    or isinstance(speedup, bool) or not math.isfinite(speedup)
                    or speedup < MIN_SPEEDUP):
                reasons.append(
                    f"case {name} speedup {speedup!r} is below {MIN_SPEEDUP}x")
            elif ("reference_median_ms" in timings
                  and "candidate_median_ms" in timings):
                measured_speedup = (
                    timings["reference_median_ms"]
                    / timings["candidate_median_ms"])
                if not math.isclose(
                        float(speedup), measured_speedup,
                        rel_tol=1e-9, abs_tol=1e-12):
                    reasons.append(
                        f"case {name} speedup does not match medians")
    return {
        "qualified": not reasons,
        "reasons": reasons,
    }


def _relative_l2(actual: Any, expected: Any) -> float:
    import torch

    difference = (actual.float() - expected.float()).norm().item()
    denominator = expected.float().norm().item()
    if denominator == 0:
        return 0.0 if difference == 0 else math.inf
    return difference / denominator


def _update_online(scores: Any, value: Any, m: Any, l: Any, output: Any) -> None:
    import torch

    block_max = scores.amax(dim=-1)
    new_max = torch.maximum(m, block_max)
    correction = torch.exp(m - new_max)
    probabilities = torch.exp(scores - new_max.unsqueeze(-1))
    l.mul_(correction).add_(probabilities.sum(dim=-1))
    output.mul_(correction.unsqueeze(-1)).add_(
        torch.matmul(probabilities, value))
    m.copy_(new_max)


def _reference(q: Any, k_new: Any, v_new: Any, key_cache: Any,
               value_cache: Any, block_table: Any, ctx_len: int,
               scale: float) -> tuple[Any, Any]:
    import torch

    q_len = q.shape[0]
    query = q.permute(1, 0, 2).float().mul(scale).unsqueeze(0)
    m = torch.full(
        (1, NUM_Q_HEADS, q_len),
        float("-inf"),
        dtype=torch.float32,
        device=q.device,
    )
    l = torch.zeros_like(m)
    output = torch.zeros(
        (1, NUM_Q_HEADS, q_len, HEAD_DIM),
        dtype=torch.float32,
        device=q.device,
    )
    tile_tokens = BLOCKS_PER_TILE * BLOCK_SIZE
    for token_start in range(0, ctx_len, tile_tokens):
        token_end = min(token_start + tile_tokens, ctx_len)
        first_block = token_start // BLOCK_SIZE
        last_block = (token_end + BLOCK_SIZE - 1) // BLOCK_SIZE
        block_ids = block_table[first_block:last_block]
        keys = (
            key_cache[block_ids]
            .permute(0, 3, 1, 2, 4)
            .contiguous()
            .view(-1, NUM_KV_HEADS, HEAD_DIM)
        )[:token_end - token_start]
        values = (
            value_cache[block_ids]
            .permute(0, 3, 1, 2)
            .contiguous()
            .view(-1, NUM_KV_HEADS, HEAD_DIM)
        )[:token_end - token_start]
        key_matrix = keys.permute(1, 0, 2).unsqueeze(1).transpose(-1, -2).float()
        value_matrix = values.permute(1, 0, 2).unsqueeze(1).float()
        _update_online(
            torch.matmul(query, key_matrix), value_matrix, m, l, output)

    for key_start in range(0, q_len, tile_tokens):
        key_end = min(key_start + tile_tokens, q_len)
        key_matrix = (
            k_new[key_start:key_end]
            .permute(1, 0, 2).unsqueeze(1).transpose(-1, -2).float()
        )
        value_matrix = (
            v_new[key_start:key_end].permute(1, 0, 2).unsqueeze(1).float()
        )
        scores = torch.matmul(query, key_matrix)
        key_positions = torch.arange(key_start, key_end, device=q.device)
        query_positions = torch.arange(q_len, device=q.device)
        mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        _update_online(scores, value_matrix, m, l, output)

    normalized = output.div(l.unsqueeze(-1))
    lse = m + torch.log(l)
    return (
        normalized.squeeze(0).permute(1, 0, 2).to(q.dtype),
        lse.squeeze(0).permute(1, 0),
    )


def _make_case(torch: Any, device: Any, ctx_len: int, q_len: int,
               seed: int) -> tuple[Any, ...]:
    if ctx_len % BLOCK_SIZE:
        raise ValueError("fixed context lengths must be block aligned")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    q = torch.randn(
        (q_len, NUM_Q_HEADS, HEAD_DIM),
        dtype=torch.float16,
        device=device,
        generator=generator,
    )
    k_new = torch.randn(
        (q_len, NUM_KV_HEADS, HEAD_DIM),
        dtype=torch.float16,
        device=device,
        generator=generator,
    )
    v_new = torch.randn(
        (q_len, NUM_KV_HEADS, HEAD_DIM),
        dtype=torch.float16,
        device=device,
        generator=generator,
    )
    num_blocks = ctx_len // BLOCK_SIZE
    if num_blocks:
        k_context = torch.randn(
            (ctx_len, NUM_KV_HEADS, HEAD_DIM),
            dtype=torch.float16,
            device=device,
            generator=generator,
        )
        v_context = torch.randn(
            (ctx_len, NUM_KV_HEADS, HEAD_DIM),
            dtype=torch.float16,
            device=device,
            generator=generator,
        )
        logical_key_cache = (
            k_context.view(num_blocks, BLOCK_SIZE, NUM_KV_HEADS,
                           HEAD_DIM // 8, 8)
            .permute(0, 2, 3, 1, 4)
            .contiguous()
        )
        logical_value_cache = (
            v_context.view(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
            .permute(0, 2, 3, 1)
            .contiguous()
        )
        physical_ids = torch.roll(torch.arange(
            num_blocks, dtype=torch.int64, device=device), shifts=1)
        key_cache = torch.empty_like(logical_key_cache)
        value_cache = torch.empty_like(logical_value_cache)
        key_cache[physical_ids] = logical_key_cache
        value_cache[physical_ids] = logical_value_cache
        block_table = physical_ids.to(torch.int32)
    else:
        key_cache = torch.empty(
            (0, NUM_KV_HEADS, HEAD_DIM // 8, BLOCK_SIZE, 8),
            dtype=torch.float16,
            device=device,
        )
        value_cache = torch.empty(
            (0, NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE),
            dtype=torch.float16,
            device=device,
        )
        block_table = torch.empty((0,), dtype=torch.int32, device=device)
    return q, k_new, v_new, key_cache, value_cache, block_table


def _measure(torch: Any, operation: Callable[[], Any]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    operation()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary_path = Path(handle.name)
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    extension = importlib.import_module("vllm.corex_fused_paged_prefill")
    forward = getattr(extension, "forward", None)
    if not callable(forward):
        raise RuntimeError("fused paged prefill extension has no forward")
    scale = HEAD_DIM**-0.5

    safety_inputs = list(_make_case(torch, device, BLOCK_SIZE, 1, 20260720))
    invalid_block_table = safety_inputs[-1].clone()
    invalid_block_table[0] = safety_inputs[3].shape[0]
    safety_inputs[-1] = invalid_block_table
    invalid_block_rejected = False
    try:
        forward(*safety_inputs, BLOCK_SIZE, scale)
        torch.cuda.synchronize(device)
    except RuntimeError:
        invalid_block_rejected = True
    if not invalid_block_rejected:
        raise RuntimeError("fused extension accepted an invalid physical block")
    del safety_inputs, invalid_block_table

    case_reports = {}
    for case_index, (name, ctx_len, q_len, performance_case) in enumerate(CASES):
        inputs = _make_case(torch, device, ctx_len, q_len, 20260721 + case_index)
        q, k_new, v_new, key_cache, value_cache, block_table = inputs
        expected_output, expected_lse = _reference(
            *inputs, ctx_len=ctx_len, scale=scale)
        actual_output, actual_lse = forward(
            *inputs, ctx_len, scale)
        torch.cuda.synchronize(device)
        if tuple(actual_output.shape) != tuple(expected_output.shape):
            raise RuntimeError(f"case {name} output shape mismatch")
        if tuple(actual_lse.shape) != tuple(expected_lse.shape):
            raise RuntimeError(f"case {name} LSE shape mismatch")

        output_difference = actual_output.float() - expected_output.float()
        finite = bool(
            torch.isfinite(actual_output).all().item()
            and torch.isfinite(actual_lse).all().item()
        )
        report = {
            "ctx_len": ctx_len,
            "finite": finite,
            "lse_relative_l2": _relative_l2(actual_lse, expected_lse),
            "output_max_abs": output_difference.abs().max().item(),
            "output_relative_l2": _relative_l2(
                actual_output, expected_output),
            "physical_block_permutation": (
                ctx_len <= BLOCK_SIZE or not torch.equal(
                    block_table,
                    torch.arange(
                        block_table.numel(), dtype=torch.int32,
                        device=device))),
            "q_len": q_len,
            "total_kv_len": ctx_len + q_len,
        }
        if performance_case:
            reference_operation = lambda: _reference(
                *inputs, ctx_len=ctx_len, scale=scale)
            candidate_operation = lambda: forward(
                *inputs, ctx_len, scale)
            for _ in range(WARMUP_TRIALS):
                reference_operation()
                candidate_operation()
            torch.cuda.synchronize(device)
            reference_ms = []
            candidate_ms = []
            for trial in range(MEASURED_TRIALS):
                operations = (
                    (reference_operation, reference_ms),
                    (candidate_operation, candidate_ms),
                )
                if trial % 2:
                    operations = tuple(reversed(operations))
                for operation, destination in operations:
                    destination.append(_measure(torch, operation))
            reference_median = statistics.median(reference_ms)
            candidate_median = statistics.median(candidate_ms)
            report.update({
                "candidate_median_ms": candidate_median,
                "candidate_trials_ms": candidate_ms,
                "reference_median_ms": reference_median,
                "reference_trials_ms": reference_ms,
                "speedup": reference_median / candidate_median,
            })
        case_reports[name] = report
        del inputs, q, k_new, v_new, key_cache, value_cache, block_table
        torch.cuda.empty_cache()

    decision = evaluate(case_reports)
    result = {
        "cases": case_reports,
        "decision": decision,
        "device": args.device,
        "device_name": torch.cuda.get_device_name(device),
        "protocol": {
            "block_size": BLOCK_SIZE,
            "blocks_per_tile": BLOCKS_PER_TILE,
            "block_table": "deterministic one-block physical rotation",
            "head_dim": HEAD_DIM,
            "measured_trials": MEASURED_TRIALS,
            "num_kv_heads": NUM_KV_HEADS,
            "num_q_heads": NUM_Q_HEADS,
            "warmup_trials": WARMUP_TRIALS,
        },
        "safety": {
            "invalid_physical_block_rejected": invalid_block_rejected,
        },
        "schema": "bi100-fused-paged-prefill-gate-v1",
        "thresholds": {
            "max_abs": MAX_ABS_LIMIT,
            "minimum_speedup": MIN_SPEEDUP,
            "relative_l2": RELATIVE_L2_LIMIT,
        },
        "torch_version": torch.__version__,
        "version": 1,
    }
    _write_json_atomic(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if decision["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
