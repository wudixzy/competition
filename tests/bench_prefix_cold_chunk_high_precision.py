#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

import torch

import bench_prefix_cold_chunk_hybrid as frozen
from bench_prefix_attention_breakdown import make_case, run_segment


SCHEMA = "bi100-m1-100-prefix-cold-high-precision-v1"
PRODUCTION_QUERY_LEN = 8176
QUERY_HEADS = 4
KV_HEADS = 1
HEAD_DIM = 256
BLOCK_SIZE = 16
TILE_SIZE = 512
PRIMARY_CONTEXT = 65536
PARTIAL_CONTEXT = 65552
SEEDS = (20260716, 20260727)
SAMPLE_QUERY_HEAD_PAIRS = (
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (7, 0),
    (15, 1),
    (31, 2),
    (63, 3),
    (127, 0),
    (255, 1),
    (511, 2),
    (1023, 3),
    (2047, 0),
    (4095, 1),
    (6143, 2),
    (8175, 3),
)
WARMUP = 1
REPEATS = 3
ORACLE_CPU_THREADS = 8
PARTIAL_SEED_OFFSET = 100
MIN_PRIMARY_REDUCTION = 0.15
NONINFERIOR_RELATIVE_L2_SLACK = 1e-8
FROZEN_ARTIFACTS = {
    "bench_prefix_attention_breakdown.py":
        "2ab82f69e7833dc2965b03e4cbcebe5beafd9d4954a3e3babda101bb54a0ddd2",
    "bench_prefix_cold_chunk_hybrid.py":
        "e2dffa151c99f4cf28d827877db68bbcb0a0c0bd6433c466017c255df2f3d076",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_artifacts() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    observed = {
        name: digest(root / name)
        for name in FROZEN_ARTIFACTS
    }
    if observed != FROZEN_ARTIFACTS:
        raise RuntimeError("frozen E-PREFIX-08 artifact identity differs")
    return observed


def measure_once(operation: Callable[[], torch.Tensor]) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    operation()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def measure_pair(
    control: Callable[[], torch.Tensor],
    candidate: Callable[[], torch.Tensor],
    *,
    candidate_first: bool,
) -> tuple[list[float], list[float], list[str]]:
    control_trials = []
    candidate_trials = []
    measured_order = []
    for trial in range(WARMUP + REPEATS):
        first_is_candidate = candidate_first ^ bool(trial % 2)
        operations = (
            (("candidate", candidate), ("control", control))
            if first_is_candidate
            else (("control", control), ("candidate", candidate))
        )
        trial_values = {
            name: measure_once(operation)
            for name, operation in operations
        }
        if trial >= WARMUP:
            control_trials.append(trial_values["control"])
            candidate_trials.append(trial_values["candidate"])
            measured_order.append(
                "candidate/control"
                if first_is_candidate
                else "control/candidate"
            )
    return control_trials, candidate_trials, measured_order


def ordered_context(
    tensors: dict[str, torch.Tensor],
    context_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_table = tensors["block_table"]
    key_cache = tensors["key_cache"]
    value_cache = tensors["value_cache"]
    block_count = (context_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_ids = block_table[:block_count]
    key = (
        key_cache[block_ids]
        .permute(0, 3, 1, 2, 4)
        .contiguous()
        .view(-1, KV_HEADS, HEAD_DIM)[:context_len, 0]
    )
    value = (
        value_cache[block_ids]
        .permute(0, 3, 1, 2)
        .contiguous()
        .view(-1, KV_HEADS, HEAD_DIM)[:context_len, 0]
    )
    return key, value


def high_precision_samples(
    tensors: dict[str, torch.Tensor],
    context_len: int,
) -> torch.Tensor:
    context_key, context_value = ordered_context(tensors, context_len)
    all_key = torch.cat((context_key, tensors["key"][:, 0]), dim=0)
    all_value = torch.cat((context_value, tensors["value"][:, 0]), dim=0)
    all_key = all_key.cpu().to(torch.float64)
    all_value = all_value.cpu().to(torch.float64)
    query = tensors["query"].cpu().to(torch.float64)
    scale = HEAD_DIM ** -0.5
    samples = []
    for query_index, head in SAMPLE_QUERY_HEAD_PAIRS:
        end = context_len + query_index + 1
        scores = torch.mv(
            all_key[:end],
            query[query_index, head] * scale,
        )
        probabilities = torch.softmax(scores, dim=0)
        samples.append(torch.mv(
            all_value[:end].transpose(0, 1),
            probabilities,
        ))
    return torch.stack(samples)


def selected_output_samples(output: torch.Tensor) -> torch.Tensor:
    return torch.stack([
        output[query_index, head].cpu()
        for query_index, head in SAMPLE_QUERY_HEAD_PAIRS
    ])


def compare_to_rounded_oracle(
    output: torch.Tensor,
    oracle: torch.Tensor,
) -> dict[str, Any]:
    actual = selected_output_samples(output).to(torch.float64)
    rounded = oracle.to(torch.float16).to(torch.float64)
    difference = actual - rounded
    denominator = torch.linalg.vector_norm(rounded).clamp_min(1e-30)
    step_denominators = torch.linalg.vector_norm(
        rounded, dim=1).clamp_min(1e-30)
    step_relative_l2 = (
        torch.linalg.vector_norm(difference, dim=1)
        / step_denominators
    )
    return {
        "finite": bool(
            torch.isfinite(actual).all()
            and torch.isfinite(oracle).all()
        ),
        "aggregate_relative_l2": float(
            (torch.linalg.vector_norm(difference) / denominator).item()
        ),
        "maximum_step_relative_l2": float(
            step_relative_l2.max().item()
        ),
        "maximum_absolute_error": float(
            difference.abs().max().item()
        ),
        "mismatch_count": int(
            torch.count_nonzero(
                actual.to(torch.float16)
                != rounded.to(torch.float16)
            ).item()
        ),
        "sample_count": len(SAMPLE_QUERY_HEAD_PAIRS),
        "element_count": int(actual.numel()),
    }


def output_difference(
    control: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, Any]:
    control_float = control.float()
    candidate_float = candidate.float()
    difference = candidate_float - control_float
    denominator = torch.linalg.vector_norm(
        control_float).clamp_min(1e-30)
    return {
        "finite": bool(
            torch.isfinite(control).all()
            and torch.isfinite(candidate).all()
        ),
        "relative_l2": float(
            (torch.linalg.vector_norm(difference) / denominator).item()
        ),
        "maximum_absolute_error": float(
            difference.abs().max().item()
        ),
        "mismatch_count": int(
            torch.count_nonzero(candidate != control).item()
        ),
    }


def noninferior(
    control: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons = []
    if not control["finite"] or not candidate["finite"]:
        reasons.append("control or candidate contains NaN/Inf")
    if (
        candidate["aggregate_relative_l2"]
        > control["aggregate_relative_l2"]
        + NONINFERIOR_RELATIVE_L2_SLACK
    ):
        reasons.append("candidate aggregate relative L2 is worse")
    if (
        candidate["maximum_step_relative_l2"]
        > control["maximum_step_relative_l2"]
        + NONINFERIOR_RELATIVE_L2_SLACK
    ):
        reasons.append("candidate maximum step relative L2 is worse")
    if (
        candidate["maximum_absolute_error"]
        > control["maximum_absolute_error"]
    ):
        reasons.append("candidate maximum absolute error is worse")
    if candidate["mismatch_count"] > control["mismatch_count"]:
        reasons.append("candidate rounded-oracle mismatch count is worse")
    return not reasons, reasons


def run_case(
    device: torch.device,
    *,
    seed: int,
    context_len: int,
    measure_performance: bool,
) -> dict[str, Any]:
    tensors = make_case(
        PRODUCTION_QUERY_LEN,
        context_len,
        device,
        seed,
        num_query_heads=QUERY_HEADS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        block_size=BLOCK_SIZE,
    )
    frozen.randomize_cache(tensors, seed + context_len)

    def control_operation() -> torch.Tensor:
        return run_segment(
            tensors, context_len, TILE_SIZE, False)[0]

    def candidate_operation() -> torch.Tensor:
        return frozen.hybrid_segment(
            tensors, context_len, TILE_SIZE)

    previous_query_len = frozen.HYBRID_QUERY_LEN
    frozen.HYBRID_QUERY_LEN = PRODUCTION_QUERY_LEN
    try:
        control_output = control_operation()
        candidate_output = candidate_operation()
        torch.cuda.synchronize()
        oracle = high_precision_samples(tensors, context_len)
        control_oracle = compare_to_rounded_oracle(
            control_output, oracle)
        candidate_oracle = compare_to_rounded_oracle(
            candidate_output, oracle)
        qualified, reasons = noninferior(
            control_oracle, candidate_oracle)
        if measure_performance:
            (
                control_trials,
                candidate_trials,
                measured_order,
            ) = measure_pair(
                control_operation,
                candidate_operation,
                candidate_first=bool(seed % 2),
            )
        else:
            control_trials = []
            candidate_trials = []
            measured_order = []
    finally:
        frozen.HYBRID_QUERY_LEN = previous_query_len

    performance = None
    if measure_performance:
        control_median = statistics.median(control_trials)
        candidate_median = statistics.median(candidate_trials)
        performance = {
            "control_trials_ms": control_trials,
            "candidate_trials_ms": candidate_trials,
            "paired_order": measured_order,
            "control_median_ms": control_median,
            "candidate_median_ms": candidate_median,
            "speedup": control_median / candidate_median,
            "reduction": 1.0 - candidate_median / control_median,
        }
    return {
        "seed": seed,
        "context_len": context_len,
        "control_vs_rounded_fp64": control_oracle,
        "candidate_vs_rounded_fp64": candidate_oracle,
        "candidate_vs_control": output_difference(
            control_output, candidate_output),
        "numerically_noninferior": qualified,
        "noninferiority_reasons": reasons,
        "performance": performance,
    }


def aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = []
    if not all(case["numerically_noninferior"] for case in cases):
        reasons.append("at least one fixed case is numerically inferior")
    primary = [
        case for case in cases
        if case["context_len"] == PRIMARY_CONTEXT
    ]
    reductions = [
        case["performance"]["reduction"]
        for case in primary
        if case["performance"] is not None
    ]
    if len(reductions) != len(SEEDS):
        reasons.append("primary timing evidence is incomplete")
        reduction_median = None
    else:
        reduction_median = statistics.median(reductions)
        if reduction_median < MIN_PRIMARY_REDUCTION:
            reasons.append("primary median reduction is below 15%")
    return {
        "numerically_noninferior": all(
            case["numerically_noninferior"] for case in cases),
        "primary_reductions": reductions,
        "primary_reduction_median": reduction_median,
        "qualified": not reasons,
        "reasons": reasons,
        "decision": {
            "next_token_gate_authorized": not reasons,
            "service_integration_authorized": False,
            "production_promotion_authorized": False,
            "yaml_change_authorized": False,
            "main_merge_authorized": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.device.startswith("cuda:"):
        parser.error("--device must select one explicit CoreX CUDA device")

    artifacts = verify_frozen_artifacts()
    torch.set_grad_enabled(False)
    torch.set_num_threads(ORACLE_CPU_THREADS)
    if not torch.cuda.is_available():
        raise RuntimeError("CoreX CUDA device is unavailable")
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    cases = []
    for seed in SEEDS:
        cases.append(run_case(
            device,
            seed=seed,
            context_len=PRIMARY_CONTEXT,
            measure_performance=True,
        ))
        cases.append(run_case(
            device,
            seed=seed + PARTIAL_SEED_OFFSET,
            context_len=PARTIAL_CONTEXT,
            measure_performance=False,
        ))
    summary = aggregate(cases)
    report = {
        "schema": SCHEMA,
        "version": 1,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "frozen_artifacts": artifacts,
        "config": {
            "production_query_len": PRODUCTION_QUERY_LEN,
            "query_heads": QUERY_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "block_size": BLOCK_SIZE,
            "tile_size": TILE_SIZE,
            "primary_context": PRIMARY_CONTEXT,
            "partial_context": PARTIAL_CONTEXT,
            "seeds": list(SEEDS),
            "sample_query_head_pairs": [
                list(value) for value in SAMPLE_QUERY_HEAD_PAIRS
            ],
            "warmup": WARMUP,
            "repeats": REPEATS,
            "oracle_cpu_threads": ORACLE_CPU_THREADS,
            "partial_seed_offset": PARTIAL_SEED_OFFSET,
            "minimum_primary_reduction": MIN_PRIMARY_REDUCTION,
            "relative_l2_noninferiority_slack":
                NONINFERIOR_RELATIVE_L2_SLACK,
            "oracle": "CPU FP64 sampled full-sequence attention rounded once to FP16",
        },
        "cases": cases,
        "summary": summary,
    }
    if not math.isfinite(
        summary["primary_reduction_median"]
        if summary["primary_reduction_median"] is not None
        else math.nan
    ):
        report["summary"]["qualified"] = False
        report["summary"]["reasons"].append(
            "primary reduction median is not finite")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["summary"]["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
