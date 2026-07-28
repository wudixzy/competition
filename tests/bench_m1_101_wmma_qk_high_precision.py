#!/usr/bin/env python3
"""High-precision non-inferiority gate for the frozen M1-28 WMMA QK tile."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
from typing import Any, Callable

import torch


SCHEMA = "bi100-m1-101-wmma-qk-high-precision-v1"
TILES = 128
QUERY_ROWS = 16
KEY_ROWS = 32
HEAD_DIM = 256
MAGNITUDES = (0.5, 1.0, 2.0)
SEED = 20260718
TIMING_SEED_OFFSET = 100
WARMUP = 5
REPEATS = 20
ORACLE_CPU_THREADS = 8
MINIMUM_QK_SPEEDUP = 1.5
NONINFERIOR_RELATIVE_L2_SLACK = 1e-8
FROZEN_SOURCE = (
    "exp/M1-28-wmma-qk-capability"
    "@b03cb39ad23b49eb15728c99b14d9e1a458fb7f5"
)
FROZEN_ARTIFACTS = {
    "bench_attention_wmma_qk.py":
        "55a4ed735abda6e88f2bbb3f4cc264af1b9629062fb62c9dfc130f683c63895f",
    "build_corex_attention_wmma_qk_probe.sh":
        "9436cd30428f357addf3bcf90d14618a984d48d08f593ac88db70dc6da688958",
    "corex_attention_wmma_qk_probe.cu":
        "08a68ffc068c7f5a21796b32b64e2164c03f7c1b0270e19d862e116abdd3c688",
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
        raise RuntimeError("frozen M1-28 artifact identity differs")
    return observed


def load_extension(path: Path):
    name = path.name.split(".", 1)[0]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "qk"):
        raise RuntimeError("extension does not expose qk")
    return module


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
) -> tuple[list[float], list[float], list[str]]:
    control_trials = []
    candidate_trials = []
    measured_order = []
    for trial in range(WARMUP + REPEATS):
        candidate_first = bool(trial % 2)
        operations = (
            (("candidate", candidate), ("control", control))
            if candidate_first
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
                if candidate_first
                else "control/candidate"
            )
    return control_trials, candidate_trials, measured_order


def make_case(
    device: torch.device,
    *,
    magnitude: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    query = torch.randn(
        (TILES, QUERY_ROWS, HEAD_DIM),
        generator=generator,
        device=device,
        dtype=torch.float16,
    ) * magnitude
    key = torch.randn(
        (TILES, KEY_ROWS, HEAD_DIM),
        generator=generator,
        device=device,
        dtype=torch.float16,
    ) * magnitude
    value = torch.randn(
        (TILES, KEY_ROWS, HEAD_DIM),
        generator=generator,
        device=device,
        dtype=torch.float16,
    )
    return query, key, value


def attention_output(
    scores: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    scale = HEAD_DIM ** -0.5
    return torch.bmm(
        torch.softmax(scores * scale, dim=-1),
        value.float(),
    ).to(torch.float16)


def high_precision_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    query_fp64 = query.cpu().to(torch.float64)
    key_fp64 = key.cpu().to(torch.float64)
    value_fp64 = value.cpu().to(torch.float64)
    scores = torch.bmm(query_fp64, key_fp64.transpose(1, 2))
    return torch.bmm(
        torch.softmax(scores * (HEAD_DIM ** -0.5), dim=-1),
        value_fp64,
    )


def compare_to_rounded_oracle(
    output: torch.Tensor,
    oracle: torch.Tensor,
) -> dict[str, Any]:
    actual = output.cpu().to(torch.float64)
    rounded = oracle.to(torch.float16).to(torch.float64)
    difference = actual - rounded
    denominator = torch.linalg.vector_norm(rounded).clamp_min(1e-30)
    row_denominators = torch.linalg.vector_norm(
        rounded, dim=-1).clamp_min(1e-30)
    row_relative_l2 = (
        torch.linalg.vector_norm(difference, dim=-1)
        / row_denominators
    )
    return {
        "finite": bool(
            torch.isfinite(actual).all()
            and torch.isfinite(oracle).all()
        ),
        "aggregate_relative_l2": float(
            (torch.linalg.vector_norm(difference) / denominator).item()
        ),
        "maximum_row_relative_l2": float(
            row_relative_l2.max().item()
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
        "row_count": TILES * QUERY_ROWS,
        "element_count": int(actual.numel()),
    }


def tensor_difference(
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
        candidate["maximum_row_relative_l2"]
        > control["maximum_row_relative_l2"]
        + NONINFERIOR_RELATIVE_L2_SLACK
    ):
        reasons.append("candidate maximum row relative L2 is worse")
    if (
        candidate["maximum_absolute_error"]
        > control["maximum_absolute_error"]
    ):
        reasons.append("candidate maximum absolute error is worse")
    if candidate["mismatch_count"] > control["mismatch_count"]:
        reasons.append("candidate rounded-oracle mismatch count is worse")
    return not reasons, reasons


def run_numerical_case(
    extension: Any,
    device: torch.device,
    *,
    case_index: int,
    magnitude: float,
) -> dict[str, Any]:
    query, key, value = make_case(
        device,
        magnitude=magnitude,
        seed=SEED + case_index,
    )
    control_scores = torch.bmm(
        query.float(), key.float().transpose(1, 2))
    candidate_scores = extension.qk(query, key)
    control_output = attention_output(control_scores, value)
    candidate_output = attention_output(candidate_scores, value)
    torch.cuda.synchronize()
    oracle = high_precision_output(query, key, value)
    control_oracle = compare_to_rounded_oracle(control_output, oracle)
    candidate_oracle = compare_to_rounded_oracle(candidate_output, oracle)
    qualified, reasons = noninferior(control_oracle, candidate_oracle)
    return {
        "magnitude": magnitude,
        "seed": SEED + case_index,
        "control_vs_rounded_fp64": control_oracle,
        "candidate_vs_rounded_fp64": candidate_oracle,
        "candidate_vs_control_scores": tensor_difference(
            control_scores, candidate_scores),
        "candidate_vs_control_output": tensor_difference(
            control_output, candidate_output),
        "numerically_noninferior": qualified,
        "noninferiority_reasons": reasons,
    }


def run_timing(
    extension: Any,
    device: torch.device,
) -> dict[str, Any]:
    query, key, _ = make_case(
        device,
        magnitude=1.0,
        seed=SEED + TIMING_SEED_OFFSET,
    )

    def control() -> torch.Tensor:
        return torch.bmm(
            query.float(), key.float().transpose(1, 2))

    def candidate() -> torch.Tensor:
        return extension.qk(query, key)

    control_trials, candidate_trials, order = measure_pair(
        control, candidate)
    control_median = statistics.median(control_trials)
    candidate_median = statistics.median(candidate_trials)
    speedup = control_median / candidate_median
    return {
        "seed": SEED + TIMING_SEED_OFFSET,
        "control_trials_ms": control_trials,
        "candidate_trials_ms": candidate_trials,
        "paired_order": order,
        "control_median_ms": control_median,
        "candidate_median_ms": candidate_median,
        "speedup": speedup,
        "qualified": math.isfinite(speedup)
        and speedup >= MINIMUM_QK_SPEEDUP,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.device.startswith("cuda:"):
        parser.error("--device must select one explicit CoreX CUDA device")
    if not args.extension.is_file():
        parser.error("--extension must be an existing file")

    artifacts = verify_frozen_artifacts()
    torch.set_grad_enabled(False)
    torch.set_num_threads(ORACLE_CPU_THREADS)
    if not torch.cuda.is_available():
        raise RuntimeError("CoreX CUDA device is unavailable")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    extension = load_extension(args.extension)

    cases = [
        run_numerical_case(
            extension,
            device,
            case_index=case_index,
            magnitude=magnitude,
        )
        for case_index, magnitude in enumerate(MAGNITUDES)
    ]
    timing = run_timing(extension, device)
    reasons = []
    if not all(case["numerically_noninferior"] for case in cases):
        reasons.append("at least one fixed magnitude is numerically inferior")
    if not timing["qualified"]:
        reasons.append("paired QK speedup is below 1.5x or non-finite")
    qualified = not reasons

    report = {
        "schema": SCHEMA,
        "version": 1,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "frozen_source": FROZEN_SOURCE,
        "frozen_artifacts": artifacts,
        "extension_sha256": digest(args.extension),
        "config": {
            "tiles": TILES,
            "query_rows": QUERY_ROWS,
            "key_rows": KEY_ROWS,
            "head_dim": HEAD_DIM,
            "magnitudes": list(MAGNITUDES),
            "seed": SEED,
            "timing_seed_offset": TIMING_SEED_OFFSET,
            "warmup": WARMUP,
            "repeats": REPEATS,
            "oracle_cpu_threads": ORACLE_CPU_THREADS,
            "minimum_qk_speedup": MINIMUM_QK_SPEEDUP,
            "relative_l2_noninferiority_slack":
                NONINFERIOR_RELATIVE_L2_SLACK,
            "oracle": (
                "CPU FP64 QK, softmax, and PV rounded once to FP16"
            ),
        },
        "cases": cases,
        "timing": timing,
        "summary": {
            "numerically_noninferior": all(
                case["numerically_noninferior"] for case in cases),
            "qualified": qualified,
            "reasons": reasons,
            "decision": {
                "integration_benefit_gate_authorized": qualified,
                "service_integration_authorized": False,
                "production_promotion_authorized": False,
                "yaml_change_authorized": False,
                "main_merge_authorized": False,
            },
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
