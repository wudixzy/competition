#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


SCHEMA = "bi100-moe-compensated-w13-v1"
EXPERTS = 256
TOP_K = 8
HIDDEN = 2048
INTERMEDIATE = 128
W13_ROWS = 2 * INTERMEDIATE
DEFAULT_SEEDS = (20260716, 20260727)


def load_extension(name: str, path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"extension is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(
                value,
                output,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    delta = actual.float() - expected.float()
    squared_error = float(delta.square().sum())
    squared_reference = float(expected.float().square().sum())
    return math.sqrt(squared_error / max(squared_reference, 1.0e-30))


def comparison(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, Any]:
    finite = bool(
        torch.isfinite(actual).all() and torch.isfinite(expected).all())
    mismatch_count = int(torch.count_nonzero(actual != expected))
    if not finite:
        return {
            "exact": False,
            "finite": False,
            "mismatch_count": mismatch_count,
            "max_abs": None,
            "mean_abs": None,
            "relative_l2": None,
        }
    delta = (actual.float() - expected.float()).abs()
    return {
        "exact": bool(torch.equal(actual, expected)),
        "finite": True,
        "mismatch_count": mismatch_count,
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "relative_l2": relative_l2(actual, expected),
    }


def route_ids(router_logits: torch.Tensor) -> torch.Tensor:
    return torch.topk(router_logits.float(), TOP_K, dim=-1).indices[0]


def exact_half_dot(
    selected_w13: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    selected_cpu = selected_w13.detach().cpu().to(torch.float64)
    input_cpu = hidden_states.detach().cpu().to(torch.float64).reshape(-1)
    exact = torch.mv(
        selected_cpu.reshape(-1, HIDDEN),
        input_cpu,
    ).reshape(TOP_K, W13_ROWS)
    return exact.to(torch.float16)


class SequenceAccumulator:
    def __init__(self) -> None:
        self.steps = 0
        self.rows = 0
        self.finite_steps = 0
        self.exact_steps = 0
        self.mismatch_count = 0
        self.max_abs = 0.0
        self.absolute_error = 0.0
        self.squared_error = 0.0
        self.squared_reference = 0.0
        self.max_step_relative_l2 = 0.0
        self.metrics_finite = True

    def add(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        self.steps += 1
        self.rows += actual.numel()
        finite = bool(
            torch.isfinite(actual).all()
            and torch.isfinite(expected).all())
        self.finite_steps += int(finite)
        self.exact_steps += int(torch.equal(actual, expected))
        self.mismatch_count += int(torch.count_nonzero(actual != expected))
        if not finite:
            self.metrics_finite = False
            return

        delta = (actual.float() - expected.float()).abs()
        squared_error = float(delta.square().sum())
        squared_reference = float(expected.float().square().sum())
        step_relative_l2 = math.sqrt(
            squared_error / max(squared_reference, 1.0e-30))
        self.max_abs = max(self.max_abs, float(delta.max()))
        self.absolute_error += float(delta.sum())
        self.squared_error += squared_error
        self.squared_reference += squared_reference
        self.max_step_relative_l2 = max(
            self.max_step_relative_l2,
            step_relative_l2,
        )

    def report(self) -> dict[str, Any]:
        if not self.metrics_finite:
            max_abs = None
            mean_abs = None
            aggregate_relative_l2 = None
            max_step_relative_l2 = None
        else:
            max_abs = self.max_abs
            mean_abs = self.absolute_error / max(self.rows, 1)
            aggregate_relative_l2 = math.sqrt(
                self.squared_error
                / max(self.squared_reference, 1.0e-30))
            max_step_relative_l2 = self.max_step_relative_l2
        return {
            "steps": self.steps,
            "rows": self.rows,
            "finite_steps": self.finite_steps,
            "exact_steps": self.exact_steps,
            "mismatch_count": self.mismatch_count,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "relative_l2": aggregate_relative_l2,
            "max_step_relative_l2": max_step_relative_l2,
        }


def measure(
    case: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        case()
    torch.cuda.synchronize()
    trials = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(iterations):
            case()
        torch.cuda.synchronize()
        trials.append(
            (time.perf_counter() - started) * 1000.0 / iterations)
    ordered = sorted(trials)
    return {
        "median_ms": statistics.median(trials),
        "p10_ms": ordered[max(0, int(0.1 * (len(ordered) - 1)))],
        "p90_ms": ordered[min(
            len(ordered) - 1,
            int(0.9 * (len(ordered) - 1)),
        )],
        "trials_ms": trials,
    }


def parse_seeds(raw: str) -> tuple[int, ...]:
    values = tuple(
        int(value.strip()) for value in raw.split(",") if value.strip())
    if len(values) < 2 or len(values) != len(set(values)):
        raise ValueError("at least two unique seeds are required")
    return values


def fixture(
    *,
    seed: int,
    sequence_steps: int,
    device: torch.device,
    direct: Any,
    compensated: Any,
) -> tuple[dict[str, Any], dict[str, Any], tuple[torch.Tensor, ...]]:
    generator = torch.Generator(device=device).manual_seed(seed)
    dtype = torch.float16
    hidden_states = torch.randn(
        (1, HIDDEN), device=device, dtype=dtype, generator=generator)
    router_logits = torch.randn(
        (1, EXPERTS), device=device, dtype=dtype, generator=generator)
    w13 = torch.randn(
        (EXPERTS, W13_ROWS, HIDDEN),
        device=device,
        dtype=dtype,
        generator=generator,
    ) * 0.02
    expert_ids = route_ids(router_logits)
    selected_w13 = torch.index_select(w13, 0, expert_ids)
    vendor = F.linear(
        hidden_states,
        selected_w13.reshape(-1, HIDDEN),
    ).reshape(TOP_K, W13_ROWS)
    direct_output = direct.w13(hidden_states, w13, expert_ids)
    compensated_output = compensated.w13(hidden_states, w13, expert_ids)
    exact_half = exact_half_dot(selected_w13, hidden_states)
    fixed = {
        "vendor_vs_exact": comparison(vendor.cpu(), exact_half),
        "direct_vs_vendor": comparison(
            direct_output.detach().cpu(),
            vendor.detach().cpu(),
        ),
        "compensated_vs_vendor": comparison(
            compensated_output.detach().cpu(),
            vendor.detach().cpu(),
        ),
        "compensated_vs_exact": comparison(
            compensated_output.detach().cpu(),
            exact_half,
        ),
    }

    accumulators = {
        "direct": SequenceAccumulator(),
        "compensated": SequenceAccumulator(),
    }
    for _ in range(sequence_steps):
        step_hidden = torch.randn(
            (1, HIDDEN), device=device, dtype=dtype, generator=generator)
        step_logits = torch.randn(
            (1, EXPERTS), device=device, dtype=dtype, generator=generator)
        step_ids = route_ids(step_logits)
        step_selected = torch.index_select(w13, 0, step_ids)
        expected = F.linear(
            step_hidden,
            step_selected.reshape(-1, HIDDEN),
        ).reshape(TOP_K, W13_ROWS)
        accumulators["direct"].add(
            direct.w13(step_hidden, w13, step_ids),
            expected,
        )
        accumulators["compensated"].add(
            compensated.w13(step_hidden, w13, step_ids),
            expected,
        )
    torch.cuda.synchronize()
    sequence = {
        name: accumulator.report()
        for name, accumulator in accumulators.items()
    }
    timing_state = (
        hidden_states,
        router_logits,
        w13,
        expert_ids,
        selected_w13,
    )
    return fixed, sequence, timing_state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-extension", type=Path, required=True)
    parser.add_argument("--direct-extension", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
    )
    parser.add_argument("--sequence-steps", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    for name in (
        "sequence_steps",
        "warmup",
        "iterations",
        "repeats",
        "cpu_threads",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.repeats < 3:
        parser.error("--repeats must be at least three")
    seeds = parse_seeds(args.seeds)
    candidate_sha256 = sha256_file(args.candidate_extension)
    direct_sha256 = sha256_file(args.direct_extension)
    if candidate_sha256 == direct_sha256:
        parser.error("candidate and direct extensions must differ")

    torch.set_grad_enabled(False)
    torch.set_num_threads(args.cpu_threads)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/CoreX device is unavailable")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    compensated = load_extension(
        "corex_moe_compensated_w13",
        args.candidate_extension,
    )
    direct = load_extension(
        "corex_moe_direct_routed_compensated_gate",
        args.direct_extension,
    )
    if not hasattr(compensated, "w13") or not hasattr(direct, "w13"):
        raise RuntimeError("both extensions must expose w13")

    fixed: dict[str, Any] = {}
    sequence: dict[str, Any] = {}
    timing_state: tuple[torch.Tensor, ...] | None = None
    for seed in seeds:
        fixed_row, sequence_row, timing_state = fixture(
            seed=seed,
            sequence_steps=args.sequence_steps,
            device=device,
            direct=direct,
            compensated=compensated,
        )
        fixed[str(seed)] = fixed_row
        sequence[str(seed)] = sequence_row
    if timing_state is None:
        raise RuntimeError("no timing fixture was generated")
    hidden_states, router_logits, w13, expert_ids, selected_w13 = timing_state

    def vendor_fixed() -> torch.Tensor:
        return F.linear(
            hidden_states,
            selected_w13.reshape(-1, HIDDEN),
        ).reshape(TOP_K, W13_ROWS)

    def direct_fixed() -> torch.Tensor:
        return direct.w13(hidden_states, w13, expert_ids)

    def compensated_fixed() -> torch.Tensor:
        return compensated.w13(hidden_states, w13, expert_ids)

    def vendor_routed() -> torch.Tensor:
        current_ids = route_ids(router_logits)
        current_w13 = torch.index_select(w13, 0, current_ids)
        return F.linear(
            hidden_states,
            current_w13.reshape(-1, HIDDEN),
        ).reshape(TOP_K, W13_ROWS)

    def direct_routed() -> torch.Tensor:
        return direct.w13(
            hidden_states,
            w13,
            route_ids(router_logits),
        )

    def compensated_routed() -> torch.Tensor:
        return compensated.w13(
            hidden_states,
            w13,
            route_ids(router_logits),
        )

    cases = {
        "vendor_fixed": vendor_fixed,
        "direct_fixed": direct_fixed,
        "compensated_fixed": compensated_fixed,
        "vendor_routed": vendor_routed,
        "direct_routed": direct_routed,
        "compensated_routed": compensated_routed,
    }
    timing_cases = {
        name: measure(
            case,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        for name, case in cases.items()
    }
    speedups = {
        "compensated_fixed_vs_vendor": (
            timing_cases["vendor_fixed"]["median_ms"]
            / timing_cases["compensated_fixed"]["median_ms"]
        ),
        "compensated_routed_vs_vendor": (
            timing_cases["vendor_routed"]["median_ms"]
            / timing_cases["compensated_routed"]["median_ms"]
        ),
    }

    report = {
        "schema": SCHEMA,
        "version": 1,
        "device": torch.cuda.get_device_name(device),
        "shape": {
            "experts": EXPERTS,
            "top_k": TOP_K,
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "rows_per_expert": W13_ROWS,
            "dtype": str(torch.float16),
        },
        "config": {
            "device": args.device,
            "seeds": list(seeds),
            "sequence_steps_per_seed": args.sequence_steps,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "cpu_threads": args.cpu_threads,
            "weight_scale": 0.02,
        },
        "method": {
            "algorithm": "per_lane_kahan_fp32_then_rn_warp_tree",
            "quality_reference": "torch_nn_functional_linear_fp16",
            "exact_diagnostic":
                "cpu_float64_dot_rounded_to_fp16_fixed_fixture_only",
            "fixture_generation":
                "hidden_then_router_then_w13_then_sequence",
            "production_runtime_changed": False,
        },
        "extensions": {
            "candidate_sha256": candidate_sha256,
            "candidate_size_bytes": args.candidate_extension.stat().st_size,
            "direct_sha256": direct_sha256,
            "direct_size_bytes": args.direct_extension.stat().st_size,
        },
        "fixed": fixed,
        "sequence": sequence,
        "timings": {
            "cases": timing_cases,
            "speedups": speedups,
        },
    }
    atomic_write(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
