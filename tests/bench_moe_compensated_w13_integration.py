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


SCHEMA = "bi100-moe-compensated-w13-integration-v1"
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
        actual_cpu = actual.detach().cpu()
        expected_cpu = expected.detach().cpu()
        self.steps += 1
        self.rows += actual_cpu.numel()
        finite = bool(
            torch.isfinite(actual_cpu).all()
            and torch.isfinite(expected_cpu).all())
        self.finite_steps += int(finite)
        self.exact_steps += int(torch.equal(actual_cpu, expected_cpu))
        self.mismatch_count += int(
            torch.count_nonzero(actual_cpu != expected_cpu))
        if not finite:
            self.metrics_finite = False
            return

        delta = (actual_cpu.float() - expected_cpu.float()).abs()
        squared_error = float(delta.square().sum())
        squared_reference = float(expected_cpu.float().square().sum())
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
            relative = None
            max_step_relative = None
        else:
            max_abs = self.max_abs
            mean_abs = self.absolute_error / max(self.rows, 1)
            relative = math.sqrt(
                self.squared_error
                / max(self.squared_reference, 1.0e-30))
            max_step_relative = self.max_step_relative_l2
        return {
            "steps": self.steps,
            "rows": self.rows,
            "finite_steps": self.finite_steps,
            "exact_steps": self.exact_steps,
            "mismatch_count": self.mismatch_count,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "relative_l2": relative,
            "max_step_relative_l2": max_step_relative,
        }


def route(
    router_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_logits, expert_ids = torch.topk(
        router_logits.float(),
        TOP_K,
        dim=-1,
    )
    weights = torch.softmax(topk_logits, dim=-1).to(router_logits.dtype)
    return expert_ids[0].contiguous(), weights[0].contiguous()


def strict_routed(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    exact_reduce: Any,
    activation: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    expert_ids, weights = route(router_logits)
    selected_w13 = torch.index_select(w13, 0, expert_ids)
    selected_w2 = torch.index_select(w2, 0, expert_ids)
    gate_up = F.linear(
        hidden_states,
        selected_w13.reshape(-1, HIDDEN),
    ).reshape(TOP_K, W13_ROWS)
    activated = activation(gate_up)
    expert_output = torch.bmm(
        selected_w2,
        activated.unsqueeze(-1),
    ).squeeze(-1)
    return exact_reduce.serial_float(expert_output, weights)


def native_routed(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    extension: Any,
    activation: Callable[[torch.Tensor], torch.Tensor],
    *,
    compensated: bool,
) -> torch.Tensor:
    expert_ids, weights = route(router_logits)
    if compensated:
        gate_up = extension.w13_compensated(
            hidden_states,
            w13,
            expert_ids,
        )
    else:
        gate_up = extension.w13(hidden_states, w13, expert_ids)
    activated = activation(gate_up)
    return extension.w2_reduce(
        activated,
        w2,
        expert_ids,
        weights,
    )


def timing_summary(trials: list[float]) -> dict[str, Any]:
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


def measure_alternating(
    cases: dict[str, Callable[[], torch.Tensor]],
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> tuple[dict[str, Any], list[list[str]]]:
    names = list(cases)
    for name in names:
        for _ in range(warmup):
            cases[name]()
    torch.cuda.synchronize()

    trials = {name: [] for name in names}
    orders: list[list[str]] = []
    for repeat in range(repeats):
        order = names if repeat % 2 == 0 else list(reversed(names))
        orders.append(list(order))
        for name in order:
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(iterations):
                cases[name]()
            torch.cuda.synchronize()
            elapsed_ms = (
                (time.perf_counter() - started) * 1000.0 / iterations)
            trials[name].append(elapsed_ms)
    return (
        {name: timing_summary(values) for name, values in trials.items()},
        orders,
    )


def parse_seeds(raw: str) -> tuple[int, ...]:
    values = tuple(
        int(value.strip()) for value in raw.split(",") if value.strip())
    if len(values) < 2 or len(values) != len(set(values)):
        raise ValueError("at least two unique seeds are required")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--exact-reduce-extension", type=Path, required=True)
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
    if args.repeats != 9:
        parser.error("--repeats must be exactly nine")
    seeds = parse_seeds(args.seeds)

    torch.set_grad_enabled(False)
    torch.set_num_threads(args.cpu_threads)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/CoreX device is unavailable")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    extension = load_extension(
        "corex_moe_direct_routed",
        args.extension,
    )
    exact_reduce = load_extension(
        "corex_moe_exact_reduce",
        args.exact_reduce_extension,
    )
    from vllm.model_executor.layers.activation import SiluAndMul
    activation = SiluAndMul()
    for symbol in ("w13", "w13_compensated", "w2_reduce"):
        if not callable(getattr(extension, symbol, None)):
            raise RuntimeError(f"production extension lacks {symbol}")
    if not callable(getattr(exact_reduce, "serial_float", None)):
        raise RuntimeError("exact-reduce extension lacks serial_float")

    sequence_report: dict[str, Any] = {}
    fixed_report: dict[str, Any] = {}
    timing_state: tuple[torch.Tensor, ...] | None = None
    dtype = torch.float16
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(seed)
        w13 = (
            torch.randn(
                (EXPERTS, W13_ROWS, HIDDEN),
                device=device,
                dtype=dtype,
                generator=generator,
            )
            * 0.02
        ).contiguous()
        w2 = (
            torch.randn(
                (EXPERTS, HIDDEN, INTERMEDIATE),
                device=device,
                dtype=dtype,
                generator=generator,
            )
            * 0.02
        ).contiguous()
        hidden_states = torch.randn(
            (1, HIDDEN),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        router_logits = torch.randn(
            (1, EXPERTS),
            device=device,
            dtype=dtype,
            generator=generator,
        )

        reference = strict_routed(
            hidden_states,
            router_logits,
            w13,
            w2,
            exact_reduce,
            activation,
        )
        direct = native_routed(
            hidden_states,
            router_logits,
            w13,
            w2,
            extension,
            activation,
            compensated=False,
        )
        candidate = native_routed(
            hidden_states,
            router_logits,
            w13,
            w2,
            extension,
            activation,
            compensated=True,
        )
        direct_repeat = native_routed(
            hidden_states,
            router_logits,
            w13,
            w2,
            extension,
            activation,
            compensated=False,
        )
        candidate_repeat = native_routed(
            hidden_states,
            router_logits,
            w13,
            w2,
            extension,
            activation,
            compensated=True,
        )
        torch.cuda.synchronize()
        fixed_report[str(seed)] = {
            "direct_vs_reference": comparison(
                direct.detach().cpu(),
                reference.detach().cpu(),
            ),
            "candidate_vs_reference": comparison(
                candidate.detach().cpu(),
                reference.detach().cpu(),
            ),
            "candidate_vs_direct": comparison(
                candidate.detach().cpu(),
                direct.detach().cpu(),
            ),
            "direct_repeat_exact": bool(
                torch.equal(direct, direct_repeat)),
            "candidate_repeat_exact": bool(
                torch.equal(candidate, candidate_repeat)),
        }

        direct_accumulator = SequenceAccumulator()
        candidate_accumulator = SequenceAccumulator()
        for _ in range(args.sequence_steps):
            step_hidden = torch.randn(
                (1, HIDDEN),
                device=device,
                dtype=dtype,
                generator=generator,
            )
            step_logits = torch.randn(
                (1, EXPERTS),
                device=device,
                dtype=dtype,
                generator=generator,
            )
            step_reference = strict_routed(
                step_hidden,
                step_logits,
                w13,
                w2,
                exact_reduce,
                activation,
            )
            step_direct = native_routed(
                step_hidden,
                step_logits,
                w13,
                w2,
                extension,
                activation,
                compensated=False,
            )
            step_candidate = native_routed(
                step_hidden,
                step_logits,
                w13,
                w2,
                extension,
                activation,
                compensated=True,
            )
            direct_accumulator.add(step_direct, step_reference)
            candidate_accumulator.add(step_candidate, step_reference)
        sequence_report[str(seed)] = {
            "direct_vs_reference": direct_accumulator.report(),
            "candidate_vs_reference": candidate_accumulator.report(),
        }
        timing_state = (
            hidden_states,
            router_logits,
            w13,
            w2,
        )

    if timing_state is None:
        raise RuntimeError("no timing fixture was generated")
    hidden_states, router_logits, w13, w2 = timing_state
    cases = {
        "strict_reference": lambda: strict_routed(
            hidden_states,
            router_logits,
            w13,
            w2,
            exact_reduce,
            activation,
        ),
        "direct_control": lambda: native_routed(
            hidden_states,
            router_logits,
            w13,
            w2,
            extension,
            activation,
            compensated=False,
        ),
        "compensated_candidate": lambda: native_routed(
            hidden_states,
            router_logits,
            w13,
            w2,
            extension,
            activation,
            compensated=True,
        ),
    }
    timing_cases, timing_orders = measure_alternating(
        cases,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    direct_median = timing_cases["direct_control"]["median_ms"]
    candidate_median = timing_cases["compensated_candidate"]["median_ms"]
    strict_median = timing_cases["strict_reference"]["median_ms"]

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
            "dtype": str(dtype),
        },
        "config": {
            "device": args.device,
            "seeds": list(seeds),
            "sequence_steps_per_seed": args.sequence_steps,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "cpu_threads": args.cpu_threads,
            "w13_weight_scale": 0.02,
            "w2_weight_scale": 0.02,
        },
        "method": {
            "reference":
                "pytorch_gather_linear_vllm_silu_and_mul_bmm_"
                "corex_serial_float_reduce",
            "control": "production_direct_w13_and_w2_reduce",
            "candidate":
                "production_compensated_w13_and_same_w2_reduce",
            "timing_order": "alternating_forward_reverse",
            "request_semantics_changed": False,
        },
        "artifacts": {
            "extension_sha256": sha256_file(args.extension),
            "extension_size_bytes": args.extension.stat().st_size,
            "exact_reduce_sha256": sha256_file(
                args.exact_reduce_extension),
            "exact_reduce_size_bytes":
                args.exact_reduce_extension.stat().st_size,
        },
        "fixed": fixed_report,
        "sequence": sequence_report,
        "timings": {
            "cases": timing_cases,
            "orders": timing_orders,
            "candidate_vs_direct_ratio": (
                candidate_median / direct_median),
            "candidate_vs_reference_speedup": (
                strict_median / candidate_median),
        },
    }
    atomic_write(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
