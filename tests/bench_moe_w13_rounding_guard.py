#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


SCHEMA = "bi100-moe-w13-rounding-guard-v1"
EXPERTS = 256
TOP_K = 8
HIDDEN = 2048
INTERMEDIATE = 128
W13_ROWS = 2 * INTERMEDIATE


def load_extension(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    delta = actual.float() - expected.float()
    squared_error = float(delta.square().sum())
    squared_reference = float(expected.float().square().sum())
    return math.sqrt(squared_error / max(squared_reference, 1.0e-30))


def comparison(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    delta = (actual.float() - expected.float()).abs()
    return {
        "exact": bool(torch.equal(actual, expected)),
        "mismatch_count": int(torch.count_nonzero(actual != expected)),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "relative_l2": relative_l2(actual, expected),
        "finite": bool(torch.isfinite(actual).all()),
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


def evaluate_step(
    *,
    direct: Any,
    probe: Any,
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    expert_ids: torch.Tensor,
) -> dict[str, Any]:
    selected_w13 = torch.index_select(w13, 0, expert_ids)
    vendor = F.linear(
        hidden_states,
        selected_w13.reshape(-1, HIDDEN),
    ).reshape(TOP_K, W13_ROWS)
    direct_half = direct.w13(hidden_states, w13, expert_ids)
    forward_sums, reverse_sums = probe.dual_w13_sums(
        hidden_states, w13, expert_ids)
    forward_half = forward_sums.to(torch.float16)
    reverse_half = reverse_sums.to(torch.float16)
    if not torch.equal(forward_half, direct_half):
        raise RuntimeError(
            "rounding probe forward order differs from production direct W13")

    exact_half = exact_half_dot(selected_w13, hidden_states)
    vendor_cpu = vendor.detach().cpu()
    direct_cpu = direct_half.detach().cpu()
    reverse_cpu = reverse_half.detach().cpu()
    disagreement = direct_cpu != reverse_cpu
    vendor_mismatch = direct_cpu != vendor_cpu

    corrected = direct_cpu.clone()
    corrected[disagreement] = exact_half[disagreement]
    missed = vendor_mismatch & ~disagreement
    false_positive = disagreement & ~vendor_mismatch
    exact_flag_mismatch = disagreement & (exact_half != vendor_cpu)

    rows = direct_cpu.numel()
    flags = int(torch.count_nonzero(disagreement))
    mismatches = int(torch.count_nonzero(vendor_mismatch))
    true_positive = int(torch.count_nonzero(disagreement & vendor_mismatch))
    return {
        "rows": rows,
        "finite": bool(
            torch.isfinite(forward_sums).all()
            and torch.isfinite(reverse_sums).all()
            and torch.isfinite(vendor).all()),
        "production_forward_exact": True,
        "flags": flags,
        "flagged_fraction": flags / rows,
        "vendor_mismatches": mismatches,
        "flagged_vendor_mismatches": true_positive,
        "missed_vendor_mismatches": int(torch.count_nonzero(missed)),
        "false_positive_flags": int(torch.count_nonzero(false_positive)),
        "mismatch_recall": (
            1.0 if mismatches == 0 else true_positive / mismatches),
        "flag_precision": (
            1.0 if flags == 0 else true_positive / flags),
        "exact_flag_mismatches": int(torch.count_nonzero(exact_flag_mismatch)),
        "direct": comparison(direct_cpu, vendor_cpu),
        "reverse": comparison(reverse_cpu, vendor_cpu),
        "exact_half": comparison(exact_half, vendor_cpu),
        "corrected": comparison(corrected, vendor_cpu),
    }


class SequenceAccumulator:
    def __init__(self) -> None:
        self.steps = 0
        self.rows = 0
        self.flags = 0
        self.vendor_mismatches = 0
        self.flagged_vendor_mismatches = 0
        self.missed_vendor_mismatches = 0
        self.false_positive_flags = 0
        self.exact_flag_mismatches = 0
        self.finite_steps = 0
        self.max_flagged_fraction = 0.0
        self.max_corrected_step_relative_l2 = 0.0
        self.max_exact_step_relative_l2 = 0.0
        self.direct_squared_error = 0.0
        self.exact_squared_error = 0.0
        self.corrected_squared_error = 0.0
        self.squared_reference = 0.0

    def add(self, step: dict[str, Any]) -> None:
        self.steps += 1
        self.rows += step["rows"]
        self.flags += step["flags"]
        self.vendor_mismatches += step["vendor_mismatches"]
        self.flagged_vendor_mismatches += step["flagged_vendor_mismatches"]
        self.missed_vendor_mismatches += step["missed_vendor_mismatches"]
        self.false_positive_flags += step["false_positive_flags"]
        self.exact_flag_mismatches += step["exact_flag_mismatches"]
        self.finite_steps += int(step["finite"])
        self.max_flagged_fraction = max(
            self.max_flagged_fraction, step["flagged_fraction"])
        self.max_corrected_step_relative_l2 = max(
            self.max_corrected_step_relative_l2,
            step["corrected"]["relative_l2"])
        self.max_exact_step_relative_l2 = max(
            self.max_exact_step_relative_l2,
            step["exact_half"]["relative_l2"])

        for name, target in (
            ("direct", "direct_squared_error"),
            ("exact_half", "exact_squared_error"),
            ("corrected", "corrected_squared_error"),
        ):
            relative = step[name]["relative_l2"]
            # Per-step reference norms are not exposed in the privacy-safe
            # report, so aggregate the squared relative errors uniformly.
            setattr(
                self,
                target,
                getattr(self, target) + relative * relative,
            )
        self.squared_reference += 1.0

    def report(self) -> dict[str, Any]:
        mismatch_recall = (
            1.0 if self.vendor_mismatches == 0
            else self.flagged_vendor_mismatches / self.vendor_mismatches)
        flag_precision = (
            1.0 if self.flags == 0
            else self.flagged_vendor_mismatches / self.flags)
        return {
            "steps": self.steps,
            "rows": self.rows,
            "finite_steps": self.finite_steps,
            "flags": self.flags,
            "flagged_fraction": self.flags / max(self.rows, 1),
            "max_step_flagged_fraction": self.max_flagged_fraction,
            "vendor_mismatches": self.vendor_mismatches,
            "flagged_vendor_mismatches": self.flagged_vendor_mismatches,
            "missed_vendor_mismatches": self.missed_vendor_mismatches,
            "false_positive_flags": self.false_positive_flags,
            "mismatch_recall": mismatch_recall,
            "flag_precision": flag_precision,
            "exact_flag_mismatches": self.exact_flag_mismatches,
            "direct_rms_step_relative_l2": math.sqrt(
                self.direct_squared_error / max(self.squared_reference, 1.0)),
            "exact_rms_step_relative_l2": math.sqrt(
                self.exact_squared_error / max(self.squared_reference, 1.0)),
            "corrected_rms_step_relative_l2": math.sqrt(
                self.corrected_squared_error
                / max(self.squared_reference, 1.0)),
            "max_exact_step_relative_l2":
                self.max_exact_step_relative_l2,
            "max_corrected_step_relative_l2":
                self.max_corrected_step_relative_l2,
        }


def parse_seeds(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",")
                   if value.strip())
    if len(values) < 2 or len(values) != len(set(values)):
        raise ValueError("at least two unique seeds are required")
    return values


def run_fixture(
    *,
    seed: int,
    sequence_steps: int,
    device: torch.device,
    direct: Any,
    probe: Any,
) -> dict[str, Any]:
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

    started = time.perf_counter()
    fixed = evaluate_step(
        direct=direct,
        probe=probe,
        hidden_states=hidden_states,
        w13=w13,
        expert_ids=route_ids(router_logits),
    )
    sequence = SequenceAccumulator()
    for _ in range(sequence_steps):
        step_hidden = torch.randn(
            (1, HIDDEN), device=device, dtype=dtype, generator=generator)
        step_logits = torch.randn(
            (1, EXPERTS), device=device, dtype=dtype, generator=generator)
        sequence.add(evaluate_step(
            direct=direct,
            probe=probe,
            hidden_states=step_hidden,
            w13=w13,
            expert_ids=route_ids(step_logits),
        ))
    torch.cuda.synchronize()
    return {
        "seed": seed,
        "fixed": fixed,
        "sequence": sequence.report(),
        "elapsed_s": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-extension", type=Path, required=True)
    parser.add_argument("--direct-extension", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", default="20260716,20260727")
    parser.add_argument("--sequence-steps", type=int, default=500)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.sequence_steps <= 0:
        parser.error("--sequence-steps must be positive")
    if args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")
    seeds = parse_seeds(args.seeds)

    torch.set_grad_enabled(False)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    direct = load_extension(
        "corex_moe_direct_routed", args.direct_extension)
    probe = load_extension(
        "corex_moe_w13_rounding_probe", args.probe_extension)
    if not hasattr(direct, "w13"):
        raise RuntimeError("direct extension does not expose w13")
    if not hasattr(probe, "dual_w13_sums"):
        raise RuntimeError("probe extension does not expose dual_w13_sums")

    fixtures = [
        run_fixture(
            seed=seed,
            sequence_steps=args.sequence_steps,
            device=device,
            direct=direct,
            probe=probe,
        )
        for seed in seeds
    ]
    report = {
        "schema": SCHEMA,
        "version": 1,
        "device": torch.cuda.get_device_name(device),
        "shape": {
            "experts": EXPERTS,
            "top_k": TOP_K,
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "dtype": str(torch.float16),
        },
        "config": {
            "device": args.device,
            "seeds": list(seeds),
            "sequence_steps_per_seed": args.sequence_steps,
            "cpu_threads": args.cpu_threads,
        },
        "method": {
            "flag_rule": "forward_fp16_differs_from_reverse_fp16",
            "correction_oracle":
                "float64_dot_rounded_to_fp16_for_flagged_rows",
            "fixture_generation":
                "hidden_then_router_then_w13_then_sequence",
            "production_runtime_changed": False,
        },
        "fixtures": fixtures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
