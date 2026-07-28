#!/usr/bin/env python3
"""Non-qualifying standard-library screen for compensated W13 summation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import struct
import tempfile
import time
from pathlib import Path
from typing import Any


SCHEMA = "nonqualifying-cpu-compensated-w13-screen-v1"
SEED = 20260727
STEPS = 16
ROWS_PER_STEP = 256
HIDDEN = 2048
LANES = 32
WEIGHT_SCALE = 0.02


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f16(value: float) -> float:
    return struct.unpack("<e", struct.pack("<e", value))[0]


def warp_reduce(values: list[float]) -> float:
    result = list(values)
    if len(result) != LANES:
        raise ValueError(f"warp reduction requires exactly {LANES} lanes")
    for offset in (16, 8, 4, 2, 1):
        for lane in range(offset):
            result[lane] = f32(result[lane] + result[lane + offset])
    return result[0]


def simulate_row(
    input_values: list[float],
    generator: random.Random,
) -> tuple[float, float, float]:
    if len(input_values) != HIDDEN:
        raise ValueError(f"input must contain exactly {HIDDEN} values")
    direct = [0.0] * LANES
    compensated = [0.0] * LANES
    correction = [0.0] * LANES
    exact_products = []
    for index, input_value in enumerate(input_values):
        lane = index % LANES
        weight = f16(generator.gauss(0.0, 1.0) * WEIGHT_SCALE)
        # FP16 * FP16 is exactly representable in FP32, so this separate
        # product has the same per-add rounding as the production FMA.
        product = f32(weight * input_value)
        exact_products.append(weight * input_value)

        direct[lane] = f32(direct[lane] + product)

        adjusted = f32(product - correction[lane])
        next_sum = f32(compensated[lane] + adjusted)
        correction[lane] = f32(
            f32(next_sum - compensated[lane]) - adjusted)
        compensated[lane] = next_sum

    return (
        f16(warp_reduce(direct)),
        f16(warp_reduce(compensated)),
        f16(math.fsum(exact_products)),
    )


def atomic_write(path: Path, report: dict[str, Any]) -> None:
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
                report,
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


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def run() -> dict[str, Any]:
    generator = random.Random(SEED)
    stats = {
        name: {
            "squared_error": 0.0,
            "squared_reference": 0.0,
            "mismatch_count": 0,
            "max_step_relative_l2": 0.0,
        }
        for name in ("direct", "compensated")
    }
    started = time.perf_counter()
    for _ in range(STEPS):
        inputs = [
            f16(generator.gauss(0.0, 1.0))
            for _ in range(HIDDEN)
        ]
        outputs = [
            simulate_row(inputs, generator)
            for _ in range(ROWS_PER_STEP)
        ]
        squared_reference = math.fsum(
            values[2] ** 2 for values in outputs)
        for index, name in ((0, "direct"), (1, "compensated")):
            squared_error = math.fsum(
                (values[index] - values[2]) ** 2
                for values in outputs
            )
            stats[name]["squared_error"] += squared_error
            stats[name]["squared_reference"] += squared_reference
            stats[name]["mismatch_count"] += sum(
                values[index] != values[2] for values in outputs)
            stats[name]["max_step_relative_l2"] = max(
                stats[name]["max_step_relative_l2"],
                math.sqrt(
                    squared_error / max(squared_reference, 1.0e-30)),
            )

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "version": 1,
        "qualified": False,
        "warning":
            "not CoreX vendor evidence and not a qualification result",
        "config": {
            "seed": SEED,
            "steps": STEPS,
            "rows_per_step": ROWS_PER_STEP,
            "hidden": HIDDEN,
            "lanes": LANES,
            "weight_scale": WEIGHT_SCALE,
        },
        "method": {
            "direct": "fp32_lane_serial_then_fp32_warp_tree",
            "compensated":
                "fp32_kahan_lane_serial_then_fp32_warp_tree",
            "reference": "math_fsum_rounded_to_fp16",
            "input_dtype": "simulated_ieee_fp16",
            "output_dtype": "simulated_ieee_fp16",
        },
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "script_sha256": source_sha256(),
        },
        "elapsed_s": time.perf_counter() - started,
    }
    for name, values in stats.items():
        report[name] = {
            "relative_l2": math.sqrt(
                values["squared_error"]
                / max(values["squared_reference"], 1.0e-30)),
            "max_step_relative_l2":
                values["max_step_relative_l2"],
            "mismatch_count": values["mismatch_count"],
            "row_count": STEPS * ROWS_PER_STEP,
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    atomic_write(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
