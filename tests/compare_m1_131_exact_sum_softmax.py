#!/usr/bin/env python3
"""Aggregate the fixed four-cell M1-131 exact-sum component gate."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from tests import bench_m1_131_exact_sum_softmax as benchmark


SCHEMA = "bi100-m1-131-exact-sum-softmax-ab-v1"
MIN_MEDIAN_SPEEDUP = 1.10
MIN_POSITIVE_CASES = 3
MAX_SINGLE_CASE_REGRESSION = 0.02


def compare(cells: list[Any]) -> dict[str, Any]:
    reasons: list[str] = []
    rows: list[dict[str, Any]] = []
    expected_cases = set(benchmark.CASES)
    observed_cases: set[str] = set()
    source_commits: set[str] = set()
    instances: set[str] = set()
    runtime_identities: set[str] = set()
    visible_gpus: set[int] = set()
    control_hashes: set[str] = set()
    candidate_hashes: set[str] = set()
    speedups: list[float] = []

    if len(cells) != len(expected_cases):
        reasons.append(
            f"expected {len(expected_cases)} cells, got {len(cells)}"
        )
    for index, cell in enumerate(cells):
        evaluation = benchmark.evaluate_cell(cell)
        case_name = cell.get("case") if isinstance(cell, dict) else None
        label = case_name if isinstance(case_name, str) else f"cell[{index}]"
        if not evaluation["qualified"]:
            reasons.extend(
                f"{label}: {reason}" for reason in evaluation["reasons"]
            )
        if case_name in observed_cases:
            reasons.append(f"{label}: duplicate case")
        if isinstance(case_name, str):
            observed_cases.add(case_name)
        if not isinstance(cell, dict):
            continue

        source_commit = cell.get("source_commit")
        if not isinstance(source_commit, str) or not source_commit:
            reasons.append(f"{label}: source_commit is invalid")
        else:
            source_commits.add(source_commit)
        instance = cell.get("instance")
        if isinstance(instance, str) and instance:
            instances.add(instance)
        runtime_identity = cell.get("runtime_identity")
        if isinstance(runtime_identity, str) and runtime_identity:
            runtime_identities.add(runtime_identity)
        visible_gpu = cell.get("visible_physical_gpu")
        if (
            isinstance(visible_gpu, int)
            and not isinstance(visible_gpu, bool)
            and visible_gpu in range(4)
        ):
            if visible_gpu in visible_gpus:
                reasons.append(f"{label}: duplicate physical GPU")
            visible_gpus.add(visible_gpu)
        extensions = cell.get("extensions")
        if isinstance(extensions, dict):
            control = extensions.get("control")
            candidate = extensions.get("candidate")
            if isinstance(control, dict):
                digest = control.get("sha256")
                if isinstance(digest, str):
                    control_hashes.add(digest)
            if isinstance(candidate, dict):
                digest = candidate.get("sha256")
                if isinstance(digest, str):
                    candidate_hashes.add(digest)
        timings = cell.get("timings")
        speedup = (
            timings.get("control_over_candidate_speedup")
            if isinstance(timings, dict)
            else None
        )
        if (
            not isinstance(speedup, (int, float))
            or isinstance(speedup, bool)
            or not math.isfinite(float(speedup))
            or speedup <= 0
        ):
            reasons.append(f"{label}: speedup is invalid")
            continue
        speedups.append(float(speedup))
        numerical = cell.get("numerical", {})
        control_timing = timings.get("control")
        candidate_timing = timings.get("candidate")
        rows.append(
            {
                "case": case_name,
                "control_ms": (
                    control_timing.get("cuda_median_ms")
                    if isinstance(control_timing, dict)
                    else None
                ),
                "candidate_ms": (
                    candidate_timing.get("cuda_median_ms")
                    if isinstance(candidate_timing, dict)
                    else None
                ),
                "control_over_candidate_speedup": float(speedup),
                "output_exact": (
                    numerical.get("output_exact")
                    if isinstance(numerical, dict)
                    else None
                ),
                "lse_exact": (
                    numerical.get("lse_exact")
                    if isinstance(numerical, dict)
                    else None
                ),
                "output_relative_l2": (
                    numerical.get("output_relative_l2")
                    if isinstance(numerical, dict)
                    else None
                ),
                "lse_relative_l2": (
                    numerical.get("lse_relative_l2")
                    if isinstance(numerical, dict)
                    else None
                ),
                "output_max_abs": (
                    numerical.get("output_max_abs")
                    if isinstance(numerical, dict)
                    else None
                ),
                "lse_max_abs": (
                    numerical.get("lse_max_abs")
                    if isinstance(numerical, dict)
                    else None
                ),
            }
        )

    missing_cases = sorted(expected_cases - observed_cases)
    extra_cases = sorted(observed_cases - expected_cases)
    if missing_cases:
        reasons.append(f"missing cases: {missing_cases}")
    if extra_cases:
        reasons.append(f"unexpected cases: {extra_cases}")
    if len(source_commits) != 1:
        reasons.append("all cells must use one source commit")
    if len(instances) != 1:
        reasons.append("all cells must use one instance")
    if len(runtime_identities) != 1:
        reasons.append("all cells must use one runtime identity")
    if visible_gpus != {0, 1, 2, 3}:
        reasons.append("cells must cover physical GPUs 0,1,2,3 exactly once")
    if len(control_hashes) != 1:
        reasons.append("all cells must use one control extension")
    elif control_hashes != {benchmark.CONTROL_EXTENSION_SHA256}:
        reasons.append("control extension is not the frozen M1-108 binary")
    if len(candidate_hashes) != 1:
        reasons.append("all cells must use one candidate extension")
    if control_hashes and control_hashes == candidate_hashes:
        reasons.append("control and candidate extension identities match")

    median_speedup = statistics.median(speedups) if speedups else None
    positive_cases = sum(value > 1.0 for value in speedups)
    if len(speedups) != len(expected_cases):
        reasons.append("not all fixed cases produced valid timings")
    else:
        if median_speedup is None or median_speedup < MIN_MEDIAN_SPEEDUP:
            reasons.append(
                "median control/candidate speedup is below "
                f"{MIN_MEDIAN_SPEEDUP:.2f}x"
            )
        if positive_cases < MIN_POSITIVE_CASES:
            reasons.append(
                f"candidate must improve at least {MIN_POSITIVE_CASES} cases"
            )
        minimum_speedup = 1.0 / (1.0 + MAX_SINGLE_CASE_REGRESSION)
        if min(speedups) < minimum_speedup:
            reasons.append("a fixed case regressed by more than 2%")

    qualified = not reasons
    return {
        "schema": SCHEMA,
        "qualified": qualified,
        "thresholds": {
            "require_control_exact_output": True,
            "maximum_relative_l2": benchmark.RELATIVE_L2_LIMIT,
            "maximum_absolute_error": benchmark.MAX_ABS_LIMIT,
            "minimum_median_control_over_candidate_speedup": (
                MIN_MEDIAN_SPEEDUP
            ),
            "minimum_positive_cases": MIN_POSITIVE_CASES,
            "maximum_single_case_regression": MAX_SINGLE_CASE_REGRESSION,
        },
        "source_commit": (
            next(iter(source_commits)) if len(source_commits) == 1 else None
        ),
        "control_extension_sha256": (
            next(iter(control_hashes)) if len(control_hashes) == 1 else None
        ),
        "candidate_extension_sha256": (
            next(iter(candidate_hashes))
            if len(candidate_hashes) == 1
            else None
        ),
        "rows": sorted(
            rows,
            key=lambda row: (
                "" if row["case"] is None else str(row["case"])
            ),
        ),
        "median_control_over_candidate_speedup": median_speedup,
        "positive_cases": positive_cases,
        "reasons": reasons,
        "decision": {
            "tp4_service_experiment_authorized": qualified,
            "main_or_yaml_change_authorized": False,
            "official_score_claim_authorized": False,
        },
    }


def load_cells(paths: list[Path]) -> tuple[list[Any], list[str]]:
    cells: list[Any] = []
    errors: list[str] = []
    for index, path in enumerate(paths):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            value = None
            errors.append(f"cell[{index}] could not be loaded as JSON")
        cells.append(value)
    return cells, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cells, input_errors = load_cells(args.cell)
    report = compare(cells)
    report["input_error_count"] = len(input_errors)
    if input_errors:
        report["qualified"] = False
        report["reasons"].extend(input_errors)
        report["decision"]["tp4_service_experiment_authorized"] = False
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "qualified": report["qualified"],
                "median_speedup": report[
                    "median_control_over_candidate_speedup"
                ],
                "positive_cases": report["positive_cases"],
                "reasons": report["reasons"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
