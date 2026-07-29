#!/usr/bin/env python3
"""Paired binary non-inferiority statistics for capability A/B results."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Sequence


INPUT_SCHEMA = "bi100-paired-binary-outcomes-v1"
REPORT_SCHEMA = "bi100-paired-noninferiority-v1"
VERSION = 1
Json = dict[str, Any]


def exit_code(status: str) -> int:
    return {"pass": 0, "fail": 1, "inconclusive": 3}[status]


def _atomic_write(path: Path, value: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=True, indent=2,
                      sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def minimum_zero_regression_samples(
    margin: float,
    confidence: float,
) -> int:
    """Samples needed for a zero-event one-sided exact upper bound."""
    if not 0.0 < margin < 1.0:
        raise ValueError("margin must be between zero and one")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and one")
    alpha = 1.0 - confidence
    return math.ceil(math.log(alpha) / math.log(1.0 - margin))


def _lower_percentile(values: list[float], alpha: float) -> float:
    if not values:
        raise ValueError("cannot take a percentile of an empty sequence")
    ordered = sorted(values)
    index = max(0, math.ceil(alpha * len(ordered)) - 1)
    return ordered[index]


def _mcnemar_regression_p(
    baseline_only: int,
    candidate_only: int,
) -> float:
    """One-sided exact P(B-only >= observed | discordant, p=0.5)."""
    discordant = baseline_only + candidate_only
    if discordant == 0:
        return 1.0
    numerator = sum(
        math.comb(discordant, value)
        for value in range(baseline_only, discordant + 1)
    )
    return numerator / (2 ** discordant)


def _validate_outcomes(
    baseline: Sequence[bool],
    candidate: Sequence[bool],
) -> None:
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("paired outcomes must be non-empty and equal length")
    if any(type(value) is not bool for value in baseline):
        raise ValueError("baseline outcomes must be booleans")
    if any(type(value) is not bool for value in candidate):
        raise ValueError("candidate outcomes must be booleans")


def paired_noninferiority(
    baseline: Sequence[bool],
    candidate: Sequence[bool],
    *,
    margin: float,
    confidence: float,
    bootstrap_samples: int,
    seed: int,
    mode: str = "noninferiority",
) -> Json:
    _validate_outcomes(baseline, candidate)
    if mode not in {"contract", "noninferiority"}:
        raise ValueError("mode must be contract or noninferiority")
    if not 0.0 < margin < 1.0:
        raise ValueError("margin must be between zero and one")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and one")
    if bootstrap_samples < 1000:
        raise ValueError("bootstrap_samples must be at least 1000")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    pairs = list(zip(baseline, candidate))
    both_pass = sum(left and right for left, right in pairs)
    baseline_only = sum(left and not right for left, right in pairs)
    candidate_only = sum(not left and right for left, right in pairs)
    both_fail = len(pairs) - both_pass - baseline_only - candidate_only
    differences = [
        int(right) - int(left)
        for left, right in pairs
    ]
    observed_delta = sum(differences) / len(differences)
    rng = random.Random(seed)
    bootstrap_deltas = []
    for _ in range(bootstrap_samples):
        bootstrap_deltas.append(sum(
            differences[rng.randrange(len(differences))]
            for _ in differences
        ) / len(differences))
    alpha = 1.0 - confidence
    lower_bound = _lower_percentile(bootstrap_deltas, alpha)
    minimum_samples = minimum_zero_regression_samples(margin, confidence)

    if mode == "contract":
        qualified = baseline_only == 0
        status = "pass" if qualified else "fail"
        reasons = (
            []
            if qualified
            else ["candidate failed one or more baseline-passing contracts"]
        )
    else:
        reasons = []
        if len(pairs) < minimum_samples:
            reasons.append(
                "sample count is below the zero-regression power floor")
        if lower_bound < -margin:
            reasons.append(
                "paired bootstrap lower bound crosses the margin")
        qualified = not reasons
        status = "pass" if qualified else (
            "inconclusive"
            if len(pairs) < minimum_samples
            and lower_bound >= -margin
            else "fail"
        )

    return {
        "schema": REPORT_SCHEMA,
        "version": VERSION,
        "mode": mode,
        "status": status,
        "qualified": qualified,
        "sample_count": len(pairs),
        "baseline_pass_rate": sum(baseline) / len(baseline),
        "candidate_pass_rate": sum(candidate) / len(candidate),
        "candidate_minus_baseline": observed_delta,
        "paired_counts": {
            "both_pass": both_pass,
            "baseline_only": baseline_only,
            "candidate_only": candidate_only,
            "both_fail": both_fail,
        },
        "statistics": {
            "confidence": confidence,
            "noninferiority_margin": margin,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": seed,
            "one_sided_lower_bound": lower_bound,
            "minimum_zero_regression_samples": minimum_samples,
            "mcnemar_regression_p": _mcnemar_regression_p(
                baseline_only, candidate_only),
        },
        "reasons": reasons,
        "authorization": {
            "capability_surface_authorized": qualified,
            "overall_promotion_authorized": False,
        },
        "privacy": {
            "contains_sample_outcomes": False,
            "contains_prompts": False,
            "contains_model_outputs": False,
            "contains_token_ids": False,
            "contains_credentials": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--mode", choices=("contract", "noninferiority"),
                        default="noninferiority")
    parser.add_argument("--margin", type=float, required=True)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    value = json.loads(args.input.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema") != INPUT_SCHEMA
        or value.get("version") != 1
        or not isinstance(value.get("baseline"), list)
        or not isinstance(value.get("candidate"), list)
    ):
        parser.error("input is not a paired binary outcome report")
    report = paired_noninferiority(
        value["baseline"],
        value["candidate"],
        margin=args.margin,
        confidence=args.confidence,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        mode=args.mode,
    )
    report["dataset"] = value.get("dataset")
    _atomic_write(args.out, report)
    return exit_code(report["status"])


if __name__ == "__main__":
    raise SystemExit(main())
