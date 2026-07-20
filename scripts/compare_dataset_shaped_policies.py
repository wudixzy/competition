#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_summary(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("validation", {}).get("complete_matrix"):
        raise ValueError(f"incomplete matrix: {path}")
    return report


def compare(baseline: dict[str, Any],
            candidate: dict[str, Any]) -> dict[str, Any]:
    base = baseline["aggregate"]
    cand = candidate["aggregate"]
    baseline_output = float(base["output_tps_p10"])
    candidate_output = float(cand["output_tps_p10"])
    baseline_hit = float(base["cache_hit_rate"])
    candidate_hit = float(cand["cache_hit_rate"])
    baseline_score = float(base["weighted_score"])
    candidate_score = float(cand["weighted_score"])

    hit_gain = candidate_hit - baseline_hit
    score_gain = (
        candidate_score / baseline_score - 1.0
        if baseline_score > 0 else float("-inf"))
    output_ratio = (
        candidate_output / baseline_output
        if baseline_output > 0 else 0.0)

    stage_gates = {
        "complete_matrix": bool(
            candidate["validation"]["complete_matrix"]),
        "client_server_token_count_match": bool(
            candidate["validation"]["token_count_match"]),
        "target_within_one_block": bool(
            candidate["validation"]["target_within_one_block"]),
        "success_rate_at_least_99pct": float(
            candidate["validation"]["success_rate"]) >= 0.99,
        "effective_hit_gain_at_least_5pp": hit_gain + 1e-12 >= 0.05,
        "weighted_score_gain_at_least_5pct": score_gain + 1e-12 >= 0.05,
        "output_tps_p10_at_least_20": candidate_output >= 20.0,
        "output_tps_regression_at_most_2pct": output_ratio + 1e-12 >= 0.98,
    }
    final_metric_gates = {
        "output_tps_p10_at_least_20": candidate_output >= 20.0,
        "ttft_p90_at_most_5s": float(cand["ttft_p90_all_s"]) <= 5.0,
        "effective_cache_hit_at_least_50pct": candidate_hit >= 0.50,
        "success_rate_at_least_99pct": float(
            candidate["validation"]["success_rate"]) >= 0.99,
        "weighted_score_at_least_8000": candidate_score >= 8000.0,
    }
    return {
        "baseline": base,
        "candidate": cand,
        "delta": {
            "effective_hit_percentage_points": hit_gain * 100.0,
            "weighted_score_fraction": score_gain,
            "output_tps_fraction": output_ratio - 1.0,
        },
        "stage_gates": stage_gates,
        "stage_qualified": all(stage_gates.values()),
        "final_metric_gates": final_metric_gates,
        "final_metric_gates_passed": all(final_metric_gates.values()),
        "capacity_256k_preserved": None,
        "final_qualified": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = compare(
        load_summary(args.baseline), load_summary(args.candidate))
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["stage_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
