#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MIN_EFFECTIVE_HIT_GAIN = 0.02
MIN_WEIGHTED_SCORE_GAIN = 0.03
MAX_OUTPUT_TPS_REGRESSION = 0.02
MAX_TTFT_P90_REGRESSION = 0.02


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
    baseline_ttft = float(base["ttft_p90_all_s"])
    candidate_ttft = float(cand["ttft_p90_all_s"])

    hit_gain = candidate_hit - baseline_hit
    score_gain = (
        candidate_score / baseline_score - 1.0
        if baseline_score > 0 else float("-inf"))
    output_ratio = (
        candidate_output / baseline_output
        if baseline_output > 0 else 0.0)
    ttft_ratio = (
        candidate_ttft / baseline_ttft
        if baseline_ttft > 0 else float("inf"))
    contract_fields = (
        "path", "target", "pair", "phase", "prompt_salt",
        "rendered_tokens_local")
    baseline_contract = [
        tuple(item.get(field) for field in contract_fields)
        for item in baseline.get("requests", [])
    ]
    candidate_contract = [
        tuple(item.get(field) for field in contract_fields)
        for item in candidate.get("requests", [])
    ]

    benefit_paths = {
        "effective_hit_gain_at_least_2pp":
            hit_gain + 1e-12 >= MIN_EFFECTIVE_HIT_GAIN,
        "weighted_score_gain_at_least_3pct_without_hit_reduction": (
            score_gain + 1e-12 >= MIN_WEIGHTED_SCORE_GAIN
            and candidate_hit + 1e-12 >= baseline_hit
        ),
    }
    stage_gates = {
        "baseline_complete_matrix": bool(
            baseline["validation"]["complete_matrix"]),
        "baseline_client_server_token_count_match": bool(
            baseline["validation"]["token_count_match"]),
        "baseline_target_within_one_block": bool(
            baseline["validation"]["target_within_one_block"]),
        "baseline_cold_warm_pair_salts_match": bool(
            baseline["validation"]["cold_warm_pair_salts_match"]),
        "baseline_success_rate_at_least_99pct": float(
            baseline["validation"]["success_rate"]) >= 0.99,
        "complete_matrix": bool(
            candidate["validation"]["complete_matrix"]),
        "client_server_token_count_match": bool(
            candidate["validation"]["token_count_match"]),
        "target_within_one_block": bool(
            candidate["validation"]["target_within_one_block"]),
        "cold_warm_pair_salts_match": bool(
            candidate["validation"]["cold_warm_pair_salts_match"]),
        "request_contract_identical": bool(
            baseline_contract
            and baseline_contract == candidate_contract),
        "success_rate_at_least_99pct": float(
            candidate["validation"]["success_rate"]) >= 0.99,
        "effective_cache_hit_at_least_50pct": candidate_hit >= 0.50,
        "cache_benefit_path_qualified": any(benefit_paths.values()),
        "output_tps_p10_at_least_20": candidate_output >= 20.0,
        "output_tps_regression_at_most_2pct": (
            output_ratio + 1e-12 >= 1.0 - MAX_OUTPUT_TPS_REGRESSION),
        "ttft_p90_regression_at_most_2pct": (
            ttft_ratio <= 1.0 + MAX_TTFT_P90_REGRESSION + 1e-12),
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
            "ttft_p90_fraction": ttft_ratio - 1.0,
        },
        "benefit_paths": benefit_paths,
        "stage_gates": stage_gates,
        "stage_qualified": all(stage_gates.values()),
        "quality_nonregression_qualified": None,
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
