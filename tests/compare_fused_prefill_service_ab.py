#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


MIN_COLD_IMPROVEMENT = 0.20
MAX_WARM_REGRESSION = 0.02
MIN_OUTPUT_TPS_P10 = 20.0
MAX_OUTPUT_REGRESSION = 0.02


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _finite_positive(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def _indexed_cases(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result = {}
    for case in report.get("cases") or []:
        if not isinstance(case, dict):
            continue
        target = case.get("target_prompt_tokens")
        if isinstance(target, int) and not isinstance(target, bool):
            result[target] = case
    return result


def compare(control: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    rows = []
    for name, report, expected_mode in (
            ("control", control, "control"),
            ("candidate", candidate, "candidate")):
        if report.get("schema") != "bi100-m1-47-service-measurement-v1":
            reasons.append(f"{name} schema mismatch")
        if report.get("mode") != expected_mode:
            reasons.append(f"{name} mode mismatch")
        if report.get("qualified_measurement") is not True:
            reasons.append(f"{name} measurement did not qualify")
    if control.get("run_id") != candidate.get("run_id"):
        reasons.append("run_id mismatch")
    if control.get("max_tokens") != candidate.get("max_tokens"):
        reasons.append("max_tokens mismatch")

    control_cases = _indexed_cases(control)
    candidate_cases = _indexed_cases(candidate)
    if set(control_cases) != set(candidate_cases) or not control_cases:
        reasons.append("target sets differ or are empty")
    for target in sorted(set(control_cases) & set(candidate_cases)):
        control_case = control_cases[target]
        candidate_case = candidate_cases[target]
        control_cold = (control_case.get("cold") or {}).get("ttft_s")
        candidate_cold = (candidate_case.get("cold") or {}).get("ttft_s")
        control_warm = control_case.get("warm_ttft_median_s")
        candidate_warm = candidate_case.get("warm_ttft_median_s")
        if not all(_finite_positive(value) for value in (
                control_cold, candidate_cold, control_warm, candidate_warm)):
            reasons.append(f"target {target} contains invalid TTFT")
            continue
        cold_improvement = 1.0 - candidate_cold / control_cold
        warm_regression = candidate_warm / control_warm - 1.0
        if cold_improvement + 1e-12 < MIN_COLD_IMPROVEMENT:
            reasons.append(
                f"target {target} cold TTFT improvement "
                f"{cold_improvement:.3%} is below 20%")
        if warm_regression > MAX_WARM_REGRESSION + 1e-12:
            reasons.append(
                f"target {target} warm TTFT regression "
                f"{warm_regression:.3%} exceeds 2%")
        rows.append({
            "target_prompt_tokens": target,
            "control_cold_ttft_s": control_cold,
            "candidate_cold_ttft_s": candidate_cold,
            "cold_improvement": cold_improvement,
            "control_warm_ttft_median_s": control_warm,
            "candidate_warm_ttft_median_s": candidate_warm,
            "warm_regression": warm_regression,
        })

    control_output = control.get("output_tps_p10")
    candidate_output = candidate.get("output_tps_p10")
    output_regression = None
    if not all(_finite_positive(value) for value in (
            control_output, candidate_output)):
        reasons.append("invalid Output TPS P10")
    else:
        output_regression = 1.0 - candidate_output / control_output
        if candidate_output < MIN_OUTPUT_TPS_P10:
            reasons.append("candidate Output TPS P10 is below 20")
        if output_regression > MAX_OUTPUT_REGRESSION + 1e-12:
            reasons.append("candidate Output TPS P10 regressed by more than 2%")

    return {
        "schema": "bi100-m1-47-service-ab-v1",
        "thresholds": {
            "minimum_cold_ttft_improvement": MIN_COLD_IMPROVEMENT,
            "maximum_warm_ttft_regression": MAX_WARM_REGRESSION,
            "minimum_output_tps_p10": MIN_OUTPUT_TPS_P10,
            "maximum_output_tps_regression": MAX_OUTPUT_REGRESSION,
        },
        "rows": rows,
        "control_output_tps_p10": control_output,
        "candidate_output_tps_p10": candidate_output,
        "output_tps_regression": output_regression,
        "qualified": not reasons,
        "reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(_load(args.control), _load(args.candidate))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
