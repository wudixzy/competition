#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


SCHEMA = "bi100-m1-99-fused-prefill-paired-ab-v2"
MEASUREMENT_SCHEMA = "bi100-m1-47-service-measurement-v1"
PAIR_COUNT = 3
TARGETS = (65536, 235000)
LONG_TARGET = 235000
MIN_LONG_COLD_IMPROVEMENT = 0.05
MAX_SHORT_COLD_REGRESSION = 0.02
MAX_MEDIAN_WARM_REGRESSION = 0.02
MAX_SINGLE_WARM_REGRESSION = 0.05
MAX_MEDIAN_OUTPUT_REGRESSION = 0.02
MAX_SINGLE_OUTPUT_REGRESSION = 0.05


def load_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def finite_positive(value: Any, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{field} must be finite and positive")
    return float(value)


def sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def index_cases(
    report: dict[str, Any],
    field: str,
) -> dict[int, dict[str, Any]]:
    raw_cases = report.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError(f"{field}.cases must be a list")
    result = {}
    for index, case in enumerate(raw_cases):
        if not isinstance(case, dict):
            raise ValueError(f"{field}.cases[{index}] must be an object")
        target = case.get("target_prompt_tokens")
        if not isinstance(target, int) or isinstance(target, bool):
            raise ValueError(
                f"{field}.cases[{index}].target_prompt_tokens is invalid")
        if target in result:
            raise ValueError(f"{field} contains duplicate target {target}")
        result[target] = case
    if set(result) != set(TARGETS):
        raise ValueError(
            f"{field} targets differ from the fixed {list(TARGETS)} gate")
    return result


def validate_measurement(
    report: dict[str, Any],
    *,
    mode: str,
    field: str,
) -> dict[int, dict[str, Any]]:
    if report.get("schema") != MEASUREMENT_SCHEMA:
        raise ValueError(f"{field} schema mismatch")
    if report.get("mode") != mode:
        raise ValueError(f"{field} mode mismatch")
    if report.get("max_tokens") != 32:
        raise ValueError(f"{field} max_tokens differs from the fixed gate")
    if report.get("qualified_measurement") is not True:
        raise ValueError(f"{field} measurement did not qualify")
    if report.get("reasons") != []:
        raise ValueError(f"{field} contains measurement reasons")
    finite_positive(report.get("output_tps_p10"), f"{field}.output_tps_p10")
    return index_cases(report, field)


def request_identity(
    request: Any,
    field: str,
) -> tuple[str, str, int, str]:
    if not isinstance(request, dict):
        raise ValueError(f"{field} must be an object")
    output_hash = sha256(request.get("output_sha256"), f"{field}.output_sha256")
    first_token_hash = sha256(
        request.get("first_token_sha256"),
        f"{field}.first_token_sha256",
    )
    completion_tokens = request.get("completion_tokens")
    if not isinstance(completion_tokens, int) or isinstance(
        completion_tokens,
        bool,
    ):
        raise ValueError(f"{field}.completion_tokens must be an integer")
    finish_reason = request.get("finish_reason")
    if not isinstance(finish_reason, str) or not finish_reason:
        raise ValueError(f"{field}.finish_reason must be non-empty")
    return output_hash, first_token_hash, completion_tokens, finish_reason


def compare(
    controls: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    reasons: list[str] = []
    if len(controls) != PAIR_COUNT or len(candidates) != PAIR_COUNT:
        reasons.append(f"exactly {PAIR_COUNT} control/candidate pairs required")
        return result(
            reasons=reasons,
            pairs=[],
            target_summary={},
            output_summary={},
            first_generated_token_exact=False,
            completion_structure_exact=False,
            short_full_output_exact=False,
            full_output_hashes_exact=False,
            long_full_output_mismatches=[],
        )

    pairs = []
    run_ids = []
    first_generated_token_exact = True
    completion_structure_exact = True
    short_full_output_exact = True
    full_output_hashes_exact = True
    long_full_output_mismatches = []
    cold_by_target = {target: [] for target in TARGETS}
    warm_by_target = {target: [] for target in TARGETS}
    output_regressions = []
    for pair_index, (control, candidate) in enumerate(
        zip(controls, candidates),
        start=1,
    ):
        field = f"pair[{pair_index}]"
        try:
            control_cases = validate_measurement(
                control,
                mode="control",
                field=f"{field}.control",
            )
            candidate_cases = validate_measurement(
                candidate,
                mode="candidate",
                field=f"{field}.candidate",
            )
        except (KeyError, TypeError, ValueError) as error:
            reasons.append(str(error))
            continue

        control_run_id = control.get("run_id")
        candidate_run_id = candidate.get("run_id")
        if (
            not isinstance(control_run_id, str)
            or not control_run_id
            or control_run_id != candidate_run_id
        ):
            reasons.append(f"{field} run_id mismatch")
        else:
            run_ids.append(control_run_id)

        pair_rows = []
        for target in TARGETS:
            control_case = control_cases[target]
            candidate_case = candidate_cases[target]
            try:
                control_cold = finite_positive(
                    (control_case.get("cold") or {}).get("ttft_s"),
                    f"{field}.control.{target}.cold.ttft_s",
                )
                candidate_cold = finite_positive(
                    (candidate_case.get("cold") or {}).get("ttft_s"),
                    f"{field}.candidate.{target}.cold.ttft_s",
                )
                control_warm = finite_positive(
                    control_case.get("warm_ttft_median_s"),
                    f"{field}.control.{target}.warm_ttft_median_s",
                )
                candidate_warm = finite_positive(
                    candidate_case.get("warm_ttft_median_s"),
                    f"{field}.candidate.{target}.warm_ttft_median_s",
                )
                for request_name in ("cold", "warm_1", "warm_2"):
                    control_identity = request_identity(
                        control_case.get(request_name),
                        f"{field}.control.{target}.{request_name}",
                    )
                    candidate_identity = request_identity(
                        candidate_case.get(request_name),
                        f"{field}.candidate.{target}.{request_name}",
                    )
                    if (
                        candidate_identity[2:]
                        != control_identity[2:]
                    ):
                        completion_structure_exact = False
                        reasons.append(
                            f"{field} target {target} {request_name} "
                            "completion structure differs")
                    if (
                        candidate_identity[1]
                        != control_identity[1]
                    ):
                        first_generated_token_exact = False
                        reasons.append(
                            f"{field} target {target} {request_name} "
                            "first generated token differs")
                    if (
                        candidate_identity[0]
                        != control_identity[0]
                    ):
                        full_output_hashes_exact = False
                        if target == LONG_TARGET:
                            long_full_output_mismatches.append({
                                "pair": pair_index,
                                "target_prompt_tokens": target,
                                "request": request_name,
                            })
                        else:
                            short_full_output_exact = False
                            reasons.append(
                                f"{field} target {target} {request_name} "
                                "full output differs")
            except (KeyError, TypeError, ValueError) as error:
                reasons.append(str(error))
                continue

            cold_improvement = 1.0 - candidate_cold / control_cold
            warm_regression = candidate_warm / control_warm - 1.0
            cold_by_target[target].append(cold_improvement)
            warm_by_target[target].append(warm_regression)
            if warm_regression > MAX_SINGLE_WARM_REGRESSION:
                reasons.append(
                    f"{field} target {target} warm regression "
                    f"{warm_regression:.3%} exceeds 5%")
            pair_rows.append({
                "target_prompt_tokens": target,
                "control_cold_ttft_s": control_cold,
                "candidate_cold_ttft_s": candidate_cold,
                "cold_improvement": cold_improvement,
                "control_warm_ttft_median_s": control_warm,
                "candidate_warm_ttft_median_s": candidate_warm,
                "warm_regression": warm_regression,
            })

        try:
            control_output = finite_positive(
                control.get("output_tps_p10"),
                f"{field}.control.output_tps_p10",
            )
            candidate_output = finite_positive(
                candidate.get("output_tps_p10"),
                f"{field}.candidate.output_tps_p10",
            )
            output_regression = 1.0 - candidate_output / control_output
            output_regressions.append(output_regression)
            if output_regression > MAX_SINGLE_OUTPUT_REGRESSION:
                reasons.append(
                    f"{field} Output TPS regression "
                    f"{output_regression:.3%} exceeds 5%")
        except (TypeError, ValueError) as error:
            reasons.append(str(error))
            control_output = None
            candidate_output = None
            output_regression = None
        pairs.append({
            "pair": pair_index,
            "run_id": control_run_id,
            "rows": pair_rows,
            "control_output_tps_p10": control_output,
            "candidate_output_tps_p10": candidate_output,
            "output_tps_regression": output_regression,
        })

    if len(run_ids) == PAIR_COUNT and len(set(run_ids)) != PAIR_COUNT:
        reasons.append("pair run_ids must be unique")

    target_summary = {}
    for target in TARGETS:
        cold = cold_by_target[target]
        warm = warm_by_target[target]
        if len(cold) != PAIR_COUNT or len(warm) != PAIR_COUNT:
            reasons.append(f"target {target} lacks {PAIR_COUNT} valid pairs")
            continue
        cold_median = statistics.median(cold)
        warm_median = statistics.median(warm)
        positive_cold_pairs = sum(value > 0.0 for value in cold)
        if target == LONG_TARGET:
            if cold_median < MIN_LONG_COLD_IMPROVEMENT:
                reasons.append(
                    f"235K median cold improvement {cold_median:.3%} "
                    "is below 5%")
            if positive_cold_pairs < 2:
                reasons.append(
                    "235K candidate must improve at least two pairs")
        elif cold_median < -MAX_SHORT_COLD_REGRESSION:
            reasons.append(
                f"65K median cold regression {-cold_median:.3%} "
                "exceeds 2%")
        if warm_median > MAX_MEDIAN_WARM_REGRESSION:
            reasons.append(
                f"target {target} median warm regression "
                f"{warm_median:.3%} exceeds 2%")
        target_summary[str(target)] = {
            "cold_improvements": cold,
            "cold_improvement_median": cold_median,
            "positive_cold_pairs": positive_cold_pairs,
            "warm_regressions": warm,
            "warm_regression_median": warm_median,
        }

    output_summary = {}
    if len(output_regressions) != PAIR_COUNT:
        reasons.append(f"Output TPS lacks {PAIR_COUNT} valid pairs")
    else:
        output_median = statistics.median(output_regressions)
        if output_median > MAX_MEDIAN_OUTPUT_REGRESSION:
            reasons.append(
                f"median Output TPS regression {output_median:.3%} "
                "exceeds 2%")
        output_summary = {
            "regressions": output_regressions,
            "regression_median": output_median,
        }

    return result(
        reasons=reasons,
        pairs=pairs,
        target_summary=target_summary,
        output_summary=output_summary,
        first_generated_token_exact=first_generated_token_exact,
        completion_structure_exact=completion_structure_exact,
        short_full_output_exact=short_full_output_exact,
        full_output_hashes_exact=full_output_hashes_exact,
        long_full_output_mismatches=long_full_output_mismatches,
    )


def result(
    *,
    reasons: list[str],
    pairs: list[dict[str, Any]],
    target_summary: dict[str, Any],
    output_summary: dict[str, Any],
    first_generated_token_exact: bool,
    completion_structure_exact: bool,
    short_full_output_exact: bool,
    full_output_hashes_exact: bool,
    long_full_output_mismatches: list[dict[str, Any]],
) -> dict[str, Any]:
    qualified = not reasons
    required_output_gate_passed = bool(
        first_generated_token_exact
        and completion_structure_exact
        and short_full_output_exact
    )
    return {
        "schema": SCHEMA,
        "version": 1,
        "thresholds": {
            "pair_count": PAIR_COUNT,
            "long_target_prompt_tokens": LONG_TARGET,
            "minimum_long_cold_improvement":
                MIN_LONG_COLD_IMPROVEMENT,
            "maximum_short_cold_regression":
                MAX_SHORT_COLD_REGRESSION,
            "maximum_median_warm_regression":
                MAX_MEDIAN_WARM_REGRESSION,
            "maximum_single_warm_regression":
                MAX_SINGLE_WARM_REGRESSION,
            "maximum_median_output_regression":
                MAX_MEDIAN_OUTPUT_REGRESSION,
            "maximum_single_output_regression":
                MAX_SINGLE_OUTPUT_REGRESSION,
        },
        "pairs": pairs,
        "target_summary": target_summary,
        "output_tps": output_summary,
        "quality": {
            "required_output_gate_passed": required_output_gate_passed,
            "first_generated_token_exact": first_generated_token_exact,
            "completion_structure_exact": completion_structure_exact,
            "short_full_output_exact": short_full_output_exact,
            "all_full_output_hashes_exact": full_output_hashes_exact,
            "long_full_output_mismatches": long_full_output_mismatches,
            "long_full_output_exact_required": False,
        },
        "qualified": qualified,
        "reasons": reasons,
        "decision": {
            "full_tp4_quality_gate_authorized": qualified,
            "official_style_replay_authorized": False,
            "production_promotion_authorized": False,
            "yaml_change_authorized": False,
            "main_merge_authorized": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--control",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = compare(
        [load_report(path) for path in args.control],
        [load_report(path) for path in args.candidate],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
