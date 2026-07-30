#!/usr/bin/env python3
"""Qualify the M1-152 P90-oriented control/candidate TP4 pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any

from bench_fused_prefill_service import _percentile
import short_tp4_p90_funnel_service as service


CONTRACT_SCHEMA = "bi100-short-tp4-p90-pair-contract-v2"
CONTRACT_SHA256 = (
    "cc8dbf3af68a30c9192d5633767aca6a1264115915263c23917c49cb3ed60cc7"
)
RUNNER_SCHEMA = "bi100-m1-152-short-tp4-p90-screen-runner-v2"
RESULT_SCHEMA = "bi100-m1-152-short-tp4-p90-pair-qualification-v2"
REQUIRED_GATES = {
    "postflight_before", "preflight_before", "runtime_identity",
    "service_startup", "request_matrix", "dispatch",
    "health_after_requests", "scoped_cleanup", "postflight_after",
    "preflight_after", "preflight_comparison", "fatal_scan",
    "source_unchanged",
}
REQUIRED_ARTIFACTS = {
    "runtime_identity.json", "measurement.json", "dispatch_count.txt",
    "fatal_scan.json", "postflight_after.json",
    "preflight_comparison.json", "timeline_report.json",
}
RUNNER_AUTHORIZATION = {
    "long_context_confirmation_authorized": False,
    "full_capability_authorized": False,
    "main_or_yaml_change_authorized": False,
}
RUNNER_PRIVACY = {
    "prompts_recorded": False,
    "model_outputs_recorded": False,
    "token_ids_recorded": False,
    "credentials_recorded": False,
}
MEASUREMENT_AUTHORIZATION = {
    "long_context_confirmation_authorized": False,
    "full_capability_authorized": False,
    "main_or_yaml_change_authorized": False,
}
MEASUREMENT_PRIVACY = {
    "contains_prompts": False,
    "contains_model_outputs": False,
    "contains_token_ids": False,
    "contains_credentials": False,
}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            json.dump(
                value,
                stream,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _validate_contract(
    contract: Any,
    contract_sha256: str,
    invalid_reasons: list[str],
) -> dict[str, float | int]:
    value = _mapping(contract)
    expected = {
        "schema": CONTRACT_SCHEMA,
        "version": 2,
        "frozen_date": "2026-07-30",
        "scope": {
            "model": "Qwen3.6-35B-A3B",
            "device": "BI100/CoreX",
            "tensor_parallel_size": 4,
            "max_model_len": 262144,
            "role": "development_continuation_screen",
        },
        "matrix": {
            "cold_target_prompt_tokens": list(service.TARGETS),
            "partial_target_prompt_tokens": list(service.PARTIAL_TARGETS),
            "partial_residual_tokens": service.PARTIAL_RESIDUAL_TOKENS,
            "repetitions_per_shape": service.REPETITIONS,
            "max_tokens": service.MAX_TOKENS,
            "temperature": 0,
            "seed": service.SEED,
            "thinking": False,
            "stream": True,
        },
        "identity": {
            "same_source_revision": True,
            "same_runtime_identity": True,
            "same_instance": True,
            "same_prompt_set": True,
            "candidate_artifact_must_match_l2": True,
            "candidate_artifact_must_match_p90_operator_screen": True,
        },
        "hard_gates": {
            "both_arms_qualified": True,
            "same_arm_cold_warm_exact": True,
            "same_arm_partial_warm_exact": True,
            "cold_cached_tokens": 0,
            "warm_cached_token_slack": 32,
            "partial_cached_token_slack": 32,
            "candidate_dispatch_required": True,
            "control_dispatch_forbidden": True,
            "finite_metrics": True,
            "clean_lifecycle": True,
        },
        "performance_screen": {
            "minimum_cold_shapes_improved": 4,
            "minimum_partial_shapes_improved": 3,
            "minimum_cold_median_speedup": 1.08,
            "minimum_partial_median_speedup": 1.08,
            "minimum_uncached_ttft_p90_speedup": 1.08,
            "maximum_individual_uncached_ttft_regression_fraction": 0.1,
            "maximum_warm_median_regression_seconds": 0.25,
            "maximum_warm_individual_regression_seconds": 0.5,
            "cross_arm_output_identity_role": "diagnostic_only",
        },
        "authorization": {
            "long_context_confirmation_on_pass": True,
            "full_capability_authorized": False,
            "main_or_yaml_change_authorized": False,
            "official_score_claim_authorized": False,
        },
    }
    if contract_sha256 != CONTRACT_SHA256 or value != expected:
        invalid_reasons.append("P90 pair contract differs")
    return {
        "minimum_cold_shapes_improved": 4,
        "minimum_partial_shapes_improved": 3,
        "minimum_cold_median_speedup": 1.08,
        "minimum_partial_median_speedup": 1.08,
        "minimum_uncached_ttft_p90_speedup": 1.08,
        "maximum_individual_regression": 0.1,
        "maximum_warm_median_regression_s": 0.25,
        "maximum_warm_individual_regression_s": 0.5,
    }


def _validate_status(
    status: Any,
    *,
    selector: str,
    measurement_sha256: str,
    invalid_reasons: list[str],
) -> dict[str, Any]:
    value = _mapping(status)
    gates = value.get("gates")
    artifacts = _mapping(value.get("artifact_sha256"))
    extension = _mapping(value.get("candidate_extension"))
    l2 = _mapping(value.get("l2_authorization"))
    p90 = _mapping(value.get("p90_operator_authorization"))
    timing = _mapping(value.get("timing"))
    expected_extension = (
        {
            "sha256": None,
            "size_bytes": None,
            "external_override_active": False,
        }
        if selector == "control"
        else {
            "sha256": l2.get("candidate_extension_sha256"),
            "size_bytes": l2.get("candidate_extension_size_bytes"),
            "external_override_active": True,
        }
    )
    if (
        value.get("schema") != RUNNER_SCHEMA
        or value.get("version") != 2
        or value.get("qualified") is not True
        or value.get("returncode") != 0
        or value.get("terminal_stage") != "complete"
        or value.get("error_type") is not None
        or value.get("selector") != selector
        or value.get("gpu_count") != 4
        or value.get("tensor_parallel_size") != 4
        or value.get("service_startups") != 1
        or value.get("targets") != list(service.TARGETS)
        or value.get("partial_targets") != list(service.PARTIAL_TARGETS)
        or value.get("partial_residual_tokens")
        != service.PARTIAL_RESIDUAL_TOKENS
        or value.get("repetitions") != service.REPETITIONS
        or not _hex(value.get("source_revision"), 40)
        or not isinstance(value.get("instance"), str)
        or not value["instance"]
        or not isinstance(value.get("pair_id"), str)
        or not value["pair_id"]
        or not isinstance(value.get("runtime_identity"), str)
        or not value["runtime_identity"]
        or not _hex(value.get("kernel_source_sha256"), 64)
        or value.get("authorization") != RUNNER_AUTHORIZATION
        or value.get("privacy") != RUNNER_PRIVACY
        or set(timing) != {"wall_span_s", "summed_stage_s"}
        or not all(_finite_positive(item) for item in timing.values())
        or not isinstance(gates, dict)
        or set(gates) != REQUIRED_GATES
        or any(item != 0 for item in gates.values())
        or set(artifacts) != REQUIRED_ARTIFACTS
        or not all(_hex(item, 64) for item in artifacts.values())
        or artifacts.get("measurement.json") != measurement_sha256
        or extension != expected_extension
        or not _hex(l2.get("candidate_extension_sha256"), 64)
        or not isinstance(l2.get("candidate_extension_size_bytes"), int)
        or l2.get("candidate_extension_size_bytes", 0) <= 0
        or p90.get("candidate_extension_sha256")
        != l2.get("candidate_extension_sha256")
        or p90.get("kernel_source_sha256")
        != value.get("kernel_source_sha256")
        or not _hex(p90.get("runner_status_sha256"), 64)
        or not _hex(p90.get("identity_sha256"), 64)
        or p90.get("case_count") != 8
        or not _finite_positive(p90.get("minimum_speedup"))
        or p90["minimum_speedup"] < 1.2
        or not isinstance(value.get("dispatch_count"), int)
        or (
            value.get("dispatch_count") != 0
            if selector == "control"
            else value.get("dispatch_count", 0) < 2
        )
    ):
        invalid_reasons.append(
            f"{selector}: runner identity or lifecycle differs")
    return value


def _validate_measurement(
    report: Any,
    *,
    selector: str,
    invalid_reasons: list[str],
) -> dict[str, Any]:
    value = _mapping(report)
    evaluation = service.evaluate(value)
    if (
        evaluation.get("qualified") is not True
        or value.get("evaluation") != evaluation
        or value.get("qualified") is not True
        or value.get("reasons") != []
        or value.get("selector") != selector
        or not isinstance(value.get("run_id"), str)
        or not value["run_id"]
        or not isinstance(value.get("prompt_set_id"), str)
        or not value["prompt_set_id"]
        or not _finite_positive(value.get("elapsed_s"))
        or value.get("privacy") != MEASUREMENT_PRIVACY
        or value.get("authorization") != MEASUREMENT_AUTHORIZATION
    ):
        invalid_reasons.append(
            f"{selector}: measurement structure or hard gate differs")
        return value

    cold = [case["cold"]["ttft_s"] for case in value["cold_cases"]]
    partial = [
        case["partial"]["ttft_s"] for case in value["partial_cases"]]
    warm = (
        [case["warm"]["ttft_s"] for case in value["cold_cases"]]
        + [case["warm"]["ttft_s"] for case in value["partial_cases"]]
    )
    expected_scalars = {
        "cold_ttft_median_s": statistics.median(cold),
        "partial_ttft_median_s": statistics.median(partial),
        "uncached_ttft_p90_s": _percentile(cold + partial, 90.0),
        "warm_ttft_median_s": statistics.median(warm),
    }
    if any(
        not _finite_positive(value.get(name))
        or not math.isclose(
            float(value[name]), expected,
            rel_tol=1.0e-9, abs_tol=1.0e-9,
        )
        for name, expected in expected_scalars.items()
    ):
        invalid_reasons.append(
            f"{selector}: measurement aggregate is inconsistent")
    return value


def _output_identity(response: dict[str, Any]) -> tuple[Any, ...]:
    return (
        response["first_token_sha256"],
        response["output_sha256"],
        response["completion_tokens"],
        response["finish_reason"],
    )


def qualify(
    statuses: dict[str, Any],
    measurements: dict[str, Any],
    measurement_sha256s: dict[str, str],
    contract: Any,
    *,
    contract_sha256: str,
) -> dict[str, Any]:
    invalid_reasons: list[str] = []
    performance_reasons: list[str] = []
    limits = _validate_contract(
        contract, contract_sha256, invalid_reasons)
    parsed_status = {}
    parsed_measurement = {}
    for selector in ("control", "candidate"):
        parsed_status[selector] = _validate_status(
            statuses.get(selector),
            selector=selector,
            measurement_sha256=measurement_sha256s.get(selector, ""),
            invalid_reasons=invalid_reasons,
        )
        parsed_measurement[selector] = _validate_measurement(
            measurements.get(selector),
            selector=selector,
            invalid_reasons=invalid_reasons,
        )
        if (
            parsed_status[selector].get("run_id")
            != parsed_measurement[selector].get("run_id")
        ):
            invalid_reasons.append(
                f"{selector}: run identity differs")

    control_status = parsed_status["control"]
    candidate_status = parsed_status["candidate"]
    control = parsed_measurement["control"]
    candidate = parsed_measurement["candidate"]
    if (
        control_status.get("source_revision")
        != candidate_status.get("source_revision")
        or control_status.get("instance")
        != candidate_status.get("instance")
        or control_status.get("pair_id")
        != candidate_status.get("pair_id")
        or control_status.get("runtime_identity")
        != candidate_status.get("runtime_identity")
        or control_status.get("kernel_source_sha256")
        != candidate_status.get("kernel_source_sha256")
        or control_status.get("l2_authorization")
        != candidate_status.get("l2_authorization")
        or control_status.get("p90_operator_authorization")
        != candidate_status.get("p90_operator_authorization")
        or control.get("prompt_set_id")
        != candidate.get("prompt_set_id")
        or control.get("prompt_set_id")
        != control_status.get("pair_id")
    ):
        invalid_reasons.append("control/candidate pair identity differs")

    cold_rows = {}
    partial_rows = {}
    cold_improved = 0
    partial_improved = 0
    output_diagnostics = {
        "cold_prompt_matches": 0,
        "cold_first_token_matches": 0,
        "cold_output_matches": 0,
        "partial_prompt_matches": 0,
        "partial_first_token_matches": 0,
        "partial_output_matches": 0,
        "role": "diagnostic_only",
    }
    if (
        len(control.get("cold_cases", [])) == len(service.TARGETS)
        and len(candidate.get("cold_cases", [])) == len(service.TARGETS)
    ):
        for target, left, right in zip(
            service.TARGETS,
            control["cold_cases"],
            candidate["cold_cases"],
        ):
            if left.get("prompt_sha256") == right.get("prompt_sha256"):
                output_diagnostics["cold_prompt_matches"] += 1
            else:
                invalid_reasons.append(
                    f"cold/{target}: cross-arm prompt differs")
            left_response = left["cold"]
            right_response = right["cold"]
            if (
                left_response["first_token_sha256"]
                == right_response["first_token_sha256"]
            ):
                output_diagnostics["cold_first_token_matches"] += 1
            if (
                left_response["output_sha256"]
                == right_response["output_sha256"]
            ):
                output_diagnostics["cold_output_matches"] += 1
            control_ttft = float(left_response["ttft_s"])
            candidate_ttft = float(right_response["ttft_s"])
            regression = candidate_ttft / control_ttft - 1.0
            if candidate_ttft < control_ttft:
                cold_improved += 1
            if regression > limits["maximum_individual_regression"]:
                performance_reasons.append(
                    f"cold/{target}: TTFT regression exceeds screen")
            cold_rows[str(target)] = {
                "control_ttft_s": control_ttft,
                "candidate_ttft_s": candidate_ttft,
                "candidate_speedup": control_ttft / candidate_ttft,
                "candidate_regression_fraction": regression,
            }

    if (
        len(control.get("partial_cases", []))
        == len(service.PARTIAL_TARGETS)
        and len(candidate.get("partial_cases", []))
        == len(service.PARTIAL_TARGETS)
    ):
        for target, left, right in zip(
            service.PARTIAL_TARGETS,
            control["partial_cases"],
            candidate["partial_cases"],
        ):
            if (
                left.get("primer_prompt_sha256")
                == right.get("primer_prompt_sha256")
                and left.get("partial_prompt_sha256")
                == right.get("partial_prompt_sha256")
            ):
                output_diagnostics["partial_prompt_matches"] += 1
            else:
                invalid_reasons.append(
                    f"partial/{target}: cross-arm prompt differs")
            left_response = left["partial"]
            right_response = right["partial"]
            if (
                left_response["first_token_sha256"]
                == right_response["first_token_sha256"]
            ):
                output_diagnostics["partial_first_token_matches"] += 1
            if (
                left_response["output_sha256"]
                == right_response["output_sha256"]
            ):
                output_diagnostics["partial_output_matches"] += 1
            control_ttft = float(left_response["ttft_s"])
            candidate_ttft = float(right_response["ttft_s"])
            regression = candidate_ttft / control_ttft - 1.0
            if candidate_ttft < control_ttft:
                partial_improved += 1
            if regression > limits["maximum_individual_regression"]:
                performance_reasons.append(
                    f"partial/{target}: TTFT regression exceeds screen")
            partial_rows[str(target)] = {
                "control_ttft_s": control_ttft,
                "candidate_ttft_s": candidate_ttft,
                "candidate_speedup": control_ttft / candidate_ttft,
                "candidate_regression_fraction": regression,
            }

    aggregate = {}
    if (
        len(cold_rows) == len(service.TARGETS)
        and len(partial_rows) == len(service.PARTIAL_TARGETS)
    ):
        control_cold = [
            case["cold"]["ttft_s"] for case in control["cold_cases"]]
        candidate_cold = [
            case["cold"]["ttft_s"] for case in candidate["cold_cases"]]
        control_partial = [
            case["partial"]["ttft_s"]
            for case in control["partial_cases"]
        ]
        candidate_partial = [
            case["partial"]["ttft_s"]
            for case in candidate["partial_cases"]
        ]
        control_warm = (
            [case["warm"]["ttft_s"] for case in control["cold_cases"]]
            + [case["warm"]["ttft_s"]
               for case in control["partial_cases"]]
        )
        candidate_warm = (
            [case["warm"]["ttft_s"] for case in candidate["cold_cases"]]
            + [case["warm"]["ttft_s"]
               for case in candidate["partial_cases"]]
        )
        cold_speedup = (
            statistics.median(control_cold)
            / statistics.median(candidate_cold)
        )
        partial_speedup = (
            statistics.median(control_partial)
            / statistics.median(candidate_partial)
        )
        control_p90 = _percentile(
            control_cold + control_partial, 90.0)
        candidate_p90 = _percentile(
            candidate_cold + candidate_partial, 90.0)
        p90_speedup = control_p90 / candidate_p90
        warm_deltas = [
            float(right) - float(left)
            for left, right in zip(control_warm, candidate_warm)
        ]
        aggregate = {
            "cold_shapes_improved": cold_improved,
            "partial_shapes_improved": partial_improved,
            "cold_median_speedup": cold_speedup,
            "partial_median_speedup": partial_speedup,
            "control_uncached_ttft_p90_s": control_p90,
            "candidate_uncached_ttft_p90_s": candidate_p90,
            "uncached_ttft_p90_speedup": p90_speedup,
            "warm_median_regression_s": statistics.median(warm_deltas),
            "warm_max_regression_s": max(warm_deltas),
        }
        for observed, minimum, label in (
            (
                cold_improved,
                limits["minimum_cold_shapes_improved"],
                "cold shape improvement count",
            ),
            (
                partial_improved,
                limits["minimum_partial_shapes_improved"],
                "partial shape improvement count",
            ),
            (
                cold_speedup,
                limits["minimum_cold_median_speedup"],
                "cold median speedup",
            ),
            (
                partial_speedup,
                limits["minimum_partial_median_speedup"],
                "partial median speedup",
            ),
            (
                p90_speedup,
                limits["minimum_uncached_ttft_p90_speedup"],
                "uncached TTFT P90 speedup",
            ),
        ):
            if observed < minimum:
                performance_reasons.append(f"{label} is below screen")
        if (
            aggregate["warm_median_regression_s"]
            > limits["maximum_warm_median_regression_s"]
        ):
            performance_reasons.append(
                "warm median TTFT regression exceeds screen")
        if (
            aggregate["warm_max_regression_s"]
            > limits["maximum_warm_individual_regression_s"]
        ):
            performance_reasons.append(
                "warm individual TTFT regression exceeds screen")

    qualified = not invalid_reasons and not performance_reasons
    return {
        "schema": RESULT_SCHEMA,
        "version": 2,
        "qualified": qualified,
        "invalid_reasons": invalid_reasons,
        "performance_reasons": performance_reasons,
        "contract_sha256": contract_sha256,
        "source_revision": control_status.get("source_revision"),
        "runtime_identity": control_status.get("runtime_identity"),
        "instance": control_status.get("instance"),
        "pair_id": control_status.get("pair_id"),
        "candidate_extension_sha256": _mapping(
            candidate_status.get("candidate_extension")).get("sha256"),
        "cold": cold_rows,
        "partial": partial_rows,
        "aggregate": aggregate,
        "cross_arm_output_diagnostics": output_diagnostics,
        "authorization": {
            "long_context_confirmation_authorized": qualified,
            "full_capability_authorized": False,
            "main_or_yaml_change_authorized": False,
            "official_score_claim_authorized": False,
        },
    }


def _private_json(path: Path) -> tuple[dict[str, Any], str]:
    path = path.resolve(strict=True)
    if (
        not path.is_relative_to(Path("/tmp"))
        or path.stat().st_mode & 0o077
    ):
        raise ValueError("P90 TP4 evidence must be private under /tmp")
    return (
        _mapping(json.loads(path.read_text(encoding="ascii"))),
        _sha256(path),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-status", type=Path, required=True)
    parser.add_argument("--control-measurement", type=Path, required=True)
    parser.add_argument("--candidate-status", type=Path, required=True)
    parser.add_argument("--candidate-measurement", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    control_status, _ = _private_json(args.control_status)
    control_measurement, control_measurement_sha = _private_json(
        args.control_measurement)
    candidate_status, _ = _private_json(args.candidate_status)
    candidate_measurement, candidate_measurement_sha = _private_json(
        args.candidate_measurement)
    contract = _mapping(json.loads(
        args.contract.read_text(encoding="ascii")))
    contract_sha = _sha256(args.contract)
    result = qualify(
        {
            "control": control_status,
            "candidate": candidate_status,
        },
        {
            "control": control_measurement,
            "candidate": candidate_measurement,
        },
        {
            "control": control_measurement_sha,
            "candidate": candidate_measurement_sha,
        },
        contract,
        contract_sha256=contract_sha,
    )
    _atomic_json(args.out, result)
    print(json.dumps(
        result,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
