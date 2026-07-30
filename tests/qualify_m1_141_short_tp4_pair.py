#!/usr/bin/env python3
"""Qualify a hash-bound control/candidate M1-141 short TP4 pair."""

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


CONTRACT_SCHEMA = "bi100-short-tp4-pair-contract-v1"
CONTRACT_SHA256 = (
    "b841c74ca71223e7e3317f0b21b91a6177e01699abc4092ed06371effa8759d2"
)
RUNNER_SCHEMA = "bi100-m1-141-short-tp4-screen-runner-v1"
MEASUREMENT_SCHEMA = "bi100-short-tp4-funnel-service-v1"
RESULT_SCHEMA = "bi100-m1-141-short-tp4-pair-qualification-v1"
TARGETS = [4096, 32768, 65536]
REPETITIONS = 3
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
MEASUREMENT_FIELDS = {
    "schema", "version", "run_id", "prompt_set_id", "selector",
    "targets", "max_tokens", "repetitions", "elapsed_s", "qualified",
    "reasons", "cases", "cold_ttft_median_s", "warm_ttft_median_s",
    "privacy", "authorization",
}
CASE_FIELDS = {
    "target_prompt_tokens", "repetition", "prompt_sha256", "cold", "warm",
}
RESPONSE_FIELDS = {
    "ok", "elapsed_s", "ttft_s", "last_output_s", "decode_window_s",
    "output_tps", "prompt_tokens", "cached_tokens", "completion_tokens",
    "finish_reason", "first_token_sha256", "output_sha256",
}
MEASUREMENT_PRIVACY = {
    "contains_prompts": False,
    "contains_model_outputs": False,
    "contains_token_ids": False,
    "contains_credentials": False,
}
MEASUREMENT_AUTHORIZATION = {
    "long_context_authorized": False,
    "main_or_yaml_change_authorized": False,
}
RUNNER_AUTHORIZATION = {
    "long_context_authorized": False,
    "full_capability_authorized": False,
    "main_or_yaml_change_authorized": False,
}
RUNNER_PRIVACY = {
    "prompts_recorded": False,
    "model_outputs_recorded": False,
    "token_ids_recorded": False,
    "credentials_recorded": False,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _finite_positive(value: Any) -> bool:
    return _finite(value) and float(value) > 0.0


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


def _contract_values(
    contract: Any,
    contract_sha256: str,
    invalid_reasons: list[str],
) -> dict[str, float]:
    value = _mapping(contract)
    scope = _mapping(value.get("scope"))
    matrix = _mapping(value.get("matrix"))
    identity = _mapping(value.get("identity"))
    hard = _mapping(value.get("hard_gates"))
    screen = _mapping(value.get("performance_screen"))
    authorization = _mapping(value.get("authorization"))
    if (
        contract_sha256 != CONTRACT_SHA256
        or value.get("schema") != CONTRACT_SCHEMA
        or value.get("version") != 1
        or value.get("frozen_date") != "2026-07-30"
        or scope != {
            "model": "Qwen3.6-35B-A3B",
            "device": "BI100/CoreX",
            "tensor_parallel_size": 4,
            "max_model_len": 262144,
        }
        or matrix != {
            "target_prompt_tokens": TARGETS,
            "repetitions_per_target": REPETITIONS,
            "max_tokens": 8,
            "temperature": 0,
            "seed": 20260730,
            "thinking": False,
            "stream": True,
        }
        or identity != {
            "same_source_revision": True,
            "same_runtime_identity": True,
            "same_instance": True,
            "same_prompt_set": True,
            "candidate_artifact_must_match_l2": True,
        }
        or hard != {
            "both_arms_qualified": True,
            "same_arm_cold_warm_exact": True,
            "cold_cached_tokens": 0,
            "warm_cached_token_slack": 32,
            "candidate_dispatch_required": True,
            "control_dispatch_forbidden": True,
            "finite_metrics": True,
            "clean_lifecycle": True,
        }
        or screen != {
            "maximum_overall_cold_ttft_regression_fraction": 0.05,
            "maximum_per_target_cold_ttft_regression_fraction": 0.1,
            "maximum_overall_warm_ttft_regression_fraction": 0.1,
            "minimum_candidate_improvement_required": False,
            "cross_arm_output_identity_role": "diagnostic_only",
        }
        or authorization != {
            "long_context_confirmation_on_pass": True,
            "full_capability_authorized": False,
            "main_or_yaml_change_authorized": False,
        }
        or set(value) != {
            "schema", "version", "frozen_date", "scope", "matrix",
            "identity", "hard_gates", "performance_screen",
            "authorization",
        }
    ):
        invalid_reasons.append("short TP4 pair contract differs")
    return {
        "overall_cold_regression": 0.05,
        "per_target_cold_regression": 0.1,
        "overall_warm_regression": 0.1,
    }


def _validate_response(
    value: Any,
    *,
    target: int,
    warm: bool,
    label: str,
    invalid_reasons: list[str],
) -> dict[str, Any] | None:
    response = _mapping(value)
    numeric_names = {
        "elapsed_s", "ttft_s", "last_output_s",
        "decode_window_s", "output_tps",
    }
    integer_names = {
        "prompt_tokens", "cached_tokens", "completion_tokens",
    }
    if (
        set(response) != RESPONSE_FIELDS
        or response.get("ok") is not True
        or any(
            not _finite(response.get(name))
            or float(response[name]) < 0.0
            for name in numeric_names
        )
        or not _finite_positive(response.get("elapsed_s"))
        or not _finite_positive(response.get("ttft_s"))
        or response.get("last_output_s", -1)
        < response.get("ttft_s", 0)
        or response.get("elapsed_s", -1)
        < response.get("last_output_s", 0)
        or any(
            not isinstance(response.get(name), int)
            or isinstance(response[name], bool)
            or response[name] < 0
            for name in integer_names
        )
        or response.get("prompt_tokens") != target
        or response.get("completion_tokens") != 8
        or (
            response.get("cached_tokens") < target - 32
            if warm else response.get("cached_tokens") != 0
        )
        or not isinstance(response.get("finish_reason"), str)
        or not response["finish_reason"]
        or not _hex(response.get("first_token_sha256"), 64)
        or not _hex(response.get("output_sha256"), 64)
    ):
        invalid_reasons.append(f"{label}: response contract differs")
        return None
    return response


def _validate_measurement(
    value: Any,
    *,
    selector: str,
    invalid_reasons: list[str],
) -> tuple[dict[tuple[int, int], dict[str, Any]], list[float], list[float]]:
    report = _mapping(value)
    cases = report.get("cases")
    if (
        set(report) != MEASUREMENT_FIELDS
        or report.get("schema") != MEASUREMENT_SCHEMA
        or report.get("version") != 1
        or report.get("selector") != selector
        or report.get("targets") != TARGETS
        or report.get("max_tokens") != 8
        or report.get("repetitions") != REPETITIONS
        or not isinstance(report.get("run_id"), str)
        or not report["run_id"]
        or not isinstance(report.get("prompt_set_id"), str)
        or not report["prompt_set_id"]
        or not _finite_positive(report.get("elapsed_s"))
        or report.get("qualified") is not True
        or report.get("reasons") != []
        or not isinstance(cases, list)
        or len(cases) != len(TARGETS) * REPETITIONS
        or report.get("privacy") != MEASUREMENT_PRIVACY
        or report.get("authorization") != MEASUREMENT_AUTHORIZATION
    ):
        invalid_reasons.append(f"{selector}: measurement structure differs")
        return {}, [], []

    observed: dict[tuple[int, int], dict[str, Any]] = {}
    cold_ttfts = []
    warm_ttfts = []
    for index, raw_case in enumerate(cases):
        label = f"{selector} case {index}"
        case = _mapping(raw_case)
        target = case.get("target_prompt_tokens")
        repetition = case.get("repetition")
        if (
            set(case) != CASE_FIELDS
            or target not in TARGETS
            or not isinstance(target, int)
            or isinstance(target, bool)
            or not isinstance(repetition, int)
            or isinstance(repetition, bool)
            or repetition not in range(REPETITIONS)
            or not _hex(case.get("prompt_sha256"), 64)
            or (target, repetition) in observed
        ):
            invalid_reasons.append(f"{label}: case identity differs")
            continue
        cold = _validate_response(
            case.get("cold"),
            target=target,
            warm=False,
            label=f"{label} cold",
            invalid_reasons=invalid_reasons,
        )
        warm = _validate_response(
            case.get("warm"),
            target=target,
            warm=True,
            label=f"{label} warm",
            invalid_reasons=invalid_reasons,
        )
        if cold is None or warm is None:
            continue
        for field in (
            "first_token_sha256", "output_sha256",
            "completion_tokens", "finish_reason",
        ):
            if cold[field] != warm[field]:
                invalid_reasons.append(
                    f"{label}: cold/warm {field} differs")
        observed[(target, repetition)] = case
        cold_ttfts.append(float(cold["ttft_s"]))
        warm_ttfts.append(float(warm["ttft_s"]))

    expected = {
        (target, repetition)
        for target in TARGETS
        for repetition in range(REPETITIONS)
    }
    if set(observed) != expected:
        invalid_reasons.append(
            f"{selector}: measurement matrix is incomplete")
    if (
        len(cold_ttfts) == len(expected)
        and (
            not _finite_positive(report.get("cold_ttft_median_s"))
            or not math.isclose(
                report["cold_ttft_median_s"],
                statistics.median(cold_ttfts),
                rel_tol=1.0e-9,
                abs_tol=1.0e-9,
            )
            or not _finite_positive(report.get("warm_ttft_median_s"))
            or not math.isclose(
                report["warm_ttft_median_s"],
                statistics.median(warm_ttfts),
                rel_tol=1.0e-9,
                abs_tol=1.0e-9,
            )
        )
    ):
        invalid_reasons.append(
            f"{selector}: measurement median is inconsistent")
    return observed, cold_ttfts, warm_ttfts


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
    limits = _contract_values(
        contract, contract_sha256, invalid_reasons)
    parsed_statuses = {}
    parsed_cases = {}
    cold_by_arm = {}
    warm_by_arm = {}
    for selector in ("control", "candidate"):
        status = _mapping(statuses.get(selector))
        artifacts = _mapping(status.get("artifact_sha256"))
        extension = _mapping(status.get("candidate_extension"))
        gates = status.get("gates")
        l2 = _mapping(status.get("l2_authorization"))
        timing = _mapping(status.get("timing"))
        expected_extension = (
            {
                "sha256": None,
                "size_bytes": None,
                "external_override_active": False,
            }
            if selector == "control"
            else {
                "sha256": l2.get("candidate_extension_sha256"),
                "size_bytes": l2.get(
                    "candidate_extension_size_bytes"),
                "external_override_active": True,
            }
        )
        if (
            status.get("schema") != RUNNER_SCHEMA
            or status.get("version") != 1
            or status.get("qualified") is not True
            or not isinstance(status.get("returncode"), int)
            or isinstance(status["returncode"], bool)
            or status.get("returncode") != 0
            or status.get("terminal_stage") != "complete"
            or status.get("error_type") is not None
            or not isinstance(status.get("run_id"), str)
            or not status["run_id"]
            or status.get("selector") != selector
            or status.get("gpu_count") != 4
            or status.get("tensor_parallel_size") != 4
            or status.get("service_startups") != 1
            or status.get("targets") != TARGETS
            or status.get("repetitions") != REPETITIONS
            or not _hex(status.get("source_revision"), 40)
            or not isinstance(status.get("instance"), str)
            or not status["instance"]
            or not isinstance(status.get("pair_id"), str)
            or not status["pair_id"]
            or not isinstance(status.get("runtime_identity"), str)
            or not status["runtime_identity"]
            or not _hex(status.get("kernel_source_sha256"), 64)
            or status.get("authorization") != RUNNER_AUTHORIZATION
            or status.get("privacy") != RUNNER_PRIVACY
            or set(timing) != {"wall_span_s", "summed_stage_s"}
            or not _finite_positive(timing.get("wall_span_s"))
            or not _finite_positive(timing.get("summed_stage_s"))
            or extension != expected_extension
            or not isinstance(status.get("dispatch_count"), int)
            or isinstance(status["dispatch_count"], bool)
            or (
                selector == "control"
                and status["dispatch_count"] != 0
            )
            or (
                selector == "candidate"
                and status["dispatch_count"] < 2
            )
            or not isinstance(gates, dict)
            or set(gates) != REQUIRED_GATES
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value != 0
                for value in gates.values()
            )
            or set(artifacts) != REQUIRED_ARTIFACTS
            or not all(_hex(value, 64) for value in artifacts.values())
            or artifacts.get("measurement.json")
            != measurement_sha256s.get(selector)
            or not _hex(l2.get("qualification_sha256"), 64)
            or not _hex(l2.get("runner_status_sha256"), 64)
            or not _hex(l2.get("candidate_extension_sha256"), 64)
            or not isinstance(
                l2.get("candidate_extension_size_bytes"), int)
            or isinstance(
                l2.get("candidate_extension_size_bytes"), bool)
            or l2.get("candidate_extension_size_bytes", 0) <= 0
        ):
            invalid_reasons.append(
                f"{selector}: runner identity or lifecycle differs")
        parsed_statuses[selector] = status
        cases, cold, warm = _validate_measurement(
            measurements.get(selector),
            selector=selector,
            invalid_reasons=invalid_reasons,
        )
        parsed_cases[selector] = cases
        cold_by_arm[selector] = cold
        warm_by_arm[selector] = warm
        if (
            _mapping(measurements.get(selector)).get("run_id")
            != status.get("run_id")
        ):
            invalid_reasons.append(
                f"{selector}: measurement run identity differs")

    control_status = parsed_statuses["control"]
    candidate_status = parsed_statuses["candidate"]
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
        or _mapping(measurements.get("control")).get("prompt_set_id")
        != _mapping(measurements.get("candidate")).get("prompt_set_id")
        or _mapping(measurements.get("control")).get("prompt_set_id")
        != control_status.get("pair_id")
    ):
        invalid_reasons.append("control/candidate pair identity differs")

    prompt_matches = 0
    first_token_matches = 0
    output_matches = 0
    expected_keys = {
        (target, repetition)
        for target in TARGETS
        for repetition in range(REPETITIONS)
    }
    for key in sorted(expected_keys):
        control_case = parsed_cases["control"].get(key)
        candidate_case = parsed_cases["candidate"].get(key)
        if control_case is None or candidate_case is None:
            continue
        if (
            control_case["prompt_sha256"]
            != candidate_case["prompt_sha256"]
        ):
            invalid_reasons.append(
                f"{key}: cross-arm prompt identity differs")
            continue
        prompt_matches += 1
        if (
            control_case["cold"]["first_token_sha256"]
            == candidate_case["cold"]["first_token_sha256"]
        ):
            first_token_matches += 1
        if (
            control_case["cold"]["output_sha256"]
            == candidate_case["cold"]["output_sha256"]
        ):
            output_matches += 1

    per_target = {}
    if all(
        len(parsed_cases[selector]) == len(expected_keys)
        for selector in ("control", "candidate")
    ):
        for target in TARGETS:
            control_values = [
                parsed_cases["control"][(target, repetition)][
                    "cold"]["ttft_s"]
                for repetition in range(REPETITIONS)
            ]
            candidate_values = [
                parsed_cases["candidate"][(target, repetition)][
                    "cold"]["ttft_s"]
                for repetition in range(REPETITIONS)
            ]
            control_median = statistics.median(control_values)
            candidate_median = statistics.median(candidate_values)
            regression = candidate_median / control_median - 1.0
            per_target[str(target)] = {
                "control_cold_ttft_median_s": control_median,
                "candidate_cold_ttft_median_s": candidate_median,
                "candidate_speedup": control_median / candidate_median,
                "candidate_regression_fraction": regression,
            }
            if regression > limits["per_target_cold_regression"]:
                performance_reasons.append(
                    f"{target}: cold TTFT regression exceeds screen")

    overall = {}
    if all(
        len(cold_by_arm[selector]) == len(expected_keys)
        and len(warm_by_arm[selector]) == len(expected_keys)
        for selector in ("control", "candidate")
    ):
        control_cold = statistics.median(cold_by_arm["control"])
        candidate_cold = statistics.median(cold_by_arm["candidate"])
        control_warm = statistics.median(warm_by_arm["control"])
        candidate_warm = statistics.median(warm_by_arm["candidate"])
        cold_regression = candidate_cold / control_cold - 1.0
        warm_regression = candidate_warm / control_warm - 1.0
        overall = {
            "control_cold_ttft_median_s": control_cold,
            "candidate_cold_ttft_median_s": candidate_cold,
            "candidate_cold_speedup": control_cold / candidate_cold,
            "candidate_cold_regression_fraction": cold_regression,
            "control_warm_ttft_median_s": control_warm,
            "candidate_warm_ttft_median_s": candidate_warm,
            "candidate_warm_speedup": control_warm / candidate_warm,
            "candidate_warm_regression_fraction": warm_regression,
        }
        if cold_regression > limits["overall_cold_regression"]:
            performance_reasons.append(
                "overall cold TTFT regression exceeds screen")
        if warm_regression > limits["overall_warm_regression"]:
            performance_reasons.append(
                "overall warm TTFT regression exceeds screen")

    qualified = not invalid_reasons and not performance_reasons
    return {
        "schema": RESULT_SCHEMA,
        "version": 1,
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
        "case_count_per_arm": len(expected_keys),
        "per_target": per_target,
        "overall": overall,
        "cross_arm_output_diagnostics": {
            "prompt_match_count": prompt_matches,
            "first_token_match_count": first_token_matches,
            "output_match_count": output_matches,
            "denominator": len(expected_keys),
            "role": "diagnostic_only",
        },
        "authorization": {
            "long_context_authorized": qualified,
            "full_capability_authorized": False,
            "main_or_yaml_change_authorized": False,
        },
    }


def _private_json(path: Path) -> tuple[dict[str, Any], str]:
    path = path.resolve(strict=True)
    if (
        not path.is_relative_to(Path("/tmp"))
        or path.stat().st_mode & 0o077
    ):
        raise ValueError("short TP4 evidence must be private under /tmp")
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
        result, ensure_ascii=True, indent=2, sort_keys=True,
        allow_nan=False))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
