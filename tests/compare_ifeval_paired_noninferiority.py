#!/usr/bin/env python3
"""Apply the layered paired non-inferiority screen to two IFEval reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import compare_ifeval_reports as ifeval
import paired_noninferiority as paired


SCHEMA = "bi100-ifeval-paired-noninferiority-v1"
SCHEMA_V2 = "bi100-ifeval-paired-noninferiority-v2"
VERSION = 1
CONTRACT_SCHEMA = "bi100-layered-quality-gate-contract-v1"
CONTRACT_SCHEMA_V2 = "bi100-layered-quality-gate-contract-v2"
Json = dict[str, Any]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _metric_outcomes(report: Json, metric: str, label: str) -> list[bool]:
    selected = (report.get("manifest") or {}).get("selected_keys")
    cases = report.get("cases")
    if not isinstance(selected, list) or not isinstance(cases, list):
        raise ValueError(f"{label}: selected cases are missing")
    by_key = {
        case.get("key"): case
        for case in cases
        if isinstance(case, dict) and "key" in case
    }
    if len(by_key) != len(cases) or set(by_key) != set(selected):
        raise ValueError(f"{label}: selected case identities differ")

    outcomes = []
    for key in selected:
        case = by_key[key]
        instruction_ids = case.get("instruction_id_list")
        values = case.get(metric)
        if case.get("status") != "pass":
            raise ValueError(f"{label}: case status is incomplete")
        if (
            not isinstance(instruction_ids, list)
            or not instruction_ids
            or not isinstance(values, list)
            or len(values) != len(instruction_ids)
            or any(type(value) is not bool for value in values)
        ):
            raise ValueError(f"{label}: {metric} outcomes are invalid")
        outcomes.append(all(values))
    return outcomes


def _contract_values(contract: Json) -> Json:
    identity = (contract.get("schema"), contract.get("version"))
    section = (
        "paired_task_capability"
        if identity == (CONTRACT_SCHEMA_V2, 2)
        else "paired_capability"
    )
    if (
        identity not in {
            (CONTRACT_SCHEMA, 1),
            (CONTRACT_SCHEMA_V2, 2),
        }
        or not isinstance(contract.get(section), dict)
    ):
        raise ValueError("layered quality contract is invalid")
    values = contract[section]
    try:
        confidence = float(values["confidence"])
        screen_margin = float(values["small_stratum_noninferiority_margin"])
        promotion_margin = float(values["default_noninferiority_margin"])
        bootstrap_samples = int(values["bootstrap_samples"])
        seed = int(values["bootstrap_seed"])
        small_floor = int(values.get(
            "small_stratum_zero_regression_minimum_pairs",
            paired.minimum_zero_regression_samples(
                screen_margin, confidence),
        ))
        promotion_floor = int(values.get(
            "default_zero_regression_minimum_pairs",
            paired.minimum_zero_regression_samples(
                promotion_margin, confidence),
        ))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("paired capability contract is incomplete") from exc
    if (
        small_floor != paired.minimum_zero_regression_samples(
            screen_margin, confidence)
        or promotion_floor != paired.minimum_zero_regression_samples(
            promotion_margin, confidence)
    ):
        raise ValueError("paired capability power floor is inconsistent")
    return {
        "contract_version": contract["version"],
        "confidence": confidence,
        "small_margin": screen_margin,
        "small_floor": small_floor,
        "default_margin": promotion_margin,
        "default_floor": promotion_floor,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }


def compare(
    baseline: Json,
    candidate: Json,
    contract: Json,
    *,
    allowed_switches: set[str],
) -> Json:
    reasons = ifeval.pair_identity_reasons(
        baseline, candidate, allowed_switches)
    checks: dict[str, Json] = {}
    try:
        contract_values = _contract_values(contract)
        baseline_strict = _metric_outcomes(
            baseline, "strict", "baseline")
        candidate_strict = _metric_outcomes(
            candidate, "strict", "candidate")
        baseline_loose = _metric_outcomes(
            baseline, "loose", "baseline")
        candidate_loose = _metric_outcomes(
            candidate, "loose", "candidate")
    except ValueError as exc:
        reasons.append(str(exc))

    if reasons:
        status = "invalid"
        qualified = False
        contract_version = None
        confidence = None
        screen_margin = None
        promotion_margin = None
        bootstrap_samples = None
        seed = None
        sample_count = None
        promotion_power_floor = None
        screen_name = None
    else:
        sample_count = len(baseline_strict)
        contract_version = contract_values["contract_version"]
        confidence = contract_values["confidence"]
        bootstrap_samples = contract_values["bootstrap_samples"]
        seed = contract_values["seed"]
        promotion_margin = contract_values["default_margin"]
        promotion_power_floor = contract_values["default_floor"]
        use_default_margin = (
            contract_version == 2
            and sample_count >= promotion_power_floor
        )
        if use_default_margin:
            screen_name = "default-two-point"
            screen_margin = promotion_margin
        else:
            screen_name = "small-stratum-five-point"
            screen_margin = contract_values["small_margin"]
        checks = {
            "strict_prompt": paired.paired_noninferiority(
                baseline_strict,
                candidate_strict,
                margin=screen_margin,
                confidence=confidence,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            ),
            "loose_prompt": paired.paired_noninferiority(
                baseline_loose,
                candidate_loose,
                margin=screen_margin,
                confidence=confidence,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 1,
            ),
        }
        statuses = {value["status"] for value in checks.values()}
        if "fail" in statuses:
            status = "fail"
        elif "inconclusive" in statuses:
            status = "inconclusive"
        else:
            status = "pass"
        qualified = status == "pass"
        for name, value in checks.items():
            reasons.extend(
                f"{name}: {reason}" for reason in value["reasons"])

    two_point_authorized = (
        qualified
        and contract_version == 2
        and screen_name == "default-two-point"
    )
    five_point_authorized = (
        qualified and screen_name == "small-stratum-five-point"
    )
    screen = {
        "confidence": confidence,
        "noninferiority_margin": screen_margin,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "checks": checks,
    }
    authorization = {
        "five_point_screen_authorized": five_point_authorized,
        "two_point_promotion_authorized": False,
        "overall_promotion_authorized": False,
    }
    if contract_version == 2:
        screen["name"] = screen_name
        authorization["two_point_capability_surface_authorized"] = (
            two_point_authorized)
    result = {
        "schema": SCHEMA_V2 if contract_version == 2 else SCHEMA,
        "version": 2 if contract_version == 2 else VERSION,
        "status": status,
        "qualified": qualified,
        "allowed_switches": sorted(allowed_switches),
        "sample_count": sample_count,
        "screen": screen,
        "promotion_power": {
            "noninferiority_margin": promotion_margin,
            "minimum_zero_regression_samples": promotion_power_floor,
            "sufficient": (
                sample_count is not None
                and promotion_power_floor is not None
                and sample_count >= promotion_power_floor
            ),
        },
        "zero_stratum_diagnostic": {
            "qualified": not ifeval.no_regression_reasons(
                baseline, candidate) if not reasons or checks else None,
            "reason_count": len(ifeval.no_regression_reasons(
                baseline, candidate)) if not reasons or checks else None,
        },
        "reasons": reasons,
        "authorization": authorization,
        "privacy": {
            "contains_sample_outcomes": False,
            "contains_prompts": False,
            "contains_model_outputs": False,
            "contains_token_ids": False,
            "contains_credentials": False,
        },
    }
    if contract_version == 2:
        result["contract_version"] = contract_version
    return result


def _exit_code(status: str) -> int:
    return {"pass": 0, "fail": 1, "invalid": 2, "inconclusive": 3}[status]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--allowed-switch", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    report = compare(
        baseline,
        candidate,
        contract,
        allowed_switches=set(args.allowed_switch),
    )
    report["baseline_sha256"] = _sha256(args.baseline)
    report["candidate_sha256"] = _sha256(args.candidate)
    report["contract_sha256"] = _sha256(args.contract)
    _atomic_write(args.out, report)
    print(json.dumps({
        "out": str(args.out),
        "status": report["status"],
        "qualified": report["qualified"],
        "reasons": report["reasons"],
    }, sort_keys=True))
    return _exit_code(report["status"])


if __name__ == "__main__":
    raise SystemExit(main())
