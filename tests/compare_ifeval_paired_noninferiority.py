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
VERSION = 1
CONTRACT_SCHEMA = "bi100-layered-quality-gate-contract-v1"
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


def _contract_values(contract: Json) -> tuple[float, float, int, int, float]:
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("version") != 1
        or not isinstance(contract.get("paired_capability"), dict)
    ):
        raise ValueError("layered quality contract is invalid")
    values = contract["paired_capability"]
    try:
        confidence = float(values["confidence"])
        screen_margin = float(values["small_stratum_noninferiority_margin"])
        promotion_margin = float(values["default_noninferiority_margin"])
        bootstrap_samples = int(values["bootstrap_samples"])
        seed = int(values["bootstrap_seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("paired capability contract is incomplete") from exc
    return confidence, screen_margin, bootstrap_samples, seed, promotion_margin


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
        (confidence, screen_margin, bootstrap_samples, seed,
         promotion_margin) = _contract_values(contract)
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
        confidence = None
        screen_margin = None
        promotion_margin = None
        bootstrap_samples = None
        seed = None
        sample_count = None
        promotion_power_floor = None
    else:
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
        sample_count = len(baseline_strict)
        promotion_power_floor = paired.minimum_zero_regression_samples(
            promotion_margin, confidence)
        for name, value in checks.items():
            reasons.extend(
                f"{name}: {reason}" for reason in value["reasons"])

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "status": status,
        "qualified": qualified,
        "allowed_switches": sorted(allowed_switches),
        "sample_count": sample_count,
        "screen": {
            "confidence": confidence,
            "noninferiority_margin": screen_margin,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": seed,
            "checks": checks,
        },
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
        "authorization": {
            "five_point_screen_authorized": qualified,
            "two_point_promotion_authorized": False,
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
