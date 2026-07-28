#!/usr/bin/env python3
"""Privacy-safe three-pair admission64/fine32 policy-v2 comparator."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

SCHEMA = "bi100-m1-104-admission64-paired-ab-v1"
MEASUREMENT_SCHEMA = "bi100-m1-104-measurement-v1"
PAIR_COUNT = 3
REQUEST_COUNT = 18
MIN_HIT = 50.0
MIN_PAIRS_POSITIVE = 2
MIN_HIT_BENEFIT = 2.0
MIN_WEIGHTED_BENEFIT = 3.0
MAX_OUTPUT_MEDIAN_REGRESSION = 2.0
MAX_OUTPUT_SINGLE_REGRESSION = 5.0
MAX_TTFT_MEDIAN_REGRESSION = 2.0
MAX_TTFT_SINGLE_REGRESSION = 5.0


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def finite(value: Any, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{field} must be finite and >= {minimum}")
    return result


def digest(value: Any, field: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def canonical_hash(value: dict[str, Any]) -> str:
    payload = {key: value[key] for key in sorted(value) if key != "input_sha256"}
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def request_identity(request: Any, field: str) -> tuple[Any, ...]:
    if not isinstance(request, dict):
        raise ValueError(f"{field} must be an object")
    request_id = request.get("request_id")
    target = request.get("target_prompt_tokens")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError(f"{field}.request_id is invalid")
    if not isinstance(target, int) or isinstance(target, bool) or target < 1:
        raise ValueError(f"{field}.target_prompt_tokens is invalid")
    salt = digest(request.get("salt_sha256"), f"{field}.salt_sha256")
    output = digest(request.get("output_sha256"), f"{field}.output_sha256")
    first = digest(request.get("first_token_sha256"), f"{field}.first_token_sha256")
    finish = request.get("finish_reason")
    completion = request.get("completion_tokens")
    if not isinstance(finish, str) or not finish:
        raise ValueError(f"{field}.finish_reason is invalid")
    if not isinstance(completion, int) or isinstance(completion, bool) or completion < 0:
        raise ValueError(f"{field}.completion_tokens is invalid")
    return request_id, target, salt, output, first, finish, completion


def validate(report: dict[str, Any], mode: str, field: str) -> list[dict[str, Any]]:
    if report.get("schema") != MEASUREMENT_SCHEMA or report.get("mode") != mode:
        raise ValueError(f"{field} schema or mode mismatch")
    if report.get("request_count") != REQUEST_COUNT:
        raise ValueError(f"{field} must contain exactly 18 requests")
    if report.get("qualified_measurement") is not True or report.get("reasons") != []:
        raise ValueError(f"{field} measurement is not qualified")
    aggregate = report.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError(f"{field}.aggregate must be an object")
    finite(aggregate.get("effective_hit_rate"), f"{field}.aggregate.effective_hit_rate", minimum=0)
    finite(aggregate.get("success_rate"), f"{field}.aggregate.success_rate", minimum=0)
    finite(aggregate.get("cold_cached_tokens"), f"{field}.aggregate.cold_cached_tokens", minimum=0)
    finite(aggregate.get("output_tps_p10"), f"{field}.aggregate.output_tps_p10", minimum=0)
    finite(aggregate.get("ttft_p90_s"), f"{field}.aggregate.ttft_p90_s", minimum=0)
    finite(aggregate.get("weighted"), f"{field}.aggregate.weighted", minimum=0)
    requests = report.get("requests")
    if not isinstance(requests, list) or len(requests) != REQUEST_COUNT:
        raise ValueError(f"{field}.requests must contain 18 objects")
    identities = []
    for index, request in enumerate(requests):
        identity = request_identity(request, f"{field}.requests[{index}]")
        for phase in ("cold", "warm"):
            phase_data = request.get(phase)
            if not isinstance(phase_data, dict):
                raise ValueError(f"{field}.requests[{index}].{phase} must be an object")
            phase_identity = request_identity(phase_data, f"{field}.requests[{index}].{phase}")
            if phase_identity[:3] != identity[:3]:
                raise ValueError(f"{field}.requests[{index}] {phase} identity mismatch")
        identities.append(identity[:3])
    if len({item[0] for item in identities}) != REQUEST_COUNT:
        raise ValueError(f"{field} request ids must be unique")
    target_order = report.get("target_order")
    if target_order != [item[0] for item in identities]:
        raise ValueError(f"{field}.target_order does not match request order")
    return requests


def compare(controls: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    pairs: list[dict[str, Any]] = []
    manifest_signatures: set[tuple[Any, ...]] = set()
    if len(controls) != PAIR_COUNT or len(candidates) != PAIR_COUNT:
        reasons.append("exactly three control/candidate pairs are required")
    for index, (control, candidate) in enumerate(zip(controls, candidates), 1):
        field = f"pair[{index}]"
        try:
            control_requests = validate(control, "control", f"{field}.control")
            candidate_requests = validate(candidate, "candidate", f"{field}.candidate")
        except ValueError as error:
            reasons.append(str(error))
            continue
        control_hash = canonical_hash(control)
        candidate_hash = canonical_hash(candidate)
        if control.get("request_manifest_sha256") != candidate.get("request_manifest_sha256"):
            reasons.append(f"{field} request manifest differs")
        if control.get("target_order") != candidate.get("target_order"):
            reasons.append(f"{field} target order differs")
        manifest_signatures.add((control.get("request_manifest_sha256"), tuple(control.get("target_order", []))))
        if [request_identity(r, "control")[:3] for r in control_requests] != [request_identity(r, "candidate")[:3] for r in candidate_requests]:
            reasons.append(f"{field} request or salt order differs")
        ca, cb = control["aggregate"], candidate["aggregate"]
        if cb["effective_hit_rate"] < MIN_HIT:
            reasons.append(f"{field} candidate effective hit is below 50%")
        if ca["success_rate"] != 100.0 or cb["success_rate"] != 100.0:
            reasons.append(f"{field} success rate is below 100%")
        if ca["cold_cached_tokens"] != 0 or cb["cold_cached_tokens"] != 0:
            reasons.append(f"{field} cold cached tokens must be zero")
        output_reg = (ca["output_tps_p10"] - cb["output_tps_p10"]) / ca["output_tps_p10"] * 100
        ttft_reg = (cb["ttft_p90_s"] - ca["ttft_p90_s"]) / ca["ttft_p90_s"] * 100
        if output_reg > MAX_OUTPUT_SINGLE_REGRESSION:
            reasons.append(f"{field} Output TPS regression exceeds 5%")
        if ttft_reg > MAX_TTFT_SINGLE_REGRESSION:
            reasons.append(f"{field} TTFT P90 regression exceeds 5%")
        exact = True
        for c_request, n_request in zip(control_requests, candidate_requests):
            for phase in ("cold", "warm"):
                c_id = request_identity(c_request[phase], "control")
                n_id = request_identity(n_request[phase], "candidate")
                if c_id[3:] != n_id[3:]:
                    exact = False
                    reasons.append(f"{field} {c_id[0]} {phase} output/finish/completion differs")
        hit_benefit = cb["effective_hit_rate"] - ca["effective_hit_rate"]
        weighted_benefit = (cb["weighted"] / ca["weighted"] - 1.0) * 100
        pairs.append({"pair": index, "control_input_sha256": control_hash, "candidate_input_sha256": candidate_hash, "hit_benefit_pp": hit_benefit, "weighted_benefit_pct": weighted_benefit, "output_regression_pct": output_reg, "ttft_regression_pct": ttft_reg, "exact_request_outputs": exact})
    valid = len(pairs) == PAIR_COUNT
    if valid:
        if len(manifest_signatures) != 1:
            reasons.append("all three pairs must use the same request manifest and target order")
        hit = [p["hit_benefit_pp"] for p in pairs]
        weighted = [p["weighted_benefit_pct"] for p in pairs]
        positive = sum(value > 0 for value in hit)
        median_hit, median_weighted = statistics.median(hit), statistics.median(weighted)
        if positive < MIN_PAIRS_POSITIVE:
            reasons.append("fewer than two pairs have positive benefit")
        if median_hit < MIN_HIT_BENEFIT and not (median_weighted >= MIN_WEIGHTED_BENEFIT and median_hit >= 0):
            reasons.append("median policy-v2 benefit is below hit/weighted threshold")
        if any(p["output_regression_pct"] > MAX_OUTPUT_SINGLE_REGRESSION or p["ttft_regression_pct"] > MAX_TTFT_SINGLE_REGRESSION or not p["exact_request_outputs"] for p in pairs):
            pass
        if statistics.median([p["output_regression_pct"] for p in pairs]) > MAX_OUTPUT_MEDIAN_REGRESSION:
            reasons.append("median Output TPS regression exceeds 2%")
        if statistics.median([p["ttft_regression_pct"] for p in pairs]) > MAX_TTFT_MEDIAN_REGRESSION:
            reasons.append("median TTFT P90 regression exceeds 2%")
    summary = {"positive_pairs": sum(p["hit_benefit_pp"] > 0 for p in pairs), "median_hit_benefit_pp": statistics.median([p["hit_benefit_pp"] for p in pairs]) if pairs else None, "median_weighted_benefit_pct": statistics.median([p["weighted_benefit_pct"] for p in pairs]) if pairs else None}
    qualified = valid and not reasons
    return {"schema": SCHEMA, "version": 1, "pair_count": PAIR_COUNT, "thresholds": {"minimum_effective_hit_rate": MIN_HIT, "minimum_positive_pairs": MIN_PAIRS_POSITIVE, "minimum_median_hit_benefit_pp": MIN_HIT_BENEFIT, "minimum_median_weighted_benefit_pct": MIN_WEIGHTED_BENEFIT, "maximum_output_median_regression_pct": MAX_OUTPUT_MEDIAN_REGRESSION, "maximum_output_single_regression_pct": MAX_OUTPUT_SINGLE_REGRESSION, "maximum_ttft_median_regression_pct": MAX_TTFT_MEDIAN_REGRESSION, "maximum_ttft_single_regression_pct": MAX_TTFT_SINGLE_REGRESSION}, "pairs": pairs, "summary": summary, "qualified": qualified, "reasons": reasons, "decision": {"m1_85_full_quality_authorized": qualified, "official_or_yaml_authorized": False, "default_or_main_authorized": False}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", action="append", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = compare([load(p) for p in args.control], [load(p) for p in args.candidate])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
