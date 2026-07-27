#!/usr/bin/env python3
"""Compare baseline/default/image2 reduced-model HTTP compatibility arms."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any


Json = dict[str, Any]
SCHEMA = "qwen36-diagnostic-compat-http-comparison-v2"
VERSION = 2
GATE_SCHEMA = "qwen36-diagnostic-compat-http-gate-v2"
ATTRIBUTION_SCHEMA = "bi100-api-4xx-attribution-v3"


def _cases(report: Json) -> dict[str, Json]:
    rows = report.get("cases")
    if not isinstance(rows, list):
        return {}
    return {
        row["name"]: row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("name"), str)
    }


def _evidence(cases: dict[str, Json], name: str) -> Json:
    row = cases.get(name)
    if not isinstance(row, dict) or row.get("ok") is not True:
        return {}
    evidence = row.get("evidence")
    return evidence if isinstance(evidence, dict) else {}


def _message_sha(cases: dict[str, Json], name: str) -> str | None:
    value = _evidence(cases, name).get("message_sha256")
    return value if isinstance(value, str) else None


def _validate_attribution(
    report: Json,
    expected_image_count: int,
    label: str,
    reasons: list[str],
) -> None:
    if (report.get("schema") != ATTRIBUTION_SCHEMA
            or report.get("qualified") is not True):
        reasons.append(f"{label} 4xx attribution is not qualified")
        return
    if report.get("chat_4xx_access_count") != 1:
        reasons.append(f"{label} must contain exactly one chat 4xx")
    if report.get("by_reason") != {"image_count_limit": 1}:
        reasons.append(f"{label} did not isolate image_count_limit")
    shapes = report.get("request_shapes")
    if not isinstance(shapes, list) or len(shapes) != 1:
        reasons.append(f"{label} must contain exactly one request shape")
        return
    shape = shapes[0]
    expected = {
        "images": expected_image_count,
        "image_data": expected_image_count,
        "image_remote": 0,
        "image_other": 0,
        "system_part_msgs": 0,
        "system_text_parts": 0,
        "system_other_parts": 0,
    }
    for field, value in expected.items():
        if shape.get(field) != value:
            reasons.append(
                f"{label} shape {field}={shape.get(field)!r}, "
                f"expected {value}")


def compare(
    baseline: Json,
    candidate_default: Json,
    candidate_image2: Json,
    candidate_default_4xx: Json,
    candidate_image2_4xx: Json,
) -> Json:
    reasons: list[str] = []
    reports = {
        "baseline_default": baseline,
        "candidate_default": candidate_default,
        "candidate_image2": candidate_image2,
    }
    case_maps = {name: _cases(report) for name, report in reports.items()}

    for name, report in reports.items():
        if report.get("schema") != GATE_SCHEMA:
            reasons.append(f"{name} schema mismatch")
        if report.get("qualified") is not True:
            reasons.append(f"{name} gate is not qualified")
        privacy = report.get("privacy")
        if (not isinstance(privacy, dict)
                or privacy.get("contains_raw_request") is not False
                or privacy.get("contains_raw_response") is not False
                or privacy.get("contains_image_url_or_bytes") is not False
                or privacy.get("contains_credentials") is not False):
            reasons.append(f"{name} privacy contract failed")
        model = _evidence(
            case_maps[name], "models_262144_contract")
        if model.get("max_model_len") != 262144:
            reasons.append(f"{name} does not retain max_model_len=262144")

    expected_config = {
        "baseline_default": (400, 1),
        "candidate_default": (200, 1),
        "candidate_image2": (200, 2),
    }
    for name, (system_status, image_limit) in expected_config.items():
        config = reports[name].get("config") or {}
        if config.get(
                "multiple_system_parts_expected_status") != system_status:
            reasons.append(f"{name} system status contract mismatch")
        if config.get("image_limit") != image_limit:
            reasons.append(f"{name} image limit contract mismatch")

    for name in reports:
        evidence = _evidence(
            case_maps[name], "single_system_text_parts")
        if evidence.get("http_status") != 200:
            reasons.append(
                f"{name} single_system_text_parts did not return HTTP 200")
        if evidence.get("canonical_generation_exact") is not True:
            reasons.append(
                f"{name} single_system_text_parts changed generation")

    baseline_multi = _evidence(
        case_maps["baseline_default"], "multiple_system_text_parts")
    if baseline_multi.get("http_status") != 400:
        reasons.append(
            "baseline multiple_system_text_parts did not reproduce HTTP 400")
    for name in ("candidate_default", "candidate_image2"):
        evidence = _evidence(
            case_maps[name], "multiple_system_text_parts")
        if evidence.get("http_status") != 200:
            reasons.append(
                f"{name} multiple_system_text_parts did not return HTTP 200")
        if evidence.get("canonical_generation_exact") is not True:
            reasons.append(
                f"{name} multiple_system_text_parts changed generation")

    canonical_shas = {
        _message_sha(case_maps[name], "canonical_system_string")
        for name in reports
    }
    if None in canonical_shas or len(canonical_shas) != 1:
        reasons.append("canonical system output differs across runtime arms")

    one_image_shas = {
        _message_sha(case_maps[name], "one_image")
        for name in reports
    }
    if None in one_image_shas or len(one_image_shas) != 1:
        reasons.append("one-image output differs across runtime arms")

    for name, expected_count in (
            ("baseline_default", 1),
            ("candidate_default", 1),
            ("candidate_image2", 2)):
        replay = _evidence(case_maps[name], "image_at_limit_replay")
        if replay.get("image_count") != expected_count:
            reasons.append(f"{name} at-limit image count mismatch")
        if replay.get("exact_generation_match") is not True:
            reasons.append(f"{name} image replay is not deterministic")
        over_limit = _evidence(case_maps[name], "over_limit_image_400")
        if over_limit.get("http_status") != 400:
            reasons.append(f"{name} over-limit image did not return HTTP 400")
        if over_limit.get("image_count") != expected_count + 1:
            reasons.append(f"{name} over-limit image count mismatch")
        health = _evidence(case_maps[name], "post_4xx_health")
        if health.get("http_status") != 200:
            reasons.append(f"{name} service was unhealthy after HTTP 400")

    baseline_limit = _evidence(
        case_maps["baseline_default"], "image_at_limit_replay")
    candidate_limit = _evidence(
        case_maps["candidate_default"], "image_at_limit_replay")
    baseline_first = baseline_limit.get("first") or {}
    candidate_first = candidate_limit.get("first") or {}
    if (baseline_first.get("message_sha256") is None
            or baseline_first.get("message_sha256")
            != candidate_first.get("message_sha256")):
        reasons.append("default one-image output differs baseline/candidate")

    _validate_attribution(
        candidate_default_4xx, 2, "candidate_default", reasons)
    _validate_attribution(
        candidate_image2_4xx, 3, "candidate_image2", reasons)

    qualified = not reasons
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "qualified": qualified,
        "reasons": reasons,
        "checks": {
            "system_text_parts_http_fix_qualified": qualified,
            "one_image_non_regression_qualified": qualified,
            "explicit_image_two_structural_qualified": qualified,
            "candidate_4xx_v3_attribution_qualified": qualified,
        },
        "arm_contract": {
            name: {
                "multiple_system_parts_expected_status": values[0],
                "image_limit": values[1],
            }
            for name, values in expected_config.items()
        },
        "privacy": {
            "contains_raw_request": False,
            "contains_raw_response": False,
            "contains_image_url_or_bytes": False,
            "contains_credentials": False,
        },
        "semantic_quality_evaluated": False,
        "full_model_evaluated": False,
        "default_image_limit_change_authorized": False,
        "production_promotion_authorized": False,
    }


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate_default", type=Path)
    parser.add_argument("candidate_image2", type=Path)
    parser.add_argument("--candidate-default-4xx", type=Path, required=True)
    parser.add_argument("--candidate-image2-4xx", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = compare(
        json.loads(args.baseline.read_text(encoding="utf-8")),
        json.loads(args.candidate_default.read_text(encoding="utf-8")),
        json.loads(args.candidate_image2.read_text(encoding="utf-8")),
        json.loads(args.candidate_default_4xx.read_text(encoding="utf-8")),
        json.loads(args.candidate_image2_4xx.read_text(encoding="utf-8")),
    )
    _atomic_write(args.out, report)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
