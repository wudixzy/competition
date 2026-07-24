#!/usr/bin/env python3
"""Validate the frozen BI100 model-quality metric manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "bi100-quality-metric-manifest-v1"
VERSION = 1
EXPECTED_SOURCE_SHA256 = (
    "116e7edc617d8f96fc92caa3e75a3ba4692aae7619026896df1eaf69df12feac"
)
EXPECTED_MANIFEST_SHA256 = (
    "fe9b958610d9d0df8f54504d9c149442f145226c03cf76668711d2d38ed51d0e"
)
EXPECTED_CASES = 53
TIER_ORDER = ("quick", "full", "extended")
COMPARISON_MODES = {"contract", "exact", "semantic"}
EXPECTED_ALLOWED_SKIPS = {"direct": ["n_2"]}


def validate(value: Any) -> list[str]:
    reasons: list[str] = []
    if not isinstance(value, dict):
        return ["manifest root must be an object"]
    if value.get("schema") != SCHEMA or value.get("version") != VERSION:
        reasons.append("manifest schema or version is invalid")
    source = value.get("source")
    if (not isinstance(source, dict)
            or source.get("sha256") != EXPECTED_SOURCE_SHA256
            or source.get("rows") != EXPECTED_CASES
            or source.get("snapshot_redistributed") is not False):
        reasons.append("metric source provenance is invalid")
    if value.get("promotion_tier") != "extended":
        reasons.append("promotion tier must be extended")
    if value.get("allowed_skips") != EXPECTED_ALLOWED_SKIPS:
        reasons.append("allowed skip policy is invalid")
    if tuple(value.get("tier_order") or ()) != TIER_ORDER:
        reasons.append("tier order is invalid")

    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASES:
        reasons.append(f"manifest must contain {EXPECTED_CASES} cases")
        cases = []
    expected_keys = {"ordinal", "id", "group", "tier", "comparison"}
    ids = []
    for index, case in enumerate(cases, 1):
        if not isinstance(case, dict) or set(case) != expected_keys:
            reasons.append(f"case {index} fields are invalid")
            continue
        if case.get("ordinal") != index:
            reasons.append(f"case {index} ordinal is invalid")
        case_id = case.get("id")
        if (not isinstance(case_id, str) or not case_id
                or not case_id.replace("_", "").isalnum()
                or case_id.lower() != case_id):
            reasons.append(f"case {index} id is invalid")
        else:
            ids.append(case_id)
        if not isinstance(case.get("group"), str) or not case["group"]:
            reasons.append(f"case {index} group is invalid")
        if case.get("tier") not in TIER_ORDER:
            reasons.append(f"case {index} tier is invalid")
        if case.get("comparison") not in COMPARISON_MODES:
            reasons.append(f"case {index} comparison mode is invalid")
    if len(ids) != len(set(ids)):
        reasons.append("case ids must be unique")
    if not any(case.get("tier") == "extended" for case in cases
               if isinstance(case, dict)):
        reasons.append("extended tier must contain at least one case")
    return reasons


def validate_source(path: Path) -> list[str]:
    payload = path.read_bytes()
    reasons = []
    if hashlib.sha256(payload).hexdigest() != EXPECTED_SOURCE_SHA256:
        reasons.append("metric source SHA-256 differs")
    rows = payload.decode("utf-8").splitlines()
    if len(rows) != EXPECTED_CASES + 1:
        reasons.append("metric source row count differs")
    elif rows[0] != "分组\t测试名称\t验证目的":
        reasons.append("metric source header differs")
    elif any(len(row.split("\t")) != 3 for row in rows[1:]):
        reasons.append("metric source TSV shape differs")
    return reasons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=Path("quality/official_metrics_manifest.v1.json"),
    )
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    payload = args.manifest.read_bytes()
    value = json.loads(payload)
    reasons = validate(value)
    if hashlib.sha256(payload).hexdigest() != EXPECTED_MANIFEST_SHA256:
        reasons.append("metric manifest SHA-256 differs")
    source_checked = args.source is not None
    if args.source is not None:
        reasons.extend(validate_source(args.source))
    result = {
        "schema": "bi100-quality-metric-manifest-validation-v1",
        "version": 1,
        "qualified": not reasons,
        "reasons": reasons,
        "cases": len(value.get("cases") or []) if isinstance(value, dict) else 0,
        "promotion_tier": (
            value.get("promotion_tier") if isinstance(value, dict) else None),
        "source_checked": source_checked,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not reasons else 1


if __name__ == "__main__":
    raise SystemExit(main())
