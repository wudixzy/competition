#!/usr/bin/env python3
"""Validate quality-data provenance and the generated long-context matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_EXTERNAL = {
    "google_ifeval": (
        "966cd89545d6b6acfd7638bc708b98261ca58e84", "Apache-2.0"),
    "zai_longbench_v2": (
        "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9", "Apache-2.0"),
    "berkeley_bfcl_v1_3": (
        "ea13468e4423454d0c213704fb87cf7cb3990433", "Apache-2.0"),
}
EXPECTED_LOCAL = {
    "official_metric_collection": (
        "116e7edc617d8f96fc92caa3e75a3ba4692aae7619026896df1eaf69df12feac",
        6214,
    ),
    "official_workload_characteristics": (
        "c82acb0ca3f59577e27a99c549e417fd0e83626d696413cea26202aba86f228d",
        1835857,
    ),
    "selected_13_turn_regression": (
        "dac6afc77621b51dbc09cfa046c008a1e51a779bb771edcb27cb6a686f8884c8",
        2975,
    ),
}
TIERS = ("quick", "full", "extended")
TARGET_LENGTHS = {512, 4096, 32768, 65536, 131000, 235000, 261888}
REQUIRED_CAPABILITIES = {
    "cache", "partial_prefix", "multi_turn", "tool_calling",
    "large_tools_schema", "long_tool_result", "reasoning", "multimodal",
    "capacity",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_provenance(value: Any, root: Path = ROOT) -> list[str]:
    reasons = []
    if not isinstance(value, dict):
        return ["provenance root must be an object"]
    if (value.get("schema") != "bi100-quality-source-provenance-v1"
            or value.get("version") != 1):
        reasons.append("provenance schema or version is invalid")
    if value.get("repository_visibility_required") != "private":
        reasons.append("repository visibility requirement must remain private")
    sources = value.get("sources")
    if not isinstance(sources, list) or len(sources) != 7:
        reasons.append("provenance must contain seven frozen sources")
        sources = []
    source_map = {}
    required_fields = {
        "id", "kind", "author_or_org", "source_url", "revision", "license",
        "downloaded_at_utc", "file_sha256", "bytes", "split",
        "selection_rule", "transformation", "redistribution_allowed",
        "repository_snapshot", "privacy_or_restriction_review", "status",
    }
    for source in sources:
        if not isinstance(source, dict) or not required_fields <= set(source):
            reasons.append("provenance source fields are incomplete")
            continue
        source_id = source["id"]
        if source_id in source_map:
            reasons.append(f"duplicate provenance source {source_id}")
        source_map[source_id] = source
        if source.get("repository_snapshot") is True:
            repo_path = source.get("repository_path")
            if not isinstance(repo_path, str) or not (root / repo_path).is_file():
                reasons.append(f"repository snapshot path is invalid for {source_id}")
            elif (_sha256(root / repo_path) != source.get("file_sha256")
                  or (root / repo_path).stat().st_size != source.get("bytes")):
                reasons.append(f"repository snapshot identity differs for {source_id}")

    for source_id, (revision, license_name) in EXPECTED_EXTERNAL.items():
        source = source_map.get(source_id) or {}
        if (source.get("revision") != revision
                or source.get("license") != license_name
                or source.get("status") != "evaluated_not_ingested"
                or source.get("downloaded_at_utc") is not None
                or source.get("file_sha256") is not None
                or source.get("repository_snapshot") is not False):
            reasons.append(f"external candidate contract differs for {source_id}")
    deferred = source_map.get("swe_bench_verified") or {}
    if (deferred.get("status") != "deferred_license_review"
            or deferred.get("redistribution_allowed") is not False
            or deferred.get("revision") is not None):
        reasons.append("SWE-bench must remain deferred pending license review")
    for source_id, (digest, size) in EXPECTED_LOCAL.items():
        source = source_map.get(source_id) or {}
        if source.get("file_sha256") != digest or source.get("bytes") != size:
            reasons.append(f"local source identity differs for {source_id}")

    serialized = json.dumps(value, ensure_ascii=True).lower()
    for marker in ("begin openssh private key", "github_pat_", "ghp_"):
        if marker in serialized:
            reasons.append("provenance contains a credential marker")
    return reasons


def validate_operator_files(
    metrics: Path,
    workload: Path,
) -> list[str]:
    reasons = []
    expected = (
        (metrics, *EXPECTED_LOCAL["official_metric_collection"]),
        (workload, *EXPECTED_LOCAL["official_workload_characteristics"]),
    )
    for path, digest, size in expected:
        if not path.is_file():
            reasons.append(f"operator source is missing: {path.name}")
        elif _sha256(path) != digest or path.stat().st_size != size:
            reasons.append(f"operator source identity differs: {path.name}")
    return reasons


def validate_matrix(value: Any) -> list[str]:
    reasons = []
    if not isinstance(value, dict):
        return ["matrix root must be an object"]
    if (value.get("schema") != "bi100-long-context-quality-matrix-v1"
            or value.get("version") != 1
            or value.get("seed") != 20260724):
        reasons.append("matrix schema, version, or seed is invalid")
    if (value.get("max_model_len") != 262144
            or value.get("stable_order") is not True
            or value.get("contains_raw_requests") is not False
            or value.get("promotion_tier") != "extended"):
        reasons.append("matrix global contract is invalid")
    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) != 12:
        reasons.append("matrix must contain twelve cases")
        cases = []
    expected_fields = {
        "ordinal", "id", "tier", "target_prompt_tokens", "max_tokens",
        "min_completion_tokens", "request_shape", "cache_scenario",
        "capabilities", "equivalence", "validation",
    }
    ids = []
    lengths = set()
    capabilities = set()
    for index, case in enumerate(cases, 1):
        if not isinstance(case, dict) or set(case) != expected_fields:
            reasons.append(f"matrix case {index} fields are invalid")
            continue
        if case["ordinal"] != index:
            reasons.append(f"matrix case {index} ordinal differs")
        if case["tier"] not in TIERS:
            reasons.append(f"matrix case {index} tier is invalid")
        if not isinstance(case["id"], str) or not case["id"]:
            reasons.append(f"matrix case {index} id is invalid")
        else:
            ids.append(case["id"])
        target = case["target_prompt_tokens"]
        maximum = case["max_tokens"]
        minimum = case["min_completion_tokens"]
        if (not all(isinstance(value, int) and not isinstance(value, bool)
                    for value in (target, maximum, minimum))
                or target <= 0 or maximum <= 0 or not 0 <= minimum <= maximum
                or target + maximum > 262144):
            reasons.append(f"matrix case {index} token budget is invalid")
        lengths.add(target)
        case_capabilities = case["capabilities"]
        if (not isinstance(case_capabilities, list)
                or not case_capabilities
                or len(case_capabilities) != len(set(case_capabilities))):
            reasons.append(f"matrix case {index} capabilities are invalid")
        else:
            capabilities.update(case_capabilities)
    if len(ids) != len(set(ids)):
        reasons.append("matrix case ids must be unique")
    if not TARGET_LENGTHS <= lengths:
        reasons.append("matrix does not cover every fixed context length")
    if not REQUIRED_CAPABILITIES <= capabilities:
        reasons.append("matrix does not cover every required capability")
    case_map = {case.get("id"): case for case in cases if isinstance(case, dict)}
    near = case_map.get("near_262k_capacity") or {}
    if near.get("target_prompt_tokens", 0) + near.get("max_tokens", 0) != 262144:
        reasons.append("near-262K case must exercise the exact capacity boundary")
    agent = case_map.get("235k_agent_large_output_budget") or {}
    if agent.get("target_prompt_tokens") != 235000 or agent.get("max_tokens") != 8192:
        reasons.append("235K Agent case must retain the official large output budget")
    return reasons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provenance", type=Path,
        default=ROOT / "quality/source_provenance.v1.json")
    parser.add_argument(
        "--matrix", type=Path,
        default=ROOT / "quality/long_context_matrix.v1.json")
    parser.add_argument("--metrics-source", type=Path)
    parser.add_argument("--workload-source", type=Path)
    args = parser.parse_args()
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    reasons = validate_provenance(provenance)
    reasons.extend(validate_matrix(matrix))
    operator_checked = (
        args.metrics_source is not None and args.workload_source is not None)
    if (args.metrics_source is None) != (args.workload_source is None):
        reasons.append("both operator source paths must be provided together")
    elif operator_checked:
        reasons.extend(validate_operator_files(
            args.metrics_source, args.workload_source))
    result = {
        "schema": "bi100-quality-data-manifest-validation-v1",
        "version": 1,
        "qualified": not reasons,
        "reasons": reasons,
        "provenance_sha256": _sha256(args.provenance),
        "matrix_sha256": _sha256(args.matrix),
        "operator_sources_checked": operator_checked,
        "source_count": len(provenance.get("sources") or []),
        "matrix_case_count": len(matrix.get("cases") or []),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not reasons else 1


if __name__ == "__main__":
    raise SystemExit(main())
