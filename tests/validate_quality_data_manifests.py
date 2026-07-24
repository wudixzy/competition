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
EXPECTED_GENERATED_ASSETS = {
    "red_png_data_url_sha256": (
        "49571191557f87f03b1f83e2f233241df962dae58e0604fc1c1c7d7d51c60da4"),
    "blue_png_data_url_sha256": (
        "57d7214b7255958657b84ae8922a1c886b4933cbbd0242084ec0cdb5eb0d9d55"),
    "large_tools_65k_sha256": (
        "0a2f2730ca84ad390766666e1f4cb622fe9f1030e21e89529375ddc469877ac8"),
    "large_tools_235k_sha256": (
        "846f986c5cd8d376fa41cb73040fe2937ed6822ce356f87b5df73ac174d4ee14"),
    "fetch_record_tool_sha256": (
        "f3da291816c2a09bfd5ca73709a678da32fc1b423ffdb4626c462f65e9ec11f4"),
}
EXPECTED_MATRIX_SHA256 = (
    "3217ec047f7b78af6747269c3f85baed6bfdd86c6527aca6335dbfa7d9f0452b"
)
EXPECTED_AGENT_MATRIX_SHA256 = (
    "962d19f51cfbeb3f414e62444a225029616ed547682e5a97219b0af98c8959ba"
)
EXPECTED_AGENT_CASES = [
    "forced_terminal", "forced_read", "forced_edit", "forced_web_search",
    "auto_terminal", "stream_forced_terminal", "stream_auto_terminal",
    "tool_result_roundtrip", "long_history", "large_tool_schema",
    "multiple_system",
]


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
    if (value.get("schema") != "bi100-long-context-quality-matrix-v2"
            or value.get("version") != 2
            or value.get("seed") != 20260724):
        reasons.append("matrix schema, version, or seed is invalid")
    if (value.get("max_model_len") != 262144
            or value.get("stable_order") is not True
            or value.get("contains_raw_requests") is not False
            or value.get("promotion_tier") != "extended"):
        reasons.append("matrix global contract is invalid")
    expected_provenance = {
        "kind": "deterministic_project_generated",
        "author_or_org": "BI100 competition project",
        "license": "project-internal",
        "redistribution_allowed": False,
        "contains_external_dataset_rows": False,
        "contains_personal_or_secret_data": False,
    }
    if value.get("provenance") != expected_provenance:
        reasons.append("matrix provenance contract is invalid")
    generator = value.get("generator") or {}
    if (generator.get("id") != "bi100-long-context-quality-generator"
            or generator.get("version") != 2
            or generator.get("runner") != "tests/long_context_quality_api.py"
            or generator.get("exact_prompt_module")
            != "tests/exact_chat_prompt.py"
            or not isinstance(generator.get("construction"), str)
            or not generator["construction"]):
        reasons.append("matrix generator contract is invalid")
    expected_tokenization = {
        "text_cases": "server_exact_post_chat_template_tokens",
        "multimodal_cases": (
            "local_post_chat_template_tokens_plus_server_vision_expansion"),
        "runtime_tokenizer_artifact_sha256_required": True,
        "runtime_chat_template_sha256_required": True,
        "chat_template_kwargs_mode_required": True,
    }
    if value.get("tokenization_contract") != expected_tokenization:
        reasons.append("matrix tokenization contract is invalid")
    expected_cache_trace = {
        "version": 4,
        "required_for_multimodal_isolation": True,
        "same_image_prompt_chain_must_match": True,
        "different_image_first_prompt_hash_must_differ": True,
        "contains_raw_tokens_or_media": False,
    }
    if value.get("cache_trace_contract") != expected_cache_trace:
        reasons.append("matrix cache trace contract is invalid")
    if value.get("generated_assets") != EXPECTED_GENERATED_ASSETS:
        reasons.append("matrix generated asset identities differ")
    if value.get("capacity_prompt_boundaries") != [261887, 261888]:
        reasons.append("matrix capacity boundaries differ")
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
    serialized = json.dumps(value, ensure_ascii=True).lower()
    if any(marker in serialized for marker in (
            "begin openssh private key", "github_pat_", "ghp_",
            "modelhub_access_token")):
        reasons.append("matrix contains a credential marker")
    return reasons


def validate_agent_manifest(value: Any) -> list[str]:
    reasons = []
    if not isinstance(value, dict):
        return ["Agent manifest root must be an object"]
    if (value.get("schema") != "bi100-agent-workload-manifest-v1"
            or value.get("version") != 1
            or value.get("seed") != 20260716
            or value.get("revision") != "agent-workload-v1.1-seed-20260716"):
        reasons.append("Agent manifest schema, version, or revision is invalid")
    expected_privacy = {
        "contains_raw_requests": False,
        "contains_raw_model_outputs": False,
        "contains_tool_arguments": False,
    }
    if (value.get("author_or_org")
            != "private BI100 optimization project"
            or value.get("source_kind") != "project_generated"
            or value.get("license") != "private project data"
            or value.get("redistribution_allowed") is not False
            or value.get("contains_external_dataset_rows") is not False
            or value.get("contains_restricted_evaluation_data") is not False
            or value.get("contains_credentials_or_private_user_data") is not False
            or value.get("expected_report_privacy") != expected_privacy):
        reasons.append("Agent manifest provenance or privacy contract is invalid")
    cases = value.get("cases")
    if (not isinstance(cases, list)
            or [case.get("id") for case in cases if isinstance(case, dict)]
            != EXPECTED_AGENT_CASES
            or any(set(case) != {"id", "goal"} for case in cases
                   if isinstance(case, dict))):
        reasons.append("Agent manifest case identity or order differs")
    serialized = json.dumps(value, ensure_ascii=True).lower()
    if any(marker in serialized for marker in (
            "begin openssh private key", "github_pat_", "ghp_",
            "modelhub_access_token")):
        reasons.append("Agent manifest contains a credential marker")
    return reasons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provenance", type=Path,
        default=ROOT / "quality/source_provenance.v1.json")
    parser.add_argument(
        "--matrix", type=Path,
        default=ROOT / "quality/long_context_matrix.v2.json")
    parser.add_argument(
        "--agent-matrix", type=Path,
        default=ROOT / "quality/agent_workload_matrix.v1.json")
    parser.add_argument("--metrics-source", type=Path)
    parser.add_argument("--workload-source", type=Path)
    args = parser.parse_args()
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    agent_matrix = json.loads(args.agent_matrix.read_text(encoding="utf-8"))
    reasons = validate_provenance(provenance)
    reasons.extend(validate_matrix(matrix))
    reasons.extend(validate_agent_manifest(agent_matrix))
    if _sha256(args.matrix) != EXPECTED_MATRIX_SHA256:
        reasons.append("long-context matrix SHA-256 differs")
    if _sha256(args.agent_matrix) != EXPECTED_AGENT_MATRIX_SHA256:
        reasons.append("Agent workload matrix SHA-256 differs")
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
        "agent_matrix_sha256": _sha256(args.agent_matrix),
        "operator_sources_checked": operator_checked,
        "source_count": len(provenance.get("sources") or []),
        "matrix_case_count": len(matrix.get("cases") or []),
        "agent_case_count": len(agent_matrix.get("cases") or []),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not reasons else 1


if __name__ == "__main__":
    raise SystemExit(main())
