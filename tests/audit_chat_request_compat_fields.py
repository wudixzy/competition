#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "bi100-chat-completion-request-field-audit-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _chat_request_fields(protocol_path: Path) -> set[str]:
    tree = ast.parse(protocol_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ChatCompletionRequest":
            return {
                item.target.id
                for item in node.body
                if isinstance(item, ast.AnnAssign)
                and isinstance(item.target, ast.Name)
                and not item.target.id.startswith("_")
            }
    raise ValueError("ChatCompletionRequest class is missing")


def _class_methods(protocol_path: Path, class_name: str) -> set[str]:
    tree = ast.parse(protocol_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise ValueError(f"{class_name} class is missing")


def audit(root: Path) -> dict[str, Any]:
    protocol_path = root / "qwen3_6_scripts" / "protocol.py"
    contract_path = (
        root / "quality" / "chat_completion_request_compatibility.v1.json"
    )
    interactions_path = (
        root / "quality" / "chat_completion_field_interactions.v1.json"
    )
    protocol_tests_path = root / "tests" / "test_protocol_unit.py"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    interactions = json.loads(
        interactions_path.read_text(encoding="utf-8"))
    fields = _chat_request_fields(protocol_path)
    chat_methods = _class_methods(protocol_path, "ChatCompletionRequest")
    response_format_methods = _class_methods(protocol_path, "ResponseFormat")
    protocol_tests = protocol_tests_path.read_text(encoding="utf-8")
    reasons: list[str] = []

    if contract.get("schema") != "bi100-chat-completion-request-compatibility-v1":
        reasons.append("compatibility contract schema differs")
    if contract.get("strict_extra_policy") != "forbid":
        reasons.append("strict extra-field policy differs")
    if 'ConfigDict(extra="forbid")' not in protocol_path.read_text(
            encoding="utf-8"):
        reasons.append("runtime protocol no longer forbids unknown fields")

    alias_fields = {
        field
        for group in contract.get("lossless_alias_groups", [])
        for field in group.get("top_level_fields", [])
    }
    missing_alias_fields = sorted(alias_fields - fields)
    if missing_alias_fields:
        reasons.append(
            f"lossless alias fields are missing: {missing_alias_fields}")

    upstream_fields = set(
        contract.get("upstream_reference_top_level_fields", []))
    upstream_only = upstream_fields - fields
    classified_upstream_only = set(
        contract.get("classified_upstream_only_fields", {}))
    if upstream_only != classified_upstream_only:
        reasons.append(
            "upstream-only field classification differs: "
            f"actual={sorted(upstream_only)} "
            f"classified={sorted(classified_upstream_only)}")

    openai_fields = set(
        contract.get("openai_reference_top_level_fields", []))
    openai_only = openai_fields - fields
    classified_openai_only = set(
        contract.get("classified_openai_only_fields", {}))
    if openai_only != classified_openai_only:
        reasons.append(
            "OpenAI-only field classification differs: "
            f"actual={sorted(openai_only)} "
            f"classified={sorted(classified_openai_only)}")

    review_required = set(
        contract.get("review_required_non_alias_fields", {}))
    unexpectedly_accepted = sorted(review_required.intersection(fields))
    if unexpectedly_accepted:
        reasons.append(
            "review-required fields became accepted without contract update: "
            f"{unexpectedly_accepted}")

    if interactions.get("schema") != \
            "bi100-chat-completion-field-interactions-v1":
        reasons.append("field interaction contract schema differs")
    if interactions.get("strict_extra_policy") != "forbid":
        reasons.append("field interaction strict extra policy differs")

    relationship_names: list[str] = []
    for relationship in interactions.get("relationships", []):
        name = relationship.get("name", "unknown")
        relationship_names.append(name)
        missing_fields = sorted(
            set(relationship.get("top_level_fields", [])) - fields)
        if missing_fields:
            reasons.append(
                f"interaction {name} fields are missing: {missing_fields}")
        missing_methods = sorted(
            set(relationship.get("required_methods", [])) - chat_methods)
        if missing_methods:
            reasons.append(
                f"interaction {name} methods are missing: {missing_methods}")
        missing_response_methods = sorted(
            set(relationship.get(
                "required_response_format_methods", []))
            - response_format_methods)
        if missing_response_methods:
            reasons.append(
                f"interaction {name} response-format methods are missing: "
                f"{missing_response_methods}")
        for test_name in relationship.get("required_tests", []):
            if f"def {test_name}(" not in protocol_tests:
                reasons.append(
                    f"interaction {name} test is missing: {test_name}")

    fail_closed = set(interactions.get("fail_closed_lookalikes", {}))
    accepted_lookalikes = sorted(fail_closed.intersection(fields))
    if accepted_lookalikes:
        reasons.append(
            "fail-closed lookalike fields became accepted: "
            f"{accepted_lookalikes}")
    classified_lookalikes = classified_openai_only.union(review_required)
    unclassified_lookalikes = sorted(fail_closed - classified_lookalikes)
    if unclassified_lookalikes:
        reasons.append(
            "fail-closed lookalikes are unclassified: "
            f"{unclassified_lookalikes}")

    required_fail_closed_test = interactions.get(
        "required_fail_closed_test")
    if (not isinstance(required_fail_closed_test, str)
            or f"def {required_fail_closed_test}(" not in protocol_tests):
        reasons.append("fail-closed lookalike test is missing")
    malformed_test = interactions.get(
        "malformed_input_rule", {}).get("required_test")
    if (not isinstance(malformed_test, str)
            or f"def {malformed_test}(" not in protocol_tests):
        reasons.append("malformed multi-field input test is missing")

    return {
        "schema": SCHEMA,
        "qualified": not reasons,
        "reasons": reasons,
        "protocol_path": str(protocol_path.relative_to(root)),
        "protocol_sha256": _sha256(protocol_path),
        "contract_path": str(contract_path.relative_to(root)),
        "contract_sha256": _sha256(contract_path),
        "interaction_contract_path": str(
            interactions_path.relative_to(root)),
        "interaction_contract_sha256": _sha256(interactions_path),
        "protocol_tests_path": str(protocol_tests_path.relative_to(root)),
        "protocol_tests_sha256": _sha256(protocol_tests_path),
        "strict_extra_policy": contract.get("strict_extra_policy"),
        "local_field_count": len(fields),
        "lossless_alias_fields": sorted(alias_fields),
        "classified_upstream_only_fields": sorted(classified_upstream_only),
        "classified_openai_only_fields": sorted(classified_openai_only),
        "review_required_non_alias_fields": sorted(review_required),
        "interaction_relationships": relationship_names,
        "fail_closed_lookalike_fields": sorted(fail_closed),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = audit(args.root.resolve())
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
