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


def audit(root: Path) -> dict[str, Any]:
    protocol_path = root / "qwen3_6_scripts" / "protocol.py"
    contract_path = (
        root / "quality" / "chat_completion_request_compatibility.v1.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    fields = _chat_request_fields(protocol_path)
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

    review_required = set(
        contract.get("review_required_non_alias_fields", {}))
    unexpectedly_accepted = sorted(review_required.intersection(fields))
    if unexpectedly_accepted:
        reasons.append(
            "review-required fields became accepted without contract update: "
            f"{unexpectedly_accepted}")

    return {
        "schema": SCHEMA,
        "qualified": not reasons,
        "reasons": reasons,
        "protocol_path": str(protocol_path.relative_to(root)),
        "protocol_sha256": _sha256(protocol_path),
        "contract_path": str(contract_path.relative_to(root)),
        "contract_sha256": _sha256(contract_path),
        "strict_extra_policy": contract.get("strict_extra_policy"),
        "local_field_count": len(fields),
        "lossless_alias_fields": sorted(alias_fields),
        "classified_upstream_only_fields": sorted(classified_upstream_only),
        "review_required_non_alias_fields": sorted(review_required),
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
