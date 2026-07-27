#!/usr/bin/env python3
"""Compare baseline and candidate Qwen3.6 tool HTTP reports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any


Json = dict[str, Any]
SCHEMA = "qwen36-diagnostic-tool-http-ab-v2"
VERSION = 2
GATE_SCHEMA = "qwen36-diagnostic-tool-http-gate-v2"
ATTRIBUTION_SCHEMA = "bi100-api-4xx-attribution-v3"
EXPECTED_4XX_REASONS = {
    "invalid_tool_arguments_json": 1,
    "request_validation_tool_strict": 1,
    "unsupported_tool_choice_required": 1,
}
CASE_NAMES = (
    "models_262144_contract",
    "function_tool_default",
    "function_tool_strict_false",
    "tool_arguments_json_string",
    "tool_arguments_json_object",
    "stream_function_tool_default",
    "stream_function_tool_strict_false",
    "stream_tool_arguments_json_string",
    "stream_tool_arguments_json_object",
    "tool_arguments_invalid_json_400",
    "function_tool_strict_true_400",
    "tool_choice_required_400",
    "post_4xx_health",
)


def _case_map(report: Json, label: str, reasons: list[str]) -> dict[str, Json]:
    if report.get("schema") != GATE_SCHEMA:
        reasons.append(f"{label} gate schema mismatch")
    if report.get("qualified") is not True:
        reasons.append(f"{label} gate is not qualified")
    cases = report.get("cases")
    if not isinstance(cases, list):
        reasons.append(f"{label} cases are missing")
        return {}
    mapped = {
        case.get("name"): case
        for case in cases
        if isinstance(case, dict) and isinstance(case.get("name"), str)
    }
    if len(mapped) != len(cases):
        reasons.append(f"{label} contains duplicate or unnamed cases")
    if set(mapped) != set(CASE_NAMES):
        reasons.append(f"{label} case set mismatch")
    if any(case.get("ok") is not True for case in mapped.values()):
        reasons.append(f"{label} contains a failed case")
    return mapped


def _evidence(cases: dict[str, Json], name: str) -> Json:
    value = cases.get(name, {}).get("evidence")
    return value if isinstance(value, dict) else {}


def _status(cases: dict[str, Json], name: str) -> Any:
    return _evidence(cases, name).get("http_status")


def _generation_contract(value: Json) -> tuple[Any, ...]:
    return (
        value.get("message_sha256"),
        value.get("finish_reason"),
        value.get("prompt_tokens"),
        value.get("completion_tokens"),
        value.get("has_content"),
        value.get("has_reasoning_content"),
        value.get("tool_call_count"),
    )


def _stream_generation_contract(value: Json) -> tuple[Any, ...]:
    return (
        value.get("semantic_output_sha256"),
        value.get("finish_reason"),
        value.get("prompt_tokens"),
        value.get("completion_tokens"),
        value.get("has_content"),
        value.get("has_reasoning_content"),
        value.get("tool_call_count"),
    )


def compare(
    baseline: Json,
    candidate: Json,
    candidate_attribution: Json,
) -> Json:
    reasons: list[str] = []
    baseline_cases = _case_map(baseline, "baseline", reasons)
    candidate_cases = _case_map(candidate, "candidate", reasons)

    if baseline.get("config", {}).get(
            "strict_false_expected_status") != 400:
        reasons.append("baseline strict=false contract is not HTTP 400")
    if baseline.get("config", {}).get(
            "object_history_expected_status") != 400:
        reasons.append("baseline object-history contract is not HTTP 400")
    if candidate.get("config", {}).get(
            "strict_false_expected_status") != 200:
        reasons.append("candidate strict=false contract is not HTTP 200")
    if candidate.get("config", {}).get(
            "object_history_expected_status") != 200:
        reasons.append("candidate object-history contract is not HTTP 200")

    for name in (
            "function_tool_strict_false",
            "tool_arguments_json_object",
            "stream_function_tool_strict_false",
            "stream_tool_arguments_json_object"):
        if _status(baseline_cases, name) != 400:
            reasons.append(f"baseline {name} did not reproduce HTTP 400")
        if _status(candidate_cases, name) != 200:
            reasons.append(f"candidate {name} did not return HTTP 200")

    if _evidence(
            candidate_cases,
            "function_tool_strict_false",
    ).get("default_generation_exact") is not True:
        reasons.append("candidate strict=false generation is not exact")
    if _evidence(
            candidate_cases,
            "tool_arguments_json_object",
    ).get("string_generation_exact") is not True:
        reasons.append("candidate object-history generation is not exact")
    if _evidence(
            candidate_cases,
            "stream_function_tool_strict_false",
    ).get("default_stream_generation_exact") is not True:
        reasons.append(
            "candidate streaming strict=false generation is not exact")
    if _evidence(
            candidate_cases,
            "stream_tool_arguments_json_object",
    ).get("string_stream_generation_exact") is not True:
        reasons.append(
            "candidate streaming object-history generation is not exact")

    for name in (
            "function_tool_default",
            "tool_arguments_json_string"):
        left = _evidence(baseline_cases, name)
        right = _evidence(candidate_cases, name)
        if _generation_contract(left) != _generation_contract(right):
            reasons.append(f"{name} changed across runtime overlays")
    for name in (
            "stream_function_tool_default",
            "stream_tool_arguments_json_string"):
        left = _evidence(baseline_cases, name)
        right = _evidence(candidate_cases, name)
        if _stream_generation_contract(
                left) != _stream_generation_contract(right):
            reasons.append(f"{name} changed across runtime overlays")

    baseline_streaming = baseline.get("streaming_contract") or {}
    candidate_streaming = candidate.get("streaming_contract") or {}
    if baseline_streaming.get("qualified") is not True:
        reasons.append("baseline streaming contract is not qualified")
    if candidate_streaming.get("qualified") is not True:
        reasons.append("candidate streaming contract is not qualified")
    if candidate_streaming.get(
            "accepted_equivalence_qualified") is not True:
        reasons.append(
            "candidate streaming accepted equivalence is not qualified")

    for name in (
            "tool_arguments_invalid_json_400",
            "function_tool_strict_true_400",
            "tool_choice_required_400"):
        if _status(baseline_cases, name) != 400:
            reasons.append(f"baseline {name} did not return HTTP 400")
        if _status(candidate_cases, name) != 400:
            reasons.append(f"candidate {name} did not return HTTP 400")
    if _status(candidate_cases, "post_4xx_health") != 200:
        reasons.append("candidate health failed after expected 4xx responses")

    if candidate_attribution.get("schema") != ATTRIBUTION_SCHEMA:
        reasons.append("candidate 4xx attribution schema mismatch")
    if candidate_attribution.get("qualified") is not True:
        reasons.append("candidate 4xx attribution is not qualified")
    if candidate_attribution.get("complete") is not True:
        reasons.append("candidate 4xx attribution is incomplete")
    if candidate_attribution.get("chat_4xx_access_count") != 3:
        reasons.append("candidate did not produce exactly three expected 4xx")
    if candidate_attribution.get("attributed_count") != 3:
        reasons.append("candidate did not attribute all expected 4xx")
    by_reason = candidate_attribution.get("by_reason") or {}
    if by_reason != EXPECTED_4XX_REASONS:
        reasons.append("candidate 4xx reason set differs from the contract")
    if any(
            name in by_reason
            for name in (
                "unclassified_chat_error",
                "request_validation_unknown",
                "unknown",
            )):
        reasons.append("candidate contains an unclassified tool 4xx")
    privacy = candidate_attribution.get("privacy") or {}
    if any(privacy.values()):
        reasons.append("candidate 4xx attribution contains private payloads")

    checks = {
        "strict_false_http_fix_qualified": (
            _status(baseline_cases, "function_tool_strict_false") == 400
            and _status(candidate_cases, "function_tool_strict_false") == 200
            and _evidence(
                candidate_cases,
                "function_tool_strict_false",
            ).get("default_generation_exact") is True
        ),
        "object_history_http_fix_qualified": (
            _status(baseline_cases, "tool_arguments_json_object") == 400
            and _status(candidate_cases, "tool_arguments_json_object") == 200
            and _evidence(
                candidate_cases,
                "tool_arguments_json_object",
            ).get("string_generation_exact") is True
        ),
        "streaming_strict_false_http_fix_qualified": (
            _status(
                baseline_cases,
                "stream_function_tool_strict_false",
            ) == 400
            and _status(
                candidate_cases,
                "stream_function_tool_strict_false",
            ) == 200
            and _evidence(
                candidate_cases,
                "stream_function_tool_strict_false",
            ).get("default_stream_generation_exact") is True
        ),
        "streaming_object_history_http_fix_qualified": (
            _status(
                baseline_cases,
                "stream_tool_arguments_json_object",
            ) == 400
            and _status(
                candidate_cases,
                "stream_tool_arguments_json_object",
            ) == 200
            and _evidence(
                candidate_cases,
                "stream_tool_arguments_json_object",
            ).get("string_stream_generation_exact") is True
        ),
        "streaming_contract_qualified": (
            baseline_streaming.get("qualified") is True
            and candidate_streaming.get("qualified") is True
            and candidate_streaming.get(
                "accepted_equivalence_qualified") is True
        ),
        "candidate_4xx_attribution_qualified": (
            candidate_attribution.get("qualified") is True
            and candidate_attribution.get("chat_4xx_access_count") == 3
            and candidate_attribution.get("attributed_count") == 3
            and by_reason == EXPECTED_4XX_REASONS
        ),
    }
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "qualified": not reasons,
        "reasons": reasons,
        "checks": checks,
        "semantic_quality_evaluated": False,
        "full_model_evaluated": False,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--candidate-4xx",
        type=Path,
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    result = compare(
        json.loads(args.baseline.read_text(encoding="utf-8")),
        json.loads(args.candidate.read_text(encoding="utf-8")),
        json.loads(args.candidate_4xx.read_text(encoding="utf-8")),
    )
    _atomic_write(args.out, result)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
