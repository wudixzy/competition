#!/usr/bin/env python3
"""Reconcile privacy-safe chat 4xx markers with access-log responses."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any

REPORT_SCHEMA = "bi100-api-4xx-attribution-v3"
REPORT_VERSION = 3
MARKER = "[BI100 4XX]"
DETAIL_MARKER = "[BI100 4XX DETAIL]"
ACCESS_RE = re.compile(
    r'"POST /v1/chat/completions HTTP/1\.[01]" (?P<code>4\d\d)\b'
)
FIELD_RE = re.compile(r"(?P<key>[a-z_]+)=(?P<value>[^\s]+)")
VALIDATION_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
ALLOWED_ENDPOINTS = {"chat", "request_validation"}
ALLOWED_REASONS = {
    "chat_template_failed",
    "empty_messages",
    "context_length_exceeded",
    "invalid_tool_arguments_json",
    "invalid_tool_arguments_type",
    "invalid_max_tokens",
    "invalid_top_p",
    "image_count_limit",
    "image_model_type_unsupported",
    "multimodal_load_failed",
    "n_exceeds_max_num_seqs",
    "request_validation_generation",
    "request_validation_message_content",
    "request_validation_message_role",
    "request_validation_message_tool_call_id",
    "request_validation_message_tool_calls",
    "request_validation_messages",
    "request_validation_model",
    "request_validation_other",
    "request_validation_response_format",
    "request_validation_sampling",
    "request_validation_streaming",
    "request_validation_tool_choice",
    "request_validation_tool_parameters",
    "request_validation_tool_strict",
    "request_validation_tools",
    "request_validation_unknown",
    "tool_parser_unavailable",
    "unclassified_chat_error",
    "unsupported_tool_choice_required",
}
INTEGER_SHAPE_FIELDS = (
    "messages",
    "systems",
    "system_part_msgs",
    "system_text_parts",
    "system_other_parts",
    "tools",
    "tool_msgs",
    "assistant_tool_msgs",
    "strict_false",
    "strict_true",
    "images",
    "image_data",
    "image_remote",
    "image_other",
    "stream",
)
SHAPE_FIELDS = INTEGER_SHAPE_FIELDS + ("choice", "n")
LEGACY_INTEGER_SHAPE_FIELDS = (
    "messages",
    "systems",
    "tools",
    "tool_msgs",
    "assistant_tool_msgs",
    "strict_false",
    "strict_true",
    "image",
    "stream",
)
LEGACY_SHAPE_FIELDS = LEGACY_INTEGER_SHAPE_FIELDS + ("choice", "n")
ALLOWED_CHOICES = {"unset", "none", "auto", "required", "named", "other"}
ALLOWED_DETAIL_STAGES = {"chat_template", "multimodal_load"}


def require_int(value: str | None, field: str, *, minimum: int = 0) -> int:
    if value is None:
        raise ValueError(f"missing {field}")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"invalid {field}")
    return parsed


def require_validation_identifier(
        value: str | None, field: str, *, legacy_default: str) -> str:
    if value is None:
        return legacy_default
    if not VALIDATION_IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"invalid {field}")
    return value


def parse_marker(line: str) -> dict[str, Any]:
    payload = line.split(MARKER, 1)[1]
    fields = {
        match.group("key"): match.group("value")
        for match in FIELD_RE.finditer(payload)
    }
    endpoint = fields.get("endpoint")
    reason = fields.get("reason")
    if endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError("unknown endpoint")
    if reason not in ALLOWED_REASONS:
        raise ValueError("unknown reason")

    code = require_int(fields.get("code"), "code", minimum=400)
    if code >= 500:
        raise ValueError("code is not 4xx")

    record: dict[str, Any] = {
        "endpoint": endpoint,
        "reason": reason,
        "code": code,
    }
    if endpoint == "request_validation":
        record["errors"] = require_int(
            fields.get("errors"), "errors", minimum=0)
        record["validation_field"] = require_validation_identifier(
            fields.get("validation_field"),
            "validation_field",
            legacy_default="unknown",
        )
        record["validation_type"] = require_validation_identifier(
            fields.get("validation_type"),
            "validation_type",
            legacy_default="unknown",
        )

    has_shape = any(
        field in fields
        for field in set(SHAPE_FIELDS) | set(LEGACY_SHAPE_FIELDS)
    )
    if has_shape:
        legacy_shape = "image" in fields and "images" not in fields
        integer_fields = (
            LEGACY_INTEGER_SHAPE_FIELDS
            if legacy_shape else INTEGER_SHAPE_FIELDS
        )
        for field in integer_fields:
            record[field] = require_int(fields.get(field), field)
        if record["stream"] not in (0, 1):
            raise ValueError("invalid stream")
        if legacy_shape:
            if record["image"] not in (0, 1):
                raise ValueError("invalid image")
            record["_shape_version"] = 2
        else:
            if record["system_part_msgs"] > record["systems"]:
                raise ValueError("invalid system_part_msgs")
            image_sources = (
                record["image_data"]
                + record["image_remote"]
                + record["image_other"]
            )
            if image_sources != record["images"]:
                raise ValueError("image source counts do not match images")
            record["_shape_version"] = 3
        choice = fields.get("choice")
        if choice not in ALLOWED_CHOICES:
            raise ValueError("invalid choice")
        record["choice"] = choice
        n_value = fields.get("n")
        record["n"] = (
            None if n_value == "unset"
            else require_int(n_value, "n")
        )
    return record


def parse_detail_marker(line: str) -> dict[str, str]:
    payload = line.split(DETAIL_MARKER, 1)[1]
    fields = {
        match.group("key"): match.group("value")
        for match in FIELD_RE.finditer(payload)
    }
    if fields.get("endpoint") != "chat":
        raise ValueError("unknown detail endpoint")
    stage = fields.get("stage")
    if stage not in ALLOWED_DETAIL_STAGES:
        raise ValueError("unknown detail stage")
    exception_type = require_validation_identifier(
        fields.get("exception_type"),
        "exception_type",
        legacy_default="unknown",
    )
    return {
        "stage": stage,
        "exception_type": exception_type,
    }


def shape_key(record: dict[str, Any]) -> tuple[Any, ...] | None:
    if "messages" not in record:
        return None
    version = record["_shape_version"]
    shape_fields = LEGACY_SHAPE_FIELDS if version == 2 else SHAPE_FIELDS
    return (version, *(record[field] for field in shape_fields))


def shape_report(key: tuple[Any, ...], count: int) -> dict[str, Any]:
    version = key[0]
    shape_fields = LEGACY_SHAPE_FIELDS if version == 2 else SHAPE_FIELDS
    report = dict(zip(shape_fields, key[1:]))
    if version == 2:
        report["shape_version"] = 2
    report["count"] = count
    return report


def summarize(log_path: Path) -> tuple[dict[str, Any], bool]:
    access_codes: collections.Counter[int] = collections.Counter()
    attributed_codes: collections.Counter[int] = collections.Counter()
    endpoints: collections.Counter[str] = collections.Counter()
    reasons: collections.Counter[str] = collections.Counter()
    validation_fields: collections.Counter[str] = collections.Counter()
    validation_types: collections.Counter[str] = collections.Counter()
    failure_stages: collections.Counter[str] = collections.Counter()
    exception_types: collections.Counter[str] = collections.Counter()
    shapes: collections.Counter[tuple[Any, ...]] = collections.Counter()
    malformed = 0
    malformed_details = 0
    attributed = 0

    with log_path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            access = ACCESS_RE.search(line)
            if access:
                access_codes[int(access.group("code"))] += 1
            if DETAIL_MARKER in line:
                try:
                    detail = parse_detail_marker(line)
                except (TypeError, ValueError):
                    malformed_details += 1
                    continue
                failure_stages[detail["stage"]] += 1
                exception_types[detail["exception_type"]] += 1
                continue
            if MARKER not in line:
                continue
            try:
                record = parse_marker(line)
            except (TypeError, ValueError):
                malformed += 1
                continue
            attributed += 1
            attributed_codes[record["code"]] += 1
            endpoints[record["endpoint"]] += 1
            reasons[record["reason"]] += 1
            if record["endpoint"] == "request_validation":
                validation_fields[record["validation_field"]] += 1
                validation_types[record["validation_type"]] += 1
            key = shape_key(record)
            if key is not None:
                shapes[key] += 1

    access_total = sum(access_codes.values())
    complete = (malformed == 0 and malformed_details == 0
                and attributed_codes == access_codes)
    classified = reasons["unclassified_chat_error"] == 0
    qualified = complete and classified
    report = {
        "schema": REPORT_SCHEMA,
        "version": REPORT_VERSION,
        "complete": complete,
        "classified": classified,
        "qualified": qualified,
        "chat_4xx_access_count": access_total,
        "attributed_count": attributed,
        "attribution_delta": access_total - attributed,
        "malformed_marker_count": malformed,
        "malformed_detail_marker_count": malformed_details,
        "by_access_code": {
            str(key): value for key, value in sorted(access_codes.items())
        },
        "by_attributed_code": {
            str(key): value for key, value in sorted(
                attributed_codes.items())
        },
        "by_endpoint": dict(sorted(endpoints.items())),
        "by_reason": dict(sorted(reasons.items())),
        "by_validation_field": dict(sorted(validation_fields.items())),
        "by_validation_type": dict(sorted(validation_types.items())),
        "by_failure_stage": dict(sorted(failure_stages.items())),
        "by_exception_type": dict(sorted(exception_types.items())),
        "request_shapes": [
            shape_report(key, count)
            for key, count in sorted(
                shapes.items(),
                key=lambda item: tuple(
                    "" if value is None else str(value)
                    for value in item[0]
                ),
            )
        ],
        "privacy": {
            "contains_raw_log_lines": False,
            "contains_request_content": False,
            "contains_response_content": False,
            "contains_tool_schema": False,
            "contains_multimodal_url_or_bytes": False,
            "contains_validation_error_message": False,
            "contains_validation_input_value": False,
        },
    }
    return report, qualified


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize privacy-safe Chat Completions 4xx attribution."
    )
    parser.add_argument("log", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    report, qualified = summarize(args.log)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
