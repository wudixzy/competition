#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any


REPORT_SCHEMA = "bi100-api-4xx-attribution-v1"
REPORT_VERSION = 1
MARKER = "[BI100 4XX]"
ACCESS_RE = re.compile(
    r'"POST /v1/chat/completions HTTP/1\.[01]" (?P<code>4\d\d)\b'
)
FIELD_RE = re.compile(r"(?P<key>[a-z_]+)=(?P<value>[^\s]+)")
ALLOWED_ENDPOINTS = {"chat", "request_validation"}
ALLOWED_REASONS = {
    "empty_messages",
    "n_exceeds_max_num_seqs",
    "unsupported_tool_choice_required",
    "tool_parser_unavailable",
    "unclassified_chat_error",
    "request_validation_messages",
    "request_validation_tools",
    "request_validation_response_format",
    "request_validation_streaming",
    "request_validation_generation",
    "request_validation_sampling",
    "request_validation_model",
    "request_validation_other",
    "request_validation_unknown",
}
SHAPE_FIELDS = ("messages", "systems", "tools", "image", "stream", "n")


def require_int(value: str | None, field: str, *, minimum: int = 0) -> int:
    if value is None:
        raise ValueError(f"missing {field}")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"invalid {field}")
    return parsed


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
        record["errors"] = require_int(fields.get("errors"), "errors",
                                       minimum=1)

    has_shape = any(field in fields for field in SHAPE_FIELDS)
    if has_shape:
        for field in SHAPE_FIELDS:
            value = fields.get(field)
            if field == "n" and value == "unset":
                record[field] = None
            else:
                record[field] = require_int(value, field)
        for field in ("image", "stream"):
            if record[field] not in (0, 1):
                raise ValueError(f"invalid {field}")
    return record


def shape_key(record: dict[str, Any]) -> tuple[Any, ...] | None:
    if "messages" not in record:
        return None
    return tuple(record[field] for field in SHAPE_FIELDS)


def summarize(log_path: Path) -> tuple[dict[str, Any], bool]:
    access_codes: collections.Counter[int] = collections.Counter()
    attributed_codes: collections.Counter[int] = collections.Counter()
    endpoints: collections.Counter[str] = collections.Counter()
    reasons: collections.Counter[str] = collections.Counter()
    shapes: collections.Counter[tuple[Any, ...]] = collections.Counter()
    malformed = 0
    attributed = 0

    with log_path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            access = ACCESS_RE.search(line)
            if access:
                access_codes[int(access.group("code"))] += 1
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
            key = shape_key(record)
            if key is not None:
                shapes[key] += 1

    access_total = sum(access_codes.values())
    complete = malformed == 0 and attributed_codes == access_codes
    report = {
        "schema": REPORT_SCHEMA,
        "version": REPORT_VERSION,
        "complete": complete,
        "chat_4xx_access_count": access_total,
        "attributed_count": attributed,
        "attribution_delta": access_total - attributed,
        "malformed_marker_count": malformed,
        "by_access_code": {
            str(key): value for key, value in sorted(access_codes.items())
        },
        "by_attributed_code": {
            str(key): value for key, value in sorted(attributed_codes.items())
        },
        "by_endpoint": dict(sorted(endpoints.items())),
        "by_reason": dict(sorted(reasons.items())),
        "request_shapes": [
            {
                **dict(zip(SHAPE_FIELDS, key)),
                "count": count,
            }
            for key, count in sorted(
                shapes.items(),
                key=lambda item: tuple(
                    -1 if value is None else value for value in item[0]
                ),
            )
        ],
        "privacy": {
            "contains_raw_log_lines": False,
            "contains_request_content": False,
            "contains_response_content": False,
        },
    }
    return report, complete


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize privacy-safe Chat Completions 4xx attribution."
    )
    parser.add_argument("log", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    report, complete = summarize(args.log)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
