#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SOURCE_SCHEMA = "bi100-multimodal-prefix-isolation-v1"
SCHEMA = "bi100-multimodal-prefix-isolation-qualification-v2"
REQUEST_NAMES = (
    "same_image_cold",
    "same_image_warm",
    "different_image",
)


def _valid_request(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    required = {
        "cached_tokens",
        "completion_tokens",
        "elapsed_s",
        "expected_color_observed",
        "finish_reason",
        "prompt_tokens",
        "response_sha256",
    }
    if set(value) != required:
        return False
    integers = (
        value["cached_tokens"],
        value["completion_tokens"],
        value["prompt_tokens"],
    )
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0
           for item in integers):
        return False
    elapsed = value["elapsed_s"]
    if (not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool)
            or not math.isfinite(elapsed) or elapsed <= 0):
        return False
    digest = value["response_sha256"]
    return (
        value["expected_color_observed"] is True
        and isinstance(value["finish_reason"], str)
        and isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )


def qualify(source: Any) -> dict[str, Any]:
    reasons: list[str] = []
    if (not isinstance(source, dict)
            or source.get("schema") != SOURCE_SCHEMA
            or source.get("version") != 1):
        reasons.append("source schema/version is invalid")
        source = {}
    requests = source.get("requests")
    if not isinstance(requests, dict):
        reasons.append("source requests are missing")
        requests = {}
    for name in REQUEST_NAMES:
        if not _valid_request(requests.get(name)):
            reasons.append(f"request {name} is invalid")

    cold = requests.get("same_image_cold", {})
    warm = requests.get("same_image_warm", {})
    different = requests.get("different_image", {})
    checks = {
        "cold_has_no_hit": cold.get("cached_tokens") == 0,
        "different_image_isolated": different.get("cached_tokens") == 0,
        "red_cold_warm_exact": all(
            cold.get(field) == warm.get(field)
            for field in (
                "completion_tokens",
                "finish_reason",
                "prompt_tokens",
                "response_sha256",
            )
        ),
        "same_image_hits": (
            isinstance(warm.get("cached_tokens"), int)
            and warm.get("cached_tokens", 0) > 0
        ),
        "semantic_colors_observed": all(
            requests.get(name, {}).get("expected_color_observed") is True
            for name in REQUEST_NAMES
        ),
    }
    reasons.extend(name for name, passed in checks.items() if not passed)
    canonical = json.dumps(
        source, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "checks": checks,
        "different_image_prompt_token_delta": (
            different.get("prompt_tokens", 0) - cold.get("prompt_tokens", 0)
        ),
        "qualified": not reasons,
        "reasons": reasons,
        "schema": SCHEMA,
        "source_run_id": source.get("run_id"),
        "source_schema": source.get("schema"),
        "source_sha256": hashlib.sha256(canonical).hexdigest(),
        "version": 2,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = qualify(json.loads(args.source.read_text(encoding="utf-8")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
