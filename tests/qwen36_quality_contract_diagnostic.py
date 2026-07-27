#!/usr/bin/env python3
"""Run model-capability-independent API boundary checks on a diagnostic model."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import quality_gate_api as quality


Json = dict[str, Any]
SCHEMA = "qwen36-diagnostic-quality-contract-v1"
CASE_IDS = (
    "top_p_0",
    "top_p_1_1",
    "max_tokens_minus_1",
    "max_tokens_over_context",
    "empty_request_body",
    "message_missing_role",
    "message_missing_content",
    "empty_messages",
)


class DiagnosticConfig:
    model = "llm"
    max_model_len = 262144
    truncation_tokens = 32768
    endpoint_mode = "direct"
    allow_bare_engine_n2_skip = True


def _atomic_write(path: Path, value: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=True, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def run_gate(
    base: str,
    *,
    client: quality.Client | None = None,
    handlers: dict[str, quality.Handler] | None = None,
) -> Json:
    manifest, manifest_sha = quality._load_manifest(quality.DEFAULT_MANIFEST)
    cases_by_id = {case["id"]: case for case in manifest["cases"]}
    if set(CASE_IDS) - set(cases_by_id):
        raise RuntimeError("diagnostic cases are absent from the frozen manifest")
    active_client = client if client is not None else quality.Client(base)
    active_handlers = quality.HANDLERS if handlers is None else handlers
    config = DiagnosticConfig()
    results = []

    for case_id in CASE_IDS:
        started = time.perf_counter()
        metadata = cases_by_id[case_id]
        try:
            observation = active_handlers[case_id](active_client, config)
            skip_reason = observation.pop("_skip_reason", "")
            if skip_reason:
                raise quality.CaseFailure(
                    f"diagnostic boundary case unexpectedly skipped: {case_id}")
        except quality.CaseFailure as error:
            results.append({
                **metadata,
                "ok": False,
                "elapsed_s": time.perf_counter() - started,
                "error_code": str(error),
                "observation": None,
            })
        except Exception as error:
            results.append({
                **metadata,
                "ok": False,
                "elapsed_s": time.perf_counter() - started,
                "error_code": f"unexpected {type(error).__name__}",
                "observation": None,
            })
        else:
            results.append({
                **metadata,
                "ok": True,
                "elapsed_s": time.perf_counter() - started,
                "error_code": "",
                "observation": observation,
            })

    final_health = False
    try:
        active_client.models(config.model)
    except quality.CaseFailure:
        pass
    else:
        final_health = True
    qualified = all(result["ok"] for result in results) and final_health
    return {
        "schema": SCHEMA,
        "version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_sha256": hashlib.sha256(base.encode("utf-8")).hexdigest(),
        "manifest": {
            "name": quality.DEFAULT_MANIFEST.name,
            "sha256": manifest_sha,
            "source_sha256": manifest["source"]["sha256"],
        },
        "qualified": qualified,
        "case_count": len(results),
        "passed": sum(result["ok"] for result in results),
        "failed": sum(not result["ok"] for result in results),
        "final_health": final_health,
        "cases": results,
        "scope": {
            "model_capability_evaluated": False,
            "semantic_quality_evaluated": False,
            "full_model_evaluated": False,
            "tp4_performance_evaluated": False,
            "production_promotion_authorized": False,
        },
        "privacy": {
            "contains_raw_requests": False,
            "contains_raw_model_outputs": False,
            "contains_credentials": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_gate(args.base)
    _atomic_write(args.json_out, report)
    print(json.dumps({
        "qualified": report["qualified"],
        "case_count": report["case_count"],
        "passed": report["passed"],
        "failed": report["failed"],
        "final_health": report["final_health"],
        "out": str(args.json_out),
    }, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
