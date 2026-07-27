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
SCHEMA = "qwen36-diagnostic-quality-contract-v2"
CASE_IDS = (
    "top_p_0",
    "top_p_1_1",
    "n_1",
    "n_2",
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
    allow_bare_engine_n2_skip = False


def _single_positive_int(values: Any) -> int | None:
    if (not isinstance(values, list) or len(values) != 1
            or type(values[0]) is not int or values[0] <= 0):
        return None
    return values[0]


def _canonical_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _n_cross_case_contract(results: list[Json]) -> Json:
    cases = {result.get("id"): result for result in results}
    n1 = cases.get("n_1") or {}
    n2 = cases.get("n_2") or {}
    n1_observation = n1.get("observation") if n1.get("ok") else None
    n2_observation = n2.get("observation") if n2.get("ok") else None

    observations_available = (
        isinstance(n1_observation, dict)
        and isinstance(n2_observation, dict)
    )
    n1_prompt = _single_positive_int(
        n1_observation.get("prompt_tokens")
        if isinstance(n1_observation, dict) else None)
    n2_prompt = _single_positive_int(
        n2_observation.get("prompt_tokens")
        if isinstance(n2_observation, dict) else None)
    prompt_counted_once = (
        observations_available
        and n1_prompt is not None
        and n1_prompt == n2_prompt
    )

    n1_completion = _single_positive_int(
        n1_observation.get("completion_tokens")
        if isinstance(n1_observation, dict) else None)
    n2_completion = _single_positive_int(
        n2_observation.get("completion_tokens")
        if isinstance(n2_observation, dict) else None)
    completion_summed = (
        observations_available
        and n1_completion is not None
        and n2_completion == 2 * n1_completion
    )

    n1_facts = (
        n1_observation.get("facts")
        if isinstance(n1_observation, dict) else None
    )
    n2_facts = (
        n2_observation.get("facts")
        if isinstance(n2_observation, dict) else None
    )
    n1_digest = (
        n1_facts.get("choice_output_sha256")
        if isinstance(n1_facts, dict) else None
    )
    n2_digest = (
        n2_facts.get("choice_output_sha256")
        if isinstance(n2_facts, dict) else None
    )
    choice_output_exact = (
        _canonical_sha256(n1_digest)
        and _canonical_sha256(n2_digest)
        and n1_digest == n2_digest
    )
    individual_facts_valid = (
        isinstance(n1_facts, dict)
        and isinstance(n2_facts, dict)
        and type(n1_facts.get("n")) is int
        and n1_facts["n"] == 1
        and type(n2_facts.get("n")) is int
        and n2_facts["n"] == 2
        and all(
            facts.get(name) is True
            for facts in (n1_facts, n2_facts)
            for name in (
                "choice_indices_exact",
                "usage_accounted",
                "deterministic_choices_exact",
            )
        )
    )
    checks = {
        "observations_available": observations_available,
        "individual_facts_valid": individual_facts_valid,
        "prompt_counted_once": prompt_counted_once,
        "completion_summed": completion_summed,
        "choice_output_exact": choice_output_exact,
    }
    reasons = {
        "observations_available": "n=1/n=2 observations unavailable",
        "individual_facts_valid": "n=1/n=2 individual facts are invalid",
        "prompt_counted_once": "n=1/n=2 prompt usage differs",
        "completion_summed": "n=2 completion usage is not twice n=1",
        "choice_output_exact": "n=1/n=2 deterministic output digest differs",
    }
    return {
        "qualified": all(checks.values()),
        "checks": checks,
        "reasons": [
            reasons[name] for name, passed in checks.items() if not passed
        ],
    }


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
    n_contract = _n_cross_case_contract(results)
    qualified = (
        all(result["ok"] for result in results)
        and final_health
        and n_contract["qualified"]
    )
    return {
        "schema": SCHEMA,
        "version": 2,
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
        "n_cross_case_contract": n_contract,
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
        "n_cross_case_contract": report["n_cross_case_contract"],
        "out": str(args.json_out),
    }, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
