#!/usr/bin/env python3
"""Build a privacy-safe identity manifest for one restricted 881-request run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_prefix_cache_trace as trace_analyzer  # noqa: E402
import prefix_cache_baseline_contract as baseline_contract  # noqa: E402


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--author-or-org", required=True)
    parser.add_argument("--source-url")
    parser.add_argument("--license", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--captured-at-utc", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--selection-rule", required=True)
    parser.add_argument("--transformation", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    records = trace_analyzer.read([str(path) for path in args.logs])
    trace = baseline_contract.trace_identity(records, args.logs)
    manifest = {
        "schema": baseline_contract.WORKLOAD_SCHEMA,
        "version": 1,
        "workload_kind": "restricted_official_881",
        "name": args.name,
        "author_or_org": args.author_or_org,
        "source_url": args.source_url,
        "license": args.license,
        "revision": args.revision,
        "captured_at_utc": args.captured_at_utc,
        "split": args.split,
        "request_count": baseline_contract.EXPECTED_REQUESTS,
        "request_order_sha256": trace["request_order_sha256"],
        "source_artifact_sha256": trace["records_sha256"],
        "source_artifact_kind": "privacy_safe_cache_trace_v4_records",
        "selection_rule": args.selection_rule,
        "transformation": args.transformation,
        "redistribution_allowed": False,
        "contains_restricted_evaluation_data": True,
        "snapshot_redistributed": False,
    }
    digest = baseline_contract.validate_workload_manifest(
        manifest, expected_trace=trace)
    _atomic_write(args.out, manifest)
    print(json.dumps({
        "manifest_sha256": digest,
        "out": str(args.out),
        "qualified": True,
        "request_count": trace["request_count"],
        "request_order_sha256": trace["request_order_sha256"],
        "trace_records_sha256": trace["records_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        baseline_contract.BaselineContractError,
        ValueError,
        OSError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
