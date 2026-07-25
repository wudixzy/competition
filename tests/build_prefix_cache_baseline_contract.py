#!/usr/bin/env python3
"""Bind one 881-request trace to runtime, workload, and observed metrics."""

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
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument("--metrics-source", type=Path, required=True)
    parser.add_argument("--metrics-transformation", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--score-kind",
        choices=("local_881_proxy", "official_platform"),
        required=True,
    )
    parser.add_argument("--aggregation", required=True)
    parser.add_argument("--successful-requests", type=int, required=True)
    parser.add_argument("--error-requests", type=int, required=True)
    parser.add_argument("--output-tps-p10", type=float, required=True)
    parser.add_argument("--input-tps", type=float, required=True)
    parser.add_argument("--cache-tps", type=float, required=True)
    parser.add_argument("--ttft-p90-s", type=float, required=True)
    parser.add_argument("--cache-hit-rate", type=float, required=True)
    parser.add_argument(
        "--attest-same-run",
        action="store_true",
        help="Assert that trace, runtime contract, and metrics are one run.",
    )
    parser.add_argument(
        "--attest-exact-request-order",
        action="store_true",
        help="Assert that metrics cover the exact ordered trace requests.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not args.attest_same_run:
        parser.error("--attest-same-run is required")
    if not args.attest_exact_request_order:
        parser.error("--attest-exact-request-order is required")

    records = trace_analyzer.read([str(path) for path in args.logs])
    trace = baseline_contract.trace_identity(records, args.logs)
    runtime_value = json.loads(
        args.runtime_contract.read_text(encoding="utf-8"))
    runtime_sha = baseline_contract.validate_runtime_contract(runtime_value)
    workload_value = json.loads(
        args.workload_manifest.read_text(encoding="utf-8"))
    workload_sha = baseline_contract.validate_workload_manifest(
        workload_value, expected_trace=trace)

    attempted = baseline_contract.EXPECTED_REQUESTS
    success_rate = (
        args.successful_requests / attempted if attempted else 0.0)
    metrics = {
        "score_kind": args.score_kind,
        "aggregation": args.aggregation,
        "attempted_requests": attempted,
        "successful_requests": args.successful_requests,
        "error_requests": args.error_requests,
        "output_tps_p10": args.output_tps_p10,
        "input_tps": args.input_tps,
        "cache_tps": args.cache_tps,
        "ttft_p90_s": args.ttft_p90_s,
        "cache_hit_rate": args.cache_hit_rate,
        "success_rate": success_rate,
        "weighted_score": 0.0,
        "formula": baseline_contract.SCORE_FORMULA,
    }
    metrics["weighted_score"] = baseline_contract.weighted_score(metrics)
    baseline_contract.validate_metrics(metrics)

    contract = {
        "schema": baseline_contract.BASELINE_SCHEMA,
        "version": 1,
        "run_id": args.run_id,
        "runtime_contract": {
            "sha256": runtime_sha,
            "file_sha256": baseline_contract.sha256_file(
                args.runtime_contract),
            "value": runtime_value,
        },
        "workload_manifest": {
            "sha256": workload_sha,
            "file_sha256": baseline_contract.sha256_file(
                args.workload_manifest),
            "value": workload_value,
        },
        "trace": trace,
        "metrics": metrics,
        "metrics_source": baseline_contract.artifact(args.metrics_source),
        "metrics_transformation": args.metrics_transformation,
        "attestation": {
            "trace_metrics_same_service_run_asserted": True,
            "metrics_cover_exact_trace_request_order_asserted": True,
            "contains_raw_requests_or_outputs": False,
            "qualification_scope": "offline_cache_phase_gate_only",
        },
    }
    contract_sha = baseline_contract.validate_baseline_contract(
        contract, expected_trace=trace)
    _atomic_write(args.out, contract)
    print(json.dumps({
        "baseline_contract_sha256": contract_sha,
        "out": str(args.out),
        "qualified": True,
        "run_id": args.run_id,
        "runtime_contract_sha256": runtime_sha,
        "workload_manifest_sha256": workload_sha,
        "trace_session_sha256": trace["session_sha256"],
        "weighted_score": metrics["weighted_score"],
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
