import copy
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_prefix_cache_trace as analyzer
import prefix_cache_baseline_contract as contract
from tests.test_analyze_prefix_cache_trace_unit import (
    attested_baseline,
    record,
)


def fixture(root: pathlib.Path):
    raw_records = [
        record([1, 2], capacity=4, request_id=index, ordinal=index + 1)
        for index in range(881)
    ]
    trace_path = root / "trace.log"
    trace_path.write_text("\n".join(
        analyzer.MARKER + json.dumps(item) for item in raw_records) + "\n")
    records = analyzer.read([str(trace_path)])
    value = attested_baseline(records, trace_path)
    identity = contract.trace_identity(records, [trace_path])
    return trace_path, records, identity, value


class PrefixCacheBaselineContractTest(unittest.TestCase):
    def test_valid_contract_binds_runtime_workload_trace_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, identity, value = fixture(pathlib.Path(directory))
            digest = contract.validate_baseline_contract(
                value, expected_trace=identity)

            self.assertEqual(len(digest), 64)
            self.assertEqual(
                value["runtime_contract"]["value"]["gpu_count"], 4)
            self.assertEqual(
                value["runtime_contract"]["value"]["tensor_parallel_size"], 4)
            self.assertEqual(
                value["workload_manifest"]["value"]["request_count"], 881)
            self.assertEqual(
                value["trace"]["request_order_sha256"],
                value["workload_manifest"]["value"][
                    "request_order_sha256"],
            )

    def test_candidate_runtime_cannot_be_used_as_the_control_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, identity, value = fixture(pathlib.Path(directory))
            value["runtime_contract"]["value"]["environment"][
                "BI100_GDN_CACHE_POLICY"] = "admission64"
            value["runtime_contract"]["sha256"] = (
                contract.runtime_contract.sha256_json(
                    value["runtime_contract"]["value"]))

            with self.assertRaisesRegex(
                    contract.BaselineContractError,
                    "requires BI100_GDN_CACHE_POLICY=fine32"):
                contract.validate_baseline_contract(
                    value, expected_trace=identity)

    def test_runtime_source_revision_drift_breaks_wrapper_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, identity, value = fixture(pathlib.Path(directory))
            value["runtime_contract"]["value"]["source_revision"] = "f" * 40

            with self.assertRaisesRegex(
                    contract.BaselineContractError,
                    "canonical SHA-256 is inconsistent"):
                contract.validate_baseline_contract(
                    value, expected_trace=identity)

    def test_workload_request_order_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, identity, value = fixture(pathlib.Path(directory))
            workload = value["workload_manifest"]["value"]
            workload["request_order_sha256"] = "f" * 64
            value["workload_manifest"]["sha256"] = contract.sha256_json(
                workload)

            with self.assertRaisesRegex(
                    contract.BaselineContractError,
                    "request order does not match"):
                contract.validate_baseline_contract(
                    value, expected_trace=identity)

    def test_trace_log_artifact_order_is_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _, records, identity, value = fixture(root)
            second = root / "second.log"
            second.write_text("not a cache trace\n")
            two_log_identity = contract.trace_identity(
                records, [root / "trace.log", second])
            value["trace"] = copy.deepcopy(two_log_identity)

            reversed_identity = contract.trace_identity(
                records, [second, root / "trace.log"])
            with self.assertRaisesRegex(
                    contract.BaselineContractError,
                    "does not match input logs"):
                contract.validate_baseline_contract(
                    value, expected_trace=reversed_identity)
            self.assertNotEqual(identity["logs"], two_log_identity["logs"])

    def test_weighted_score_is_recomputed(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, identity, value = fixture(pathlib.Path(directory))
            value["metrics"]["weighted_score"] += 10.0

            with self.assertRaisesRegex(
                    contract.BaselineContractError,
                    "weighted score is inconsistent"):
                contract.validate_baseline_contract(
                    value, expected_trace=identity)

    def test_request_counts_and_success_rate_are_consistent(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, identity, value = fixture(pathlib.Path(directory))
            value["metrics"]["successful_requests"] = 880

            with self.assertRaisesRegex(
                    contract.BaselineContractError,
                    "request counts are inconsistent"):
                contract.validate_baseline_contract(
                    value, expected_trace=identity)

    def test_attestation_and_credential_markers_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, identity, value = fixture(pathlib.Path(directory))
            value["attestation"][
                "trace_metrics_same_service_run_asserted"] = False
            with self.assertRaisesRegex(
                    contract.BaselineContractError,
                    "same-run attestation is missing"):
                contract.validate_baseline_contract(
                    value, expected_trace=identity)

            value = fixture(pathlib.Path(directory))[3]
            value["run_id"] = "github_pat_not-a-real-token"
            with self.assertRaisesRegex(
                    contract.BaselineContractError,
                    "credential marker"):
                contract.validate_baseline_contract(
                    value, expected_trace=value["trace"])

    def test_builders_round_trip_without_copying_restricted_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            trace_path, records, identity, value = fixture(root)
            runtime_path = root / "runtime-contract.json"
            runtime_path.write_text(json.dumps(
                value["runtime_contract"]["value"], sort_keys=True))
            metrics_source = root / "metrics-source.json"
            metrics_source.write_text(json.dumps({
                "kind": "unit aggregate without requests or outputs",
            }))
            workload_path = root / "workload-manifest.json"
            baseline_path = root / "baseline-contract.json"

            workload_command = [
                sys.executable,
                str(ROOT / "tests"
                    / "build_prefix_cache_workload_manifest.py"),
                str(trace_path),
                "--name", "unit official workload",
                "--author-or-org", "unit operator",
                "--license", "restricted",
                "--revision", "unit-run-1",
                "--captured-at-utc", "2026-07-25T00:00:00Z",
                "--split", "all",
                "--selection-rule", "all requests in fixed order",
                "--transformation", "privacy-safe cache trace v4",
                "--out", str(workload_path),
            ]
            workload_process = subprocess.run(
                workload_command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(
                workload_process.returncode, 0, workload_process.stderr)

            baseline_command = [
                sys.executable,
                str(ROOT / "tests"
                    / "build_prefix_cache_baseline_contract.py"),
                str(trace_path),
                "--runtime-contract", str(runtime_path),
                "--workload-manifest", str(workload_path),
                "--metrics-source", str(metrics_source),
                "--metrics-transformation", "exact unit field mapping",
                "--run-id", "unit-run-1",
                "--score-kind", "local_881_proxy",
                "--aggregation", "fixed-order sequential aggregate",
                "--successful-requests", "881",
                "--error-requests", "0",
                "--output-tps-p10", "21",
                "--input-tps", "2800",
                "--cache-tps", "100",
                "--ttft-p90-s", "4",
                "--cache-hit-rate", "0.5",
                "--attest-same-run",
                "--attest-exact-request-order",
                "--out", str(baseline_path),
            ]
            baseline_process = subprocess.run(
                baseline_command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(
                baseline_process.returncode, 0, baseline_process.stderr)

            built, digest = contract.load_baseline_contract(
                baseline_path, expected_trace=identity)
            self.assertEqual(len(digest), 64)
            self.assertFalse(
                built["attestation"]["contains_raw_requests_or_outputs"])
            self.assertNotIn("messages", json.dumps(built).lower())
            self.assertEqual(
                built["metrics_source"], contract.artifact(metrics_source))
            self.assertEqual(len(records), 881)


if __name__ == "__main__":
    unittest.main()
