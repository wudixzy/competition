from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "docs/experiments/evidence"
    / "M1_130_CUBLAS_CONCURRENCY_20260729"
    / "result.json"
)


class M1130CublasConcurrencyEvidenceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.value = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_exact_source_and_fixed_contract(self) -> None:
        self.assertEqual(
            self.value["source_revision"],
            "73da58f0ab253407fde7e460d783845c226f38fd",
        )
        contract = self.value["fixed_contract"]
        self.assertEqual(contract["query_tokens"], [8176, 5616])
        self.assertEqual(contract["dtype"], "float32")
        self.assertEqual(contract["head_dim"], 256)
        self.assertEqual(contract["key_tokens"], 512)
        self.assertEqual(contract["trials"], 20)

    def test_outputs_are_exact_but_both_shapes_regress(self) -> None:
        contract = self.value["fixed_contract"]
        speedups = []
        self.assertEqual(len(self.value["rows"]), 2)
        for row in self.value["rows"]:
            self.assertTrue(row["finite"])
            for field in (
                "qk_relative_l2",
                "qk_max_abs",
                "pv_relative_l2",
                "pv_max_abs",
            ):
                self.assertTrue(math.isfinite(float(row[field])))
                self.assertEqual(row[field], 0.0)
            speedup = row["sequential_over_concurrent_speedup"]
            speedups.append(speedup)
            self.assertLess(speedup, contract["minimum_cell_speedup"])
            self.assertFalse(row["cell_qualified"])
        self.assertAlmostEqual(
            statistics.median(speedups),
            self.value["aggregate"]["median_speedup"],
        )
        self.assertLess(
            self.value["aggregate"]["median_speedup"],
            contract["minimum_median_speedup"],
        )

    def test_lifecycle_passed_and_overall_rejects_performance(self) -> None:
        returncodes = self.value["lifecycle_returncodes"]
        self.assertEqual(returncodes["overall"], 1)
        for name, returncode in returncodes.items():
            if name != "overall":
                self.assertEqual(returncode, 0, name)
        checks = self.value["runtime_checks"]
        self.assertTrue(checks["fatal_scan_empty"])
        self.assertTrue(checks["timeout_scan_empty"])
        self.assertTrue(
            checks["four_gpu_preflight_comparison_qualified"])

    def test_route_is_closed_without_promotion(self) -> None:
        decision = self.value["adjudication"]
        self.assertFalse(decision["qualified"])
        self.assertTrue(decision["stream_overlap_route_closed"])
        for field in (
            "double_buffer_pipeline_authorized",
            "additional_stream_or_threshold_scan_authorized",
            "tp4_service_experiment_authorized",
            "runtime_overlay_change_authorized",
            "main_or_yaml_change_authorized",
        ):
            self.assertFalse(decision[field])

    def test_evidence_contains_no_sensitive_payloads(self) -> None:
        self.assertTrue(all(
            value is False for value in self.value["privacy"].values()
        ))
        serialized = EVIDENCE.read_text(encoding="utf-8").lower()
        for marker in (
            "session_token",
            "request_contract_sha256",
            "semantic_output_sha256",
            "begin openssh private key",
            "modelhub_access_token",
        ):
            self.assertNotIn(marker, serialized)


if __name__ == "__main__":
    unittest.main()
