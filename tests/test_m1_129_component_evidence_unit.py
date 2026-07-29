from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
import unittest


ROOT = Path(__file__).resolve().parents[1]
M1_128 = (
    ROOT
    / "docs/experiments/evidence"
    / "M1_128_HALF_QK_COMPONENT_20260729"
    / "result.json"
)
M1_129 = (
    ROOT
    / "docs/experiments/evidence"
    / "M1_129_HALF_QK_DEFAULT_GEMMEX_20260729"
    / "result.json"
)


class M1129ComponentEvidenceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.previous = json.loads(M1_128.read_text(encoding="utf-8"))
        cls.value = json.loads(M1_129.read_text(encoding="utf-8"))

    def test_exact_source_and_binary_identities(self) -> None:
        self.assertEqual(
            self.value["source_revision"],
            "de9d91e32eb1feff6ec176d7084a80095fb98ef2",
        )
        self.assertEqual(
            self.value["candidate_extension"]["sha256"],
            "62fc3af12eb863801abfb6e08336e9a04f5c377ba7d03511ad0a2412b0f37f82",
        )
        self.assertEqual(
            self.value["control_extension"]["sha256"],
            self.previous["control_extension"]["sha256"],
        )

    def test_default_algorithm_repeats_m1_128_numerical_result(self) -> None:
        self.assertEqual(len(self.value["rows"]), 4)
        self.assertEqual(len(self.previous["rows"]), 4)
        for current, previous in zip(
                self.value["rows"], self.previous["rows"]):
            self.assertEqual(current["case"], previous["case"])
            for field in (
                "output_relative_l2",
                "lse_relative_l2",
                "output_max_abs",
            ):
                self.assertEqual(current[field], previous[field])

    def test_performance_passes_but_numerical_gate_fails(self) -> None:
        contract = self.value["fixed_contract"]
        speedups = []
        for row in self.value["rows"]:
            speedups.append(row["control_over_candidate_speedup"])
            self.assertTrue(all(
                math.isfinite(float(row[field]))
                for field in (
                    "output_relative_l2",
                    "lse_relative_l2",
                    "output_max_abs",
                )
            ))
            self.assertGreater(
                row["output_relative_l2"],
                contract["maximum_relative_l2"],
            )
            self.assertLessEqual(
                row["lse_relative_l2"],
                contract["maximum_relative_l2"],
            )
            self.assertLessEqual(
                row["output_max_abs"],
                contract["maximum_output_abs"],
            )
        self.assertGreaterEqual(
            statistics.median(speedups),
            contract["minimum_median_control_over_candidate_speedup"],
        )
        self.assertAlmostEqual(
            statistics.median(speedups),
            self.value["aggregate"][
                "median_control_over_candidate_speedup"],
        )

    def test_route_is_closed_and_lifecycle_is_clean(self) -> None:
        adjudication = self.value["adjudication"]
        self.assertFalse(adjudication["qualified"])
        self.assertTrue(
            adjudication["half_input_full_pipeline_route_closed"])
        self.assertFalse(
            adjudication["additional_algorithm_scan_authorized"])
        self.assertFalse(
            adjudication["tp4_service_experiment_authorized"])
        self.assertFalse(adjudication["main_or_yaml_change_authorized"])
        self.assertTrue(all(
            returncode == 0
            for returncode in self.value[
                "lifecycle_returncodes"].values()
        ))

    def test_evidence_contains_no_sensitive_payloads(self) -> None:
        self.assertTrue(all(
            value is False for value in self.value["privacy"].values()
        ))
        serialized = M1_129.read_text(encoding="utf-8").lower()
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
