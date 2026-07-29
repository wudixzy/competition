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
    / "M1_128_HALF_QK_COMPONENT_20260729"
    / "result.json"
)


class M1128ComponentEvidenceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.value = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_exact_source_and_binary_identities(self) -> None:
        self.assertEqual(
            self.value["source_revision"],
            "2135cb276ffa678c6474aacbcc669e2806b2391b",
        )
        self.assertEqual(
            self.value["control_extension"]["sha256"],
            "ad4ea7707bb2f2bfe04e07a7ad5fd58a647232be70a3056937a0d738c8254bff",
        )
        self.assertEqual(
            self.value["candidate_extension"]["sha256"],
            "acc89f2cbadb99dbe73dbb0af397ebfe9885e55e6505fa361a798bab92b345cd",
        )

    def test_all_shapes_improve_but_fail_relative_l2(self) -> None:
        contract = self.value["fixed_contract"]
        rows = self.value["rows"]
        self.assertEqual(
            [row["case"] for row in rows],
            contract["cases"],
        )
        speedups = []
        for row in rows:
            speedups.append(row["control_over_candidate_speedup"])
            self.assertGreater(row["control_over_candidate_speedup"], 1.0)
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
        self.assertAlmostEqual(
            statistics.median(speedups),
            self.value["aggregate"][
                "median_control_over_candidate_speedup"],
        )
        self.assertGreaterEqual(
            statistics.median(speedups),
            contract["minimum_median_control_over_candidate_speedup"],
        )

    def test_failure_blocks_tp4_and_default_changes(self) -> None:
        adjudication = self.value["adjudication"]
        self.assertFalse(adjudication["qualified"])
        self.assertTrue(adjudication["performance_gate_passed"])
        self.assertFalse(adjudication["relative_l2_gate_passed"])
        self.assertTrue(
            adjudication["second_accuracy_path_experiment_authorized"])
        self.assertFalse(
            adjudication["tp4_service_experiment_authorized"])
        self.assertFalse(adjudication["main_or_yaml_change_authorized"])

    def test_lifecycle_passed_and_evidence_is_private(self) -> None:
        self.assertTrue(all(
            returncode == 0
            for returncode in self.value[
                "lifecycle_returncodes"].values()
        ))
        self.assertTrue(
            self.value["adjudication"]["outer_session_reaped"])
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
