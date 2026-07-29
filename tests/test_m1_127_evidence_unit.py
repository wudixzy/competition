from __future__ import annotations

import json
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "docs/experiments/evidence"
    / "M1_127_HALF_INPUT_QK_QUALIFIED_20260729"
    / "result.json"
)


class M1127EvidenceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.value = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_identity_and_fixed_contract(self) -> None:
        self.assertEqual(
            self.value["source_revision"],
            "da640c7a93e9a5ea78775e0fc34dbd026e4e3a30",
        )
        contract = self.value["fixed_contract"]
        self.assertEqual(contract["heads"], 4)
        self.assertEqual(contract["key_tokens"], 512)
        self.assertEqual(contract["head_dim"], 256)
        self.assertEqual(contract["seed"], 20260729)
        self.assertEqual(contract["magnitudes"], [0.5, 1.0, 2.0])
        self.assertEqual(contract["error_reduction_device"], "cpu_fp64")

    def test_both_fixed_shapes_clear_numerical_and_speed_gates(self) -> None:
        contract = self.value["fixed_contract"]
        expected = (("q8176", 0, 8176), ("q5616", 1, 5616))
        self.assertEqual(len(self.value["cases"]), len(expected))
        for row, (case_name, gpu, query_tokens) in zip(
                self.value["cases"], expected):
            self.assertEqual(row["case"], case_name)
            self.assertEqual(row["physical_gpu"], gpu)
            self.assertEqual(row["query_tokens"], query_tokens)
            self.assertTrue(row["qualified"])
            self.assertEqual(row["reason_count"], 0)
            self.assertGreaterEqual(
                row["timing"]["control_over_candidate_speedup"],
                contract["minimum_qk_speedup"],
            )
            for numerical in row["numerical"]:
                self.assertTrue(numerical["finite"])
                self.assertTrue(all(
                    math.isfinite(float(numerical[field]))
                    for field in (
                        "candidate_vs_control_relative_l2",
                        "candidate_vs_control_max_abs",
                        "candidate_vs_fp64_sample_relative_l2",
                        "candidate_vs_fp64_sample_max_abs",
                    )
                ))
                self.assertLessEqual(
                    numerical["candidate_vs_control_relative_l2"],
                    contract["relative_l2_limit"],
                )
                self.assertLessEqual(
                    numerical["candidate_vs_control_max_abs"],
                    contract["max_abs_limit"],
                )

    def test_lifecycle_is_clean_and_outer_session_was_reaped(self) -> None:
        self.assertTrue(all(
            returncode == 0
            for returncode in self.value[
                "lifecycle_returncodes"].values()
        ))
        self.assertTrue(
            self.value["adjudication"]["outer_session_reaped"])

    def test_only_component_integration_is_authorized(self) -> None:
        adjudication = self.value["adjudication"]
        self.assertTrue(adjudication["qualified"])
        self.assertTrue(adjudication["corrected_cpu_fp64_run_qualified"])
        self.assertTrue(adjudication["full_pipeline_integration_authorized"])
        self.assertFalse(adjudication["tp4_service_authorized"])
        self.assertFalse(adjudication["main_or_yaml_change_authorized"])

    def test_evidence_contains_no_sensitive_payloads(self) -> None:
        self.assertTrue(all(
            value is False for value in self.value["privacy"].values()
        ))
        serialized = EVIDENCE.read_text(encoding="utf-8").lower()
        for marker in (
            "request_contract_sha256",
            "semantic_output_sha256",
            "session_token",
            "begin openssh private key",
            "modelhub_access_token",
        ):
            self.assertNotIn(marker, serialized)


if __name__ == "__main__":
    unittest.main()
