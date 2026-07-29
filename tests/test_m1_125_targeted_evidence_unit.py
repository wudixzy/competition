from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "docs/experiments/evidence"
    / "M1_125_ADMISSION64_PARTIAL_BRANCH_TARGETED_20260729"
    / "result.json"
)


class M1125TargetedEvidenceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.value = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_fixed_runtime_and_case_identity(self) -> None:
        self.assertEqual(
            self.value["source_revision"],
            "e59ebb654722d2e1de19a45633f460bbc0df27b3",
        )
        self.assertEqual(
            self.value["comparator_revision"],
            "42c1b03db0393097915810b6c7eb962836401f9d",
        )
        self.assertEqual(self.value["gpu_count"], 4)
        self.assertEqual(self.value["tensor_parallel_size"], 4)
        self.assertEqual(self.value["max_model_len"], 262144)
        self.assertEqual(self.value["case"], "32k_partial_branch")

    def test_both_arms_prove_sparse_admission_sequence(self) -> None:
        expected_cached = [0, 32752, 0, 32704]
        required_facts = {
            "branch_markers_correct",
            "cache_trace_session_attested",
            "cold_warm_exact",
            "first_sibling_effective_miss",
            "repeated_branch_admitted",
            "strict_partial_hit",
            "subsequent_sibling_restored",
            "subsequent_sibling_strict_partial_hit",
        }
        for arm in self.value["arms"].values():
            self.assertTrue(arm["qualified"])
            self.assertEqual(arm["http_statuses"], [200, 200, 200, 200])
            self.assertEqual(arm["cached_tokens"], expected_cached)
            self.assertEqual(set(arm["facts"]), required_facts)
            self.assertTrue(all(arm["facts"].values()))

    def test_lifecycle_is_clean(self) -> None:
        self.assertTrue(all(
            value == 0
            for value in self.value["arm_gate_returncodes"].values()
        ))
        expected_nonzero = {"legacy_complete_comparison": 1}
        observed_nonzero = {
            key: value
            for key, value in self.value[
                "orchestrator_returncodes"].items()
            if value != 0
        }
        self.assertEqual(observed_nonzero, expected_nonzero)
        self.assertTrue(
            self.value["adjudication"][
                "legacy_complete_comparison_failure_expected"]
        )

    def test_targeted_pass_never_authorizes_promotion(self) -> None:
        comparison = self.value["targeted_comparison"]
        self.assertTrue(comparison["qualified"])
        self.assertTrue(comparison["targeted_diagnostic_qualified"])
        self.assertEqual(comparison["reason_count"], 0)
        self.assertFalse(
            comparison["long_context_quality_non_regression_authorized"])
        self.assertFalse(comparison["overall_promotion_authorized"])
        self.assertTrue(
            self.value["adjudication"]["complete_long_context_matrix_required"])
        self.assertFalse(
            self.value["adjudication"]["main_or_yaml_change_authorized"])

    def test_evidence_retains_no_sensitive_payloads(self) -> None:
        self.assertTrue(all(
            value is False for value in self.value["privacy"].values()
        ))
        serialized = EVIDENCE.read_text(encoding="utf-8").lower()
        for marker in (
            "semantic_output_sha256",
            "content_sha256",
            "reasoning_sha256",
            "request_contract_sha256",
            "restore_key",
            "session_token",
            "begin openssh private key",
            "modelhub_access_token",
        ):
            self.assertNotIn(marker, serialized)


if __name__ == "__main__":
    unittest.main()
