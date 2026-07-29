from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "docs/experiments/evidence"
    / "M1_125_FULL_LONG_CONTEXT_20260729"
    / "result.json"
)


class M1125FullLongContextEvidenceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = EVIDENCE.read_text(encoding="utf-8")
        cls.value = json.loads(cls.raw)

    def test_fixed_tp4_runtime_identity(self) -> None:
        self.assertEqual(
            self.value["source_revision"],
            "e59ebb654722d2e1de19a45633f460bbc0df27b3",
        )
        self.assertEqual(self.value["gpu_count"], 4)
        self.assertEqual(self.value["tensor_parallel_size"], 4)
        self.assertEqual(self.value["max_model_len"], 262144)

    def test_complete_matrix_qualified(self) -> None:
        comparison = self.value["comparison"]
        self.assertTrue(comparison["qualified"])
        self.assertEqual(comparison["compared_cases"], 12)
        self.assertEqual(comparison["qualified_cases"], 12)
        self.assertEqual(comparison["failed_cases"], 0)
        self.assertEqual(comparison["reason_count"], 0)
        self.assertEqual(
            [case["ordinal"] for case in comparison["cases"]],
            list(range(1, 13)),
        )
        self.assertTrue(all(
            case["qualified"] and case["reason_count"] == 0
            for case in comparison["cases"]
        ))

    def test_required_long_context_modes_are_represented(self) -> None:
        cases = {
            case["id"]: case["mode"]
            for case in self.value["comparison"]["cases"]
        }
        self.assertEqual(cases["32k_partial_branch"], "exact")
        self.assertEqual(cases["32k_multimodal_isolation"], "exact")
        self.assertEqual(cases["131k_cold_warm_recall"], "next_token")
        self.assertEqual(cases["131k_reasoning_recall"], "semantic")
        self.assertEqual(
            cases["235k_agent_large_output_budget"],
            "semantic",
        )
        self.assertEqual(cases["near_262k_capacity"], "next_token")

    def test_lifecycle_is_clean(self) -> None:
        lifecycle = self.value["lifecycle"]
        self.assertEqual(lifecycle["returncode_file_count"], 48)
        self.assertTrue(lifecycle["all_returncodes_zero"])
        self.assertEqual(lifecycle["fatal_timeout_scan_file_count"], 6)
        self.assertTrue(lifecycle["all_fatal_timeout_scans_empty"])
        self.assertEqual(
            lifecycle["active_run_process_count_after_completion"],
            0,
        )
        for field in (
            "fatal_detected",
            "timeout_detected",
            "gloo_reset_detected",
            "worker_loss_detected",
        ):
            self.assertFalse(lifecycle[field])

    def test_authorization_remains_scoped(self) -> None:
        authorization = self.value["authorization"]
        self.assertTrue(
            authorization["long_context_quality_non_regression_authorized"]
        )
        self.assertFalse(authorization["overall_promotion_authorized"])
        self.assertFalse(authorization["main_or_yaml_change_authorized"])

    def test_evidence_retains_no_sensitive_payloads(self) -> None:
        self.assertTrue(all(
            value is False for value in self.value["privacy"].values()
        ))
        lowered = self.raw.lower()
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
            self.assertNotIn(marker, lowered)


if __name__ == "__main__":
    unittest.main()
