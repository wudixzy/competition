import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs/experiments/evidence/"
    "M1_51_FINE32_LONG_CONTEXT_20260725.json"
)


class M151LongContextEvidenceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.value = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_result_is_bound_and_not_qualified(self):
        self.assertEqual(
            self.value["source"]["revision"],
            "8c00ba6c6a916b68d1d020330ccf0e4d7fb0800c",
        )
        self.assertEqual(
            self.value["source"]["runtime_overlay_sha256"],
            "373d5d28818b8c7b42f0a169d6eac8649fc7162a4d4641e615bde166ac29a9b0",
        )
        self.assertFalse(self.value["decision"]["long_context_gate_qualified"])
        self.assertFalse(self.value["decision"]["main_or_yaml_change_authorized"])

    def test_failures_are_diagnosed_without_values(self):
        failed = {
            case["id"]: case for case in self.value["cases"]
            if case["status"] == "fail"
        }
        self.assertEqual(set(failed), {
            "65k_multiturn_large_tools",
            "131k_reasoning_recall",
            "235k_agent_large_output_budget",
        })
        self.assertEqual(failed["65k_multiturn_large_tools"]["request_count"], 2)
        self.assertEqual(failed["131k_reasoning_recall"]["request_count"], 1)
        self.assertEqual(
            self.value["diagnosis"]["65k_multiturn_large_tools"]
            ["classification"],
            "test-contract-contradiction",
        )

    def test_capacity_cache_and_lifecycle_did_not_regress(self):
        self.assertTrue(
            self.value["regression_observation"]
            ["passing_case_capabilities_preserved"])
        passed = {
            case["id"] for case in self.value["cases"]
            if case["status"] == "pass"
        }
        self.assertIn("235k_partial_branch", passed)
        self.assertIn("near_262k_capacity", passed)
        lifecycle = self.value["lifecycle"]
        for key, value in lifecycle.items():
            if key.endswith("_rc") and key not in {"overall_rc", "quality_rc"}:
                self.assertEqual(value, 0, key)
        self.assertEqual(lifecycle["free_memory_drop_bytes"], [0, 0, 0, 0])
        self.assertTrue(lifecycle["fatal_scan_empty"])

    def test_evidence_is_privacy_safe(self):
        privacy = self.value["privacy"]
        self.assertFalse(privacy["contains_credentials"])
        self.assertFalse(privacy["contains_raw_requests"])
        self.assertFalse(privacy["contains_raw_model_outputs"])
        self.assertFalse(privacy["contains_tool_argument_values"])
        self.assertFalse(privacy["raw_service_log_committed"])
        for digest in self.value["artifact_sha256"].values():
            self.assertEqual(len(digest), 64)
            int(digest, 16)


if __name__ == "__main__":
    unittest.main()
