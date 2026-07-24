import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs/experiments/evidence/"
    "QUALITY_LONG_CONTEXT_BASELINE_62B8B83_20260725.json"
)


class QualityLongContextBaselineEvidenceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.value = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_result_is_bound_but_not_qualified(self):
        self.assertEqual(
            self.value["source"]["revision"],
            "62b8b83dccf3106e9a10ddabc016fb60ed19fc6e",
        )
        self.assertEqual(
            self.value["source"]["runtime_overlay_sha256"],
            "03aad9d1411375adc6e5895933930357f31873f42f7c4eb5582429a17b386c23",
        )
        self.assertFalse(self.value["decision"]["qualified"])
        self.assertFalse(
            self.value["decision"]["main_or_yaml_change_authorized"])

    def test_cases_and_failures_are_complete(self):
        matrix = self.value["matrix"]
        self.assertEqual(
            (matrix["total"], matrix["passed"], matrix["failed"]),
            (12, 9, 3),
        )
        failed = {
            case["id"] for case in self.value["cases"]
            if case["status"] == "fail"
        }
        self.assertEqual(failed, {
            "65k_multiturn_large_tools",
            "131k_reasoning_recall",
            "235k_agent_large_output_budget",
        })
        self.assertTrue(
            self.value["confirmed_capabilities"]["262144_and_minus_one_capacity"])

    def test_lifecycle_is_clean_despite_quality_failure(self):
        lifecycle = self.value["lifecycle"]
        self.assertEqual(lifecycle["quality_rc"], 1)
        for key in (
                "runtime_identity_rc", "runtime_contract_rc",
                "prefix_allocator_rc", "gdn_action_broadcast_rc",
                "preflight_before_rc", "startup_rc",
                "startup_contract_rc", "cleanup_rc", "fatal_scan_rc",
                "preflight_after_rc", "preflight_comparison_rc"):
            self.assertEqual(lifecycle[key], 0, key)
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
