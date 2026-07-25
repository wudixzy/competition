import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs/experiments/evidence/"
    "M1_52_V5_TARGETED_QUALITY_20260725.json"
)


class M152V5TargetedEvidenceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.value = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_result_is_bound_and_targeted_only(self):
        self.assertEqual(
            self.value["source"]["revision"],
            "aee89b44d27fe4a9e94e6a2de7e6dd2fb0adbeab",
        )
        self.assertEqual(
            self.value["source"]["runtime_overlay_sha256"],
            "693b067dc8ffb9fcbe2a758a6e52b4ba4a7fc61e716f91e3fec525e40cf28d46",
        )
        self.assertTrue(self.value["matrix"]["qualified"])
        self.assertFalse(
            self.value["matrix"]["quality_run_eligible_for_baseline"])
        self.assertFalse(
            self.value["decision"]["main_or_yaml_change_authorized"])

    def test_reasoning_semantic_contract_passed(self):
        reasoning = self.value["cases"][0]
        facts = reasoning["facts"]
        self.assertEqual(reasoning["id"], "131k_reasoning_recall")
        self.assertEqual(reasoning["status"], "pass")
        for key in (
                "content_arithmetic_present", "content_contains_expected",
                "content_expected_single_occurrence",
                "content_expected_suffix", "content_markers_in_order",
                "content_markers_present", "reasoning_content_split",
                "natural_finish_before_max_tokens"):
            self.assertTrue(facts[key], key)
        self.assertLess(
            reasoning["request"]["completion_tokens"],
            reasoning["request"]["max_tokens"],
        )

    def test_agent_contract_passed_with_exact_warm_repeat(self):
        agent = self.value["cases"][1]
        self.assertEqual(agent["id"], "235k_agent_large_output_budget")
        self.assertEqual(agent["status"], "pass")
        self.assertEqual(agent["requests"]["cached_tokens"], [0, 234992])
        self.assertEqual(agent["requests"]["tool_call_counts"], [1, 1])
        self.assertTrue(agent["facts"]["cold_warm_exact"])

    def test_lifecycle_artifacts_and_privacy_are_clean(self):
        lifecycle = self.value["lifecycle"]
        for key, value in lifecycle.items():
            if key.endswith("_rc"):
                self.assertEqual(value, 0, key)
        self.assertTrue(lifecycle["fatal_scan_empty"])
        self.assertEqual(lifecycle["free_memory_drop_bytes"], [0, 0, 0, 0])
        for key, value in self.value["privacy"].items():
            self.assertFalse(value, key)
        for digest in self.value["artifact_sha256"].values():
            self.assertEqual(len(digest), 64)
            int(digest, 16)


if __name__ == "__main__":
    unittest.main()
