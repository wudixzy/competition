import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs/experiments/evidence/"
    "M1_52_V3_TARGETED_QUALITY_20260725.json"
)


class M152V3TargetedEvidenceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.value = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_result_is_exactly_bound_and_not_promotable(self):
        self.assertEqual(
            self.value["source"]["revision"],
            "6dfdab10524d71435dd5d60d2ac80135237e5ccf",
        )
        self.assertEqual(
            self.value["source"]["runtime_overlay_sha256"],
            "6a7298f9b4b60fdb4154e446d8a70cffb5a93290fb9bb6f864f97ebf23669fcd",
        )
        self.assertFalse(self.value["matrix"]["qualified"])
        self.assertFalse(
            self.value["matrix"]["quality_run_eligible_for_baseline"])
        self.assertFalse(
            self.value["decision"]["main_or_yaml_change_authorized"])
        self.assertFalse(
            self.value["decision"]["admission64_authorized"])

    def test_targeted_outcome_and_diagnoses_are_frozen(self):
        matrix = self.value["matrix"]
        self.assertEqual(
            (matrix["selected_total"], matrix["passed"], matrix["failed"]),
            (3, 1, 2),
        )
        cases = {case["id"]: case for case in self.value["cases"]}
        self.assertEqual(cases["65k_multiturn_large_tools"]["status"], "pass")
        self.assertEqual(
            cases["131k_reasoning_recall"]["completion_tokens"],
            cases["131k_reasoning_recall"]["max_tokens"],
        )
        self.assertEqual(
            cases["235k_agent_large_output_budget"]["tool_call_counts"],
            [1, 1],
        )
        self.assertEqual(
            self.value["diagnosis"]["131k_reasoning_recall"]
            ["classification"],
            "test-output-budget-exhausted",
        )
        self.assertEqual(
            self.value["diagnosis"]["235k_agent_large_output_budget"]
            ["classification"],
            "test-tool-content-contract-too-strict",
        )

    def test_platform_main_is_not_a_direct_comparator(self):
        separation = self.value["version_separation"]
        self.assertFalse(separation["platform_main_source_revision_known"])
        self.assertFalse(
            separation["direct_comparison_to_platform_main_authorized"])

    def test_lifecycle_and_privacy_are_clean(self):
        lifecycle = self.value["lifecycle"]
        for key, value in lifecycle.items():
            if key.endswith("_rc") and key not in {"overall_rc", "quality_rc"}:
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
