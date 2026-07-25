import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs/experiments/evidence/"
    "M1_52_V4_TARGETED_QUALITY_20260725.json"
)


class M152V4TargetedEvidenceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.value = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_result_is_exactly_bound_and_rejected(self):
        self.assertEqual(
            self.value["source"]["revision"],
            "57574b7f828d2326ca53cf0931352db355136737",
        )
        self.assertEqual(
            self.value["source"]["runtime_overlay_sha256"],
            "c15efcfb170721bab221c1688642c9e2d4451e17af6531ee977c9b0239f7ba55",
        )
        self.assertFalse(self.value["matrix"]["qualified"])
        self.assertFalse(
            self.value["decision"]["main_or_yaml_change_authorized"])
        self.assertFalse(
            self.value["decision"]["admission64_authorized"])

    def test_reasoning_failure_is_not_a_budget_or_service_failure(self):
        cases = {case["id"]: case for case in self.value["cases"]}
        reasoning = cases["131k_reasoning_recall"]
        self.assertEqual(reasoning["status"], "fail")
        self.assertEqual(reasoning["http_status"], 200)
        self.assertEqual(reasoning["finish_reason"], "stop")
        self.assertLess(
            reasoning["completion_tokens"], reasoning["max_tokens"])
        self.assertGreater(reasoning["content_length"], 0)
        self.assertGreater(reasoning["reasoning_length"], 0)
        diagnosis = self.value["diagnosis"]["131k_reasoning_recall"]
        self.assertFalse(diagnosis["budget_exhaustion"])
        self.assertFalse(diagnosis["service_or_protocol_failure"])

    def test_235k_agent_contract_is_confirmed(self):
        agent = next(
            case for case in self.value["cases"]
            if case["id"] == "235k_agent_large_output_budget")
        self.assertEqual(agent["status"], "pass")
        self.assertEqual(agent["cached_tokens"], [0, 234992])
        self.assertEqual(agent["tool_call_counts"], [1, 1])
        self.assertEqual(agent["tool_argument_json_types"], ["dict", "dict"])
        self.assertTrue(agent["cold_warm_exact"])

    def test_runtime_difference_is_metadata_only(self):
        note = self.value["overlay_reproducibility_note"]
        self.assertEqual(note["runtime_file_diff_count"], 2)
        self.assertTrue(note["runtime_implementation_files_identical"])
        self.assertEqual(set(note["differing_files"]), {
            "transformers-4.55.3.dist-info/RECORD",
            "transformers-4.55.3.dist-info/direct_url.json",
        })

    def test_lifecycle_artifacts_and_privacy_are_clean(self):
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
