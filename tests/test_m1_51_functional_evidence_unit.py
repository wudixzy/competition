import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs/experiments/evidence/M1_51_FINE32_FUNCTIONAL_20260725.json"
)


class M151FunctionalEvidenceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.value = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_result_is_bound_to_exact_source_and_overlay(self):
        self.assertEqual(
            self.value["source"]["revision"],
            "8c00ba6c6a916b68d1d020330ccf0e4d7fb0800c",
        )
        self.assertEqual(
            self.value["source"]["runtime_overlay_sha256"],
            "373d5d28818b8c7b42f0a169d6eac8649fc7162a4d4641e615bde166ac29a9b0",
        )

    def test_functional_and_agent_gates_pass(self):
        functional = self.value["functional"]
        self.assertEqual(
            (functional["total"], functional["passed"],
             functional["failed"], functional["skipped"]),
            (53, 52, 0, 1),
        )
        self.assertEqual(functional["skip_ids"], ["n_2"])
        self.assertTrue(functional["qualified"])
        agent = self.value["agent_workload"]
        self.assertEqual(
            (agent["total"], agent["passed"], agent["failed"]),
            (11, 11, 0),
        )
        self.assertTrue(agent["qualified"])
        self.assertTrue(
            self.value["named_tool_repair_gate"]
            ["arguments_valid_json_and_semantics_checked"])

    def test_lifecycle_passes_without_gpu_leak(self):
        lifecycle = self.value["lifecycle"]
        for key, value in lifecycle.items():
            if key.endswith("_rc"):
                self.assertEqual(value, 0, key)
        self.assertEqual(lifecycle["free_memory_drop_bytes"], [0, 0, 0, 0])
        self.assertTrue(lifecycle["fatal_scan_empty"])

    def test_functional_pass_does_not_authorize_promotion(self):
        decision = self.value["decision"]
        self.assertTrue(decision["functional_gate_qualified"])
        self.assertFalse(decision["performance_score_established"])
        self.assertFalse(decision["long_context_gate_established"])
        self.assertFalse(decision["overall_promotion_authorized"])
        self.assertFalse(decision["main_or_yaml_change_authorized"])

    def test_evidence_is_privacy_safe(self):
        privacy = self.value["privacy"]
        self.assertFalse(privacy["contains_credentials"])
        self.assertFalse(privacy["contains_raw_requests"])
        self.assertFalse(privacy["contains_raw_model_outputs"])
        self.assertFalse(privacy["contains_tool_arguments"])
        self.assertFalse(privacy["raw_service_log_committed"])
        for digest in self.value["artifact_sha256"].values():
            self.assertEqual(len(digest), 64)
            int(digest, 16)


if __name__ == "__main__":
    unittest.main()
