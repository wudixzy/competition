import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs/experiments/evidence/QUALITY_FUNCTIONAL_BASELINE_20260725.json"
)


class QualityFunctionalBaselineEvidenceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.value = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_result_is_bound_and_not_a_performance_promotion(self):
        self.assertEqual(
            self.value["source"]["revision"],
            "3cbb98dd5628c545c312cf4c185e399f8c80f32e",
        )
        self.assertEqual(
            self.value["runtime"]["overlay_sha256"],
            "84c27dacebce52620084cb2314a535cc9d409ac20ca98a6c8bbd4b7503188001",
        )
        self.assertFalse(self.value["promotion_authorized"])
        self.assertFalse(
            self.value["decision"]["official_881_score_established"])
        self.assertFalse(
            self.value["decision"]["main_or_yaml_change_authorized"])

    def test_functional_agent_and_lifecycle_gates_pass(self):
        functional = self.value["functional"]
        self.assertEqual(
            (functional["total"], functional["passed"],
             functional["failed"], functional["skipped"]),
            (53, 52, 0, 1),
        )
        self.assertEqual(functional["skip_ids"], ["n_2"])
        self.assertEqual(
            (self.value["agent_workload"]["total"],
             self.value["agent_workload"]["passed"],
             self.value["agent_workload"]["failed"]),
            (11, 11, 0),
        )
        lifecycle = self.value["lifecycle"]
        self.assertEqual(lifecycle["overall_rc"], 0)
        self.assertTrue(lifecycle["all_gate_rc_zero"])
        for key in ("fatal", "oom", "gloo", "worker_loss", "segfault",
                    "cleanup_rc", "residual_api_server_processes"):
            self.assertEqual(lifecycle[key], 0, key)
        self.assertEqual(lifecycle["free_memory_drop_bytes"], [0, 0, 0, 0])

    def test_evidence_is_privacy_safe(self):
        privacy = self.value["privacy"]
        self.assertFalse(privacy["contains_credentials"])
        self.assertFalse(privacy["contains_raw_requests"])
        self.assertFalse(privacy["contains_raw_model_outputs"])
        self.assertFalse(privacy["contains_tool_arguments"])
        self.assertFalse(privacy["raw_service_log_committed"])
        self.assertEqual(len(self.value["artifact_sha256"]), 9)
        for digest in self.value["artifact_sha256"].values():
            self.assertEqual(len(digest), 64)
            int(digest, 16)


if __name__ == "__main__":
    unittest.main()
