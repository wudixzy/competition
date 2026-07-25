import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs/experiments/evidence/"
    "M1_52_IFEVAL_4096_REJECTION_20260725.json"
)


class M152IFEval4096RejectionEvidenceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.value = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_source_and_request_contract_are_exact(self):
        self.assertEqual(
            self.value["source"]["runtime_revision"],
            "aee89b44d27fe4a9e94e6a2de7e6dd2fb0adbeab",
        )
        self.assertEqual(self.value["request_contract"]["max_tokens"], 4096)
        self.assertFalse(self.value["request_contract"]["thinking_override"])

    def test_failure_is_bound_to_the_completion_cap(self):
        observation = self.value["observation"]
        self.assertEqual(observation["attempted_requests"], 10)
        self.assertEqual(observation["http_200_count"], 10)
        self.assertEqual(observation["failed_responses"], 3)
        self.assertEqual(observation["failed_generated_tokens"], [4096] * 3)
        self.assertEqual(observation["fatal_runtime_matches"], 0)
        self.assertFalse(observation["response_error_type_persisted"])

    def test_cleanup_and_hardware_failure_are_not_hidden(self):
        lifecycle = self.value["lifecycle"]
        self.assertEqual(lifecycle["overall_rc"], 1)
        self.assertEqual(lifecycle["gates"]["cleanup"], 0)
        self.assertEqual(lifecycle["gates"]["fatal_scan"], 0)
        self.assertEqual(lifecycle["gates"]["preflight_after"], 1)
        self.assertTrue(lifecycle["checkpoint"]["absent_after_cleanup"])
        self.assertEqual(
            self.value["gpu_health"]["independent_recovery_preflight"]
            ["gpu0_stage"],
            "mem_get_info",
        )
        self.assertFalse(
            self.value["gpu_health"]["v2_run_authorized_on_current_health"])

    def test_rejection_cannot_authorize_promotion(self):
        decision = self.value["decision"]
        self.assertTrue(decision["v1_result_rejected"])
        for name, value in decision.items():
            if name.endswith("authorized"):
                self.assertFalse(value, name)
        for name, value in self.value["privacy"].items():
            self.assertFalse(value, name)


if __name__ == "__main__":
    unittest.main()
