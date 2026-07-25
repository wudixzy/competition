import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs/experiments/evidence/"
    "M1_52_V5_FUNCTIONAL_AGENT_20260725.json"
)


class M152V5FunctionalAgentEvidenceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.value = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_result_is_exact_source_and_runtime_bound(self):
        source = self.value["source"]
        self.assertEqual(
            source["runtime_revision"],
            "aee89b44d27fe4a9e94e6a2de7e6dd2fb0adbeab",
        )
        self.assertEqual(
            source["runtime_overlay_sha256"],
            "693b067dc8ffb9fcbe2a758a6e52b4ba4a7fc61e716f91e3fec525e40cf28d46",
        )
        self.assertEqual(self.value["runtime"]["max_model_len"], 262144)
        self.assertEqual(self.value["runtime"]["tensor_parallel_size"], 4)

    def test_functional_and_agent_contracts_qualified(self):
        functional = self.value["functional_gate"]
        agent = self.value["agent_gate"]
        self.assertTrue(functional["summary"]["complete"])
        self.assertTrue(functional["summary"]["qualified"])
        self.assertEqual(functional["summary"]["passed"], 52)
        self.assertEqual(functional["summary"]["failed"], 0)
        self.assertEqual(functional["summary"]["skipped"], 1)
        self.assertEqual(functional["documented_skip"]["id"], "n_2")
        self.assertTrue(agent["summary"]["complete"])
        self.assertTrue(agent["summary"]["qualified"])
        self.assertEqual(agent["summary"]["passed"], 11)
        self.assertEqual(agent["summary"]["failed"], 0)

    def test_lifecycle_artifacts_and_privacy_are_clean(self):
        lifecycle = self.value["lifecycle"]
        self.assertEqual(lifecycle["overall_rc"], 0)
        for name, rc in lifecycle["gates"].items():
            self.assertEqual(rc, 0, name)
        self.assertTrue(lifecycle["fatal_scan_empty"])
        self.assertEqual(lifecycle["free_memory_drop_bytes"], [0, 0, 0, 0])
        for key, value in self.value["privacy"].items():
            self.assertFalse(value, key)
        for digest in self.value["artifact_sha256"].values():
            self.assertEqual(len(digest), 64)
            int(digest, 16)

    def test_old_platform_main_is_not_a_comparator(self):
        separation = self.value["platform_main_separation"]
        self.assertFalse(separation["platform_source_revision_known"])
        self.assertFalse(separation["platform_runtime_overlay_known"])
        self.assertGreaterEqual(
            separation["runtime_candidate_commits_ahead_of_local_main"], 141)
        self.assertFalse(
            separation["direct_gain_or_regression_claim_authorized"])
        self.assertFalse(
            self.value["decision"]["main_or_yaml_change_authorized"])


if __name__ == "__main__":
    unittest.main()
