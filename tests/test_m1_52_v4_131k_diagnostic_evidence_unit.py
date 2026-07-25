import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs/experiments/evidence/"
    "M1_52_V4_131K_DIAGNOSTIC_20260725.json"
)


class M152V4131KDiagnosticEvidenceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.value = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_result_is_bound_and_v4_remains_rejected(self):
        self.assertEqual(
            self.value["source"]["revision"],
            "5a468d16d040f82ae1bc50d830eec72bb7d5ec58",
        )
        self.assertEqual(
            self.value["source"]["runtime_overlay_sha256"],
            "5039020d106b53d779efee22a42d5e2c2d3a69ed53fb405e368f8bb696df8544",
        )
        self.assertFalse(self.value["matrix"]["qualified"])
        self.assertFalse(self.value["decision"]["v4_result_reclassified"])
        self.assertFalse(
            self.value["decision"]["main_or_yaml_change_authorized"])

    def test_semantics_pass_but_strict_format_fails(self):
        facts = self.value["case"]["facts"]
        for key in (
                "content_arithmetic_present", "content_contains_expected",
                "content_expected_suffix", "content_markers_in_order",
                "content_markers_present"):
            self.assertTrue(facts[key], key)
        self.assertFalse(facts["content_exact_expected"])
        self.assertFalse(facts["content_expected_prefix"])
        request = self.value["case"]["request"]
        self.assertEqual(request["finish_reason"], "stop")
        self.assertLess(request["completion_tokens"], request["max_tokens"])

    def test_next_contract_keeps_independent_instruction_gate(self):
        diagnosis = self.value["diagnosis"]
        self.assertEqual(
            diagnosis["classification"],
            "semantic-long-context-pass-with-strict-format-failure",
        )
        self.assertTrue(
            self.value["decision"]["strict_instruction_gate_required_before_promotion"])
        self.assertTrue(self.value["decision"]["v5_contract_required"])

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
