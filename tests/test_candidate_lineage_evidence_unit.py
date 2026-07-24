import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs/experiments/evidence/M1_51_CANDIDATE_LINEAGE_20260725.json"
)


class CandidateLineageEvidenceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.value = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_platform_main_is_not_a_candidate_comparison(self):
        platform = self.value["evidence_partitions"]["platform_main_881"]
        self.assertIsNone(platform["source_revision"])
        self.assertIsNone(platform["runtime_overlay_sha256"])
        self.assertFalse(platform["eligible_for_candidate_comparison"])

    def test_current_sources_are_separate_and_bound(self):
        lineage = self.value["lineage"]
        baseline = lineage["formal_quality_baseline_source"]
        candidate = lineage["m1_51_implementation_source"]
        self.assertEqual(len(baseline), 40)
        self.assertEqual(len(candidate), 40)
        self.assertNotEqual(baseline, candidate)
        self.assertEqual(lineage["m1_51_parent"], baseline)
        self.assertGreaterEqual(
            lineage["local_main_to_formal_baseline"]["commits_ahead"], 120)

    def test_unqualified_candidate_cannot_change_defaults(self):
        promotion = self.value["promotion"]
        self.assertFalse(promotion["authorized"])
        self.assertFalse(promotion["main_change_authorized"])
        self.assertFalse(promotion["computility_run_change_authorized"])
        self.assertEqual(
            self.value["evidence_partitions"]["m1_51_named_tool_parser"]
            ["runtime_quality_gate"],
            "not-run",
        )


if __name__ == "__main__":
    unittest.main()
