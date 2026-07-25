import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs/experiments/evidence/M1_52_CURRENT_LINEAGE_20260725.json"
)


class M152CurrentLineageEvidenceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.value = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_candidate_is_materially_newer_than_main(self):
        lineage = self.value["lineage"]
        delta = lineage["local_main_to_runtime_candidate"]
        self.assertEqual(delta["commits_behind"], 0)
        self.assertGreaterEqual(delta["commits_ahead"], 141)
        self.assertGreaterEqual(delta["changed_files"], 254)
        self.assertNotEqual(
            lineage["local_main"], lineage["runtime_candidate"])

    def test_runtime_source_and_evidence_only_commit_are_separated(self):
        delta = self.value["lineage"][
            "runtime_candidate_to_evidence_parent"]
        self.assertEqual(delta["commits_ahead"], 1)
        self.assertFalse(delta["runtime_implementation_changed"])

    def test_cross_version_claims_and_promotion_are_forbidden(self):
        platform = self.value["result_partitions"]["platform_main_881"]
        policy = self.value["comparison_policy"]
        promotion = self.value["promotion"]
        self.assertIsNone(platform["source_revision"])
        self.assertFalse(platform["eligible_for_latest_candidate_comparison"])
        self.assertTrue(policy["cross_version_gain_claims_forbidden"])
        self.assertFalse(promotion["authorized"])
        self.assertFalse(promotion["main_change_authorized"])
        self.assertFalse(promotion["computility_run_change_authorized"])


if __name__ == "__main__":
    unittest.main()
