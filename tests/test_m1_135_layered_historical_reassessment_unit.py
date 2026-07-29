from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "docs/experiments/evidence/"
    "M1_135_LAYERED_HISTORICAL_REASSESSMENT_20260729"
)


class M1135LayeredHistoricalReassessmentTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(
            (EVIDENCE / "classification.json").read_text(encoding="utf-8"))
        cls.by_id = {
            row["id"]: row for row in cls.report["candidates"]
        }

    def test_identity_and_unique_candidates(self) -> None:
        self.assertEqual(
            self.report["schema"],
            "bi100-layered-historical-reassessment-v1",
        )
        self.assertEqual(len(self.by_id), len(self.report["candidates"]))

    def test_hard_failures_are_not_reopened_by_semantics(self) -> None:
        self.assertEqual(
            self.by_id["E-ATTN-06"]["classification"],
            "hard-operator-reject",
        )
        self.assertEqual(
            self.by_id["admission64/direct-stale-state"][
                "classification"],
            "hard-cache-transparency-reject",
        )

    def test_trajectory_and_performance_decisions_are_separate(self) -> None:
        self.assertEqual(
            self.by_id["E-MOE-04"]["trajectory_identity"],
            "different-diagnostic-only",
        )
        self.assertEqual(
            self.by_id["M1-98"]["classification"],
            "closed-end-to-end-benefit",
        )
        self.assertEqual(
            self.by_id["M1-109"]["classification"],
            "pending-control-repeat-attribution",
        )

    def test_no_release_authorization(self) -> None:
        self.assertTrue(self.report["authorization"])
        self.assertTrue(all(
            value is False
            for value in self.report["authorization"].values()
        ))

    def test_checksum_manifest(self) -> None:
        expected, name = (EVIDENCE / "SHA256SUMS").read_text(
            encoding="utf-8").split()
        self.assertEqual(name, "classification.json")
        actual = hashlib.sha256(
            (EVIDENCE / name).read_bytes()).hexdigest()
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
