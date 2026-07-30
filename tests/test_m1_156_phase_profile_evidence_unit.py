from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs" / "experiments" / "evidence"
    / "M1_156_FUSED_PREFILL_PHASE_PROFILE_20260730"
)
SOURCE_REVISION = "4caffefba14aa5d9d519a7226d6fe71e0999893a"
EXTENSION_SHA256 = (
    "f94ad8abb554c6b2eb1a972ad7198cc4d57d36a166afe3e9b869985dda543236"
)
CASES = (
    ("p90_total_16k_q8176", 1, 0.9804542370869119),
    ("p90_total_32k_q8176", 2, 0.9870458623534848),
    ("p90_total_64k_q8176", 3, 0.9902731366396842),
)


class M1156PhaseProfileEvidenceUnitTest(unittest.TestCase):

    def test_manifest_authenticates_every_evidence_file(self):
        rows = [
            line.split("  ", 1)
            for line in (EVIDENCE / "SHA256SUMS").read_text(
                encoding="ascii").splitlines()
        ]
        expected = {
            path.name
            for path in EVIDENCE.iterdir()
            if path.is_file() and path.name != "SHA256SUMS"
        }
        self.assertEqual({name for _, name in rows}, expected)
        for digest, name in rows:
            self.assertEqual(
                hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest(),
                digest,
            )

    def test_runner_and_lifecycle_qualify_without_promotion(self):
        runner = json.loads(
            (EVIDENCE / "runner_status.json").read_text(encoding="ascii"))
        self.assertTrue(runner["qualified"])
        self.assertEqual(runner["terminal_stage"], "phase_qualification")
        self.assertEqual(runner["source_revision"], SOURCE_REVISION)
        self.assertEqual(runner["extension_sha256"], EXTENSION_SHA256)
        self.assertEqual(runner["gpus"], [1, 2, 3])
        self.assertTrue(all(runner["lifecycle"].values()))
        self.assertTrue(runner["screen"]["qualified"])
        self.assertEqual(runner["screen"]["reasons"], [])
        self.assertEqual(
            runner["authorization"],
            {
                "implementation_direction_authorized": True,
                "main_or_yaml_change_authorized": False,
                "official_score_claim_authorized": False,
                "tp4_service_authorized": False,
            },
        )

    def test_fixed_profiles_bind_launches_timing_and_numeric_lineage(self):
        screen = json.loads(
            (EVIDENCE / "screen.json").read_text(encoding="ascii"))
        reports = {row["case"]: row for row in screen["rows"]}
        for case, gpu, expected_ratio in CASES:
            report = reports[case]
            cell = json.loads(
                (EVIDENCE / f"{case}.json").read_text(encoding="ascii"))
            self.assertTrue(report["qualified"])
            self.assertEqual(report["reasons"], [])
            self.assertEqual(report["visible_physical_gpu"], gpu)
            self.assertEqual(report["extension_sha256"], EXTENSION_SHA256)
            self.assertAlmostEqual(
                report["attributed_candidate_ratio"], expected_ratio)
            self.assertGreater(
                report["phases"]["qk"]["percent"]
                + report["phases"]["pv"]["percent"],
                63.0,
            )
            self.assertLess(report["phases"]["gather"]["percent"], 0.6)
            self.assertTrue(cell["evaluation"]["qualified"])
            self.assertTrue(cell["candidate_finite"])
            self.assertTrue(cell["numeric_lineage"]["qualified"])
            self.assertEqual(
                cell["numeric_lineage"]["extension_sha256"],
                EXTENSION_SHA256,
            )
            self.assertLessEqual(
                cell["numeric_lineage"]["output_relative_l2"], 1e-5)
            self.assertLessEqual(
                cell["numeric_lineage"]["lse_relative_l2"], 1e-5)
            self.assertLessEqual(
                cell["numeric_lineage"]["output_max_abs"], 1e-3)


if __name__ == "__main__":
    unittest.main()
