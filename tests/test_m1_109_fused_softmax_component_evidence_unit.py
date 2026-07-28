from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs" / "experiments" / "evidence"
    / "M1_109_FUSED_SOFTMAX_COMPONENT_20260729"
)
PREBUILT = (
    ROOT / "qwen3_6_scripts" / "prebuilt"
    / "corex-3.2.3-ivcore10" / "corex_fused_paged_prefill.so"
)
NEW_SHA = "ad4ea7707bb2f2bfe04e07a7ad5fd58a647232be70a3056937a0d738c8254bff"
OLD_SHA = "f654eee2c0677812394ff419d316e7e8c98ed1bcc84853a7f8d2ed5755503009"
SOURCE_REVISION = "354e383efd4199af45e770059bcd415ebf8fcc71"


class M1109FusedSoftmaxComponentEvidenceUnitTest(unittest.TestCase):

    def test_manifest_covers_and_authenticates_every_evidence_file(self):
        rows = [
            line.split("  ", 1)
            for line in (EVIDENCE / "SHA256SUMS").read_text(
                encoding="utf-8").splitlines()
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

    def test_component_result_qualifies_without_promoting_defaults(self):
        report = json.loads(
            (EVIDENCE / "comparison.json").read_text(encoding="utf-8"))
        self.assertTrue(report["qualified"], report["reasons"])
        self.assertEqual(report["reasons"], [])
        self.assertEqual(report["old_extension_sha256"], OLD_SHA)
        self.assertEqual(report["new_extension_sha256"], NEW_SHA)
        self.assertEqual(report["positive_cases"], 4)
        self.assertGreaterEqual(
            report["median_old_over_new_speedup"],
            1.90,
        )
        self.assertTrue(
            report["decision"]["tp4_service_experiment_authorized"])
        self.assertFalse(
            report["decision"]["main_or_yaml_change_authorized"])
        self.assertFalse(
            report["decision"]["official_score_claim_authorized"])

    def test_all_fixed_cells_pass_numerical_and_performance_limits(self):
        report = json.loads(
            (EVIDENCE / "comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(len(report["rows"]), 4)
        for row in report["rows"]:
            self.assertGreater(row["old_over_new_speedup"], 1.0)
            self.assertLessEqual(row["new_output_relative_l2"], 1e-5)
            self.assertLessEqual(row["new_lse_relative_l2"], 1e-5)
            self.assertLessEqual(row["new_output_max_abs"], 1e-3)
            cell = json.loads(
                (EVIDENCE / f"{row['case']}_new.json").read_text(
                    encoding="utf-8"))
            self.assertTrue(cell["numerical"]["finite"])
            self.assertEqual(
                cell["physical_block_permutation"],
                cell["context_len"] > 0,
            )
            self.assertTrue(cell["evaluation"]["qualified"])
            self.assertFalse(
                cell["authorization"]["main_or_yaml_change_authorized"])

    def test_source_binary_and_lifecycle_identities_match(self):
        self.assertEqual(
            (EVIDENCE / "source_revision.txt").read_text(
                encoding="utf-8").strip(),
            SOURCE_REVISION,
        )
        self.assertEqual(
            (EVIDENCE / "new_extension_sha256.txt").read_text(
                encoding="utf-8").strip(),
            NEW_SHA,
        )
        self.assertEqual(
            hashlib.sha256(PREBUILT.read_bytes()).hexdigest(),
            NEW_SHA,
        )
        for name in (
            "postflight_before.rc",
            "preflight_before.rc",
            "final_postflight.rc",
            "preflight_after.rc",
            "preflight_comparison.rc",
            "fatal_scan.rc",
        ):
            self.assertEqual(
                (EVIDENCE / name).read_text(encoding="utf-8").strip(),
                "0",
            )
        self.assertTrue(json.loads(
            (EVIDENCE / "final_postflight.json").read_text(
                encoding="utf-8"))["qualified"])
        self.assertTrue(json.loads(
            (EVIDENCE / "preflight_comparison.json").read_text(
                encoding="utf-8"))["qualified"])


if __name__ == "__main__":
    unittest.main()
