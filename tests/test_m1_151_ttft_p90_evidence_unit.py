from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs" / "experiments" / "evidence"
    / "M1_151_TTFT_P90_PREFILL_GRID_20260730"
)
EXTENSION_SHA256 = (
    "f94ad8abb554c6b2eb1a972ad7198cc4d57d36a166afe3e9b869985dda543236"
)
KERNEL_SHA256 = (
    "11c387e6012834fe634ffa8d038f7a4bf4ec19fa13ec23779ee1f414037e564b"
)


class M1151TtftP90EvidenceUnitTest(unittest.TestCase):

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

    def test_fixed_p90_grid_qualifies_without_production_authority(self):
        status = json.loads(
            (EVIDENCE / "runner_status.json").read_text(encoding="ascii"))
        self.assertTrue(status["qualified"])
        self.assertEqual(status["gpus"], [1, 2, 3])
        self.assertEqual(status["gpu_count"], 3)
        self.assertEqual(status["extension_sha256"], EXTENSION_SHA256)
        self.assertEqual(status["screen"]["reasons"], [])
        self.assertGreaterEqual(status["screen"]["minimum_speedup"], 1.9)
        self.assertGreaterEqual(status["screen"]["median_speedup"], 2.2)
        self.assertTrue(
            status["authorization"]["short_tp4_p90_screen_authorized"])
        self.assertFalse(
            status["authorization"]["l2_capture_authorized"])
        self.assertFalse(
            status["authorization"]["main_or_yaml_change_authorized"])
        self.assertFalse(
            status["authorization"]["official_score_claim_authorized"])

    def test_all_cells_pass_fixed_numeric_and_speed_screens(self):
        screen = json.loads(
            (EVIDENCE / "screen.json").read_text(encoding="ascii"))
        self.assertTrue(screen["qualified"])
        self.assertEqual(len(screen["rows"]), 8)
        self.assertEqual(
            [row["total_kv_len"] for row in screen["rows"]],
            list(range(8176, 65521, 8192)),
        )
        for row in screen["rows"]:
            self.assertTrue(row["qualified"])
            self.assertTrue(row["finite"])
            self.assertGreaterEqual(row["speedup"], 1.2)
            self.assertLessEqual(row["output_relative_l2"], 1.0e-5)
            self.assertLessEqual(row["lse_relative_l2"], 1.0e-5)
            self.assertLessEqual(row["output_max_abs"], 1.0e-3)

    def test_artifact_and_lifecycle_identities_match(self):
        identity = json.loads(
            (EVIDENCE / "identity.json").read_text(encoding="ascii"))
        self.assertEqual(identity["extension_sha256"], EXTENSION_SHA256)
        self.assertEqual(identity["kernel_source_sha256"], KERNEL_SHA256)
        comparison = json.loads(
            (EVIDENCE / "preflight_comparison.json").read_text(
                encoding="ascii"))
        self.assertTrue(comparison["qualified"])
        self.assertEqual(comparison["expected_gpus"], [1, 2, 3])
        for stage in comparison["stages"]:
            self.assertTrue(stage["qualified"])
            self.assertEqual(
                set(stage["free_memory_drop_from_first_bytes"].values()),
                {0},
            )
        fatal = json.loads(
            (EVIDENCE / "fatal_scan.json").read_text(encoding="ascii"))
        self.assertTrue(fatal["qualified"])
        self.assertEqual(set(fatal["category_counts"].values()), {0})
        self.assertFalse(fatal["raw_messages_recorded"])


if __name__ == "__main__":
    unittest.main()
