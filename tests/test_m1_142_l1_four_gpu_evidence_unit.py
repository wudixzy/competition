from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs" / "experiments" / "evidence"
    / "M1_142_L1_FOUR_GPU_20260730"
)
SOURCE_REVISION = "96311e85abe14da07c4b23460c563415fe7d1d65"
EXTENSION_SHA256 = (
    "f94ad8abb554c6b2eb1a972ad7198cc4d57d36a166afe3e9b869985dda543236"
)
CASES = (
    "production_dense_q8176",
    "production_65k_q8176",
    "production_128k_q8176",
    "production_235k_q5616",
)


class M1142L1FourGpuEvidenceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.status = json.loads(
            (EVIDENCE / "runner_status.json").read_text(encoding="ascii"))
        cls.screen = json.loads(
            (EVIDENCE / "screen.json").read_text(encoding="ascii"))

    def test_checksums_cover_and_match_every_retained_json(self):
        lines = (
            EVIDENCE / "SHA256SUMS"
        ).read_text(encoding="ascii").splitlines()
        recorded = {
            name: digest
            for digest, name in (line.split("  ", 1) for line in lines)
        }
        json_files = sorted(path.name for path in EVIDENCE.glob("*.json"))
        self.assertEqual(sorted(recorded), json_files)
        for name, expected in recorded.items():
            self.assertEqual(
                hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest(),
                expected,
            )

    def test_full_l1_pass_only_authorizes_l2(self):
        status = self.status
        self.assertEqual(status["source_revision"], SOURCE_REVISION)
        self.assertEqual(status["extension_sha256"], EXTENSION_SHA256)
        self.assertEqual(status["gpus"], [0, 1, 2, 3])
        self.assertEqual(status["waves"], 1)
        self.assertTrue(status["qualified"])
        self.assertTrue(
            status["screen"]["full_l1_contract_satisfied"])
        authorization = status["authorization"]
        self.assertTrue(authorization["l2_capture_authorized"])
        self.assertFalse(authorization["main_or_yaml_change_authorized"])
        self.assertFalse(authorization["official_score_claim_authorized"])
        self.assertFalse(any(status["privacy"].values()))

    def test_each_fixed_cell_passes_independent_numeric_and_speed_gate(self):
        rows = self.screen["rows"]
        self.assertEqual([row["case"] for row in rows], list(CASES))
        self.assertEqual([row["gpu"] for row in rows], [0, 1, 2, 3])
        for row in rows:
            self.assertTrue(row["qualified"])
            self.assertTrue(row["finite"])
            self.assertTrue(math.isfinite(row["speedup"]))
            self.assertGreaterEqual(row["speedup"], 1.5)
            self.assertLessEqual(row["output_relative_l2"], 1e-5)
            self.assertLessEqual(row["lse_relative_l2"], 1e-5)
            self.assertLessEqual(row["output_max_abs"], 1e-3)
        self.assertTrue(self.screen["screen_qualified"])
        self.assertEqual(self.screen["reasons"], [])

    def test_lifecycle_is_clean_and_gpu_memory_is_unchanged(self):
        lifecycle = self.status["lifecycle"]
        self.assertTrue(all(lifecycle.values()))
        fatal = json.loads(
            (EVIDENCE / "fatal_scan.json").read_text(encoding="ascii"))
        self.assertTrue(fatal["qualified"])
        self.assertFalse(any(fatal["category_counts"].values()))
        comparison = json.loads(
            (EVIDENCE / "preflight_comparison.json").read_text(
                encoding="ascii"))
        self.assertTrue(comparison["qualified"])
        for stage in comparison["stages"]:
            self.assertTrue(stage["qualified"])
            self.assertFalse(any(
                stage["free_memory_drop_from_first_bytes"].values()))
            self.assertEqual(
                [row["free"] for row in stage["results"]],
                [34_057_748_480] * 4,
            )
        postflight = json.loads(
            (EVIDENCE / "postflight_after.json").read_text(
                encoding="ascii"))
        self.assertTrue(postflight["qualified"])
        self.assertEqual(postflight["api_server_pids"], [])
        self.assertEqual(postflight["worker_pids"], [])
        self.assertEqual(postflight["gpu_processes"], [])


if __name__ == "__main__":
    unittest.main()
