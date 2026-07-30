from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "docs"
    / "experiments"
    / "evidence"
    / "M1_157_FP16_QK_AB_20260730"
)
SOURCE_REVISION = "1e9104ede5d7a62260ca60d643827dea66228fe4"
BASELINE_SHA256 = (
    "f94ad8abb554c6b2eb1a972ad7198cc4d57d36a166afe3e9b869985dda543236"
)
CANDIDATE_SHA256 = (
    "9d7a9da47d540e58ade3fb2d1ce44ecf50d6a057ab55afade9700ec09679e9df"
)
CASES = (
    ("p90_total_16k_q8176", 1, 1.152648132033854),
    ("p90_total_32k_q8176", 2, 1.1646559973718142),
    ("p90_total_64k_q8176", 3, 1.172346654430389),
)


class M1157Fp16QkEvidenceTest(unittest.TestCase):
    def test_manifest_authenticates_every_evidence_file(self):
        rows = [
            line.split("  ", 1)
            for line in (EVIDENCE / "SHA256SUMS")
            .read_text(encoding="ascii")
            .splitlines()
        ]
        expected = {
            f"./{path.relative_to(EVIDENCE).as_posix()}"
            for path in EVIDENCE.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        }
        self.assertEqual({name for _, name in rows}, expected)
        for digest, name in rows:
            self.assertEqual(
                hashlib.sha256(
                    (EVIDENCE / name.removeprefix("./")).read_bytes()
                ).hexdigest(),
                digest,
            )

    def test_numeric_failure_blocks_promotion_but_lifecycle_passes(self):
        runner = json.loads(
            (EVIDENCE / "runner_status.json").read_text(encoding="ascii")
        )
        self.assertFalse(runner["qualified"])
        self.assertEqual(runner["source_revision"], SOURCE_REVISION)
        self.assertEqual(
            runner["baseline_extension_sha256"], BASELINE_SHA256
        )
        self.assertEqual(
            runner["candidate_extension_sha256"], CANDIDATE_SHA256
        )
        self.assertTrue(all(runner["lifecycle"].values()))
        self.assertFalse(
            runner["authorization"]["short_tp4_screen_authorized"]
        )
        self.assertFalse(
            runner["authorization"]["main_or_yaml_change_authorized"]
        )

    def test_fixed_cells_show_speedup_and_frozen_numeric_failure(self):
        for case, gpu, expected_speedup in CASES:
            cell = json.loads(
                (EVIDENCE / f"{case}.json").read_text(encoding="ascii")
            )
            self.assertEqual(cell["visible_physical_gpu"], gpu)
            self.assertAlmostEqual(
                cell["timings"]["speedup"], expected_speedup
            )
            self.assertGreaterEqual(cell["timings"]["speedup"], 1.15)
            self.assertFalse(cell["evaluation"]["qualified"])
            candidate = cell["numerical"]["candidate_vs_reference"]
            self.assertTrue(candidate["finite"])
            self.assertGreater(candidate["output_relative_l2"], 1e-5)
            self.assertLess(candidate["output_relative_l2"], 2e-5)
            self.assertLess(candidate["lse_relative_l2"], 4e-8)
            self.assertLess(candidate["output_max_abs"], 1e-4)


if __name__ == "__main__":
    unittest.main()
