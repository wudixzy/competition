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
    / "M1_159_NORM64_AB_20260730"
)
SOURCE_REVISION = "0b44754787e3deecb1dacd112c91a2eba8dc1c33"
BASELINE_SHA256 = (
    "f94ad8abb554c6b2eb1a972ad7198cc4d57d36a166afe3e9b869985dda543236"
)
CANDIDATE_SHA256 = (
    "e144f47637435eabdbc9701b5dbf0abfe4a5b1dad6ce548840dea52dfa6be9f6"
)
EXPECTED = (
    ("p90_total_16k_q8176", 1, 1.0109656685539663),
    ("p90_total_32k_q8176", 2, 1.0113745783299237),
    ("p90_total_64k_q8176", 3, 1.0123108208938625),
)


class M1159Norm64EvidenceTest(unittest.TestCase):
    def test_manifest_authenticates_recursive_evidence(self):
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
            path = EVIDENCE / name.removeprefix("./")
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_numeric_pass_but_speed_gate_blocks_continuation(self):
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
        self.assertEqual(
            runner["screen"]["reasons"],
            ["median speedup is below 1.08x"],
        )
        self.assertFalse(
            runner["authorization"]["short_tp4_screen_authorized"]
        )
        self.assertFalse(
            runner["authorization"]["main_or_yaml_change_authorized"]
        )

    def test_fixed_p90_cells_are_exact_but_too_slow(self):
        for case, gpu, expected_speedup in EXPECTED:
            cell = json.loads(
                (EVIDENCE / f"{case}.json").read_text(encoding="ascii")
            )
            self.assertEqual(cell["visible_physical_gpu"], gpu)
            self.assertAlmostEqual(
                cell["timings"]["speedup"], expected_speedup
            )
            self.assertLess(cell["timings"]["speedup"], 1.08)
            candidate = cell["numerical"]["candidate_vs_reference"]
            delta = cell["numerical"]["candidate_vs_baseline"]
            self.assertTrue(candidate["finite"])
            self.assertLessEqual(candidate["output_relative_l2"], 1e-5)
            self.assertLessEqual(candidate["output_max_abs"], 1e-3)
            self.assertEqual(delta["output_relative_l2"], 0.0)
            self.assertEqual(delta["output_max_abs"], 0.0)
            self.assertEqual(delta["lse_relative_l2"], 0.0)


if __name__ == "__main__":
    unittest.main()
