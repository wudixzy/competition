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
    / "M1_162_CALIBRATED_FP16_QK_20260730"
)
SOURCE_REVISION = "eea631a68beaa66a2b4e84346cef2912d5f59e8f"
BASELINE_SHA256 = (
    "ad4ea7707bb2f2bfe04e07a7ad5fd58a647232be70a3056937a0d738c8254bff"
)
CANDIDATE_SHA256 = (
    "36e043f138aa87c635178e4aa6a30af710b87c3f3d7c2a3f1838fc0e365bd368"
)
CASES = (
    ("p90_total_16k_q8176", 1, 1.153618947610942),
    ("p90_total_32k_q8176", 2, 1.165179094285798),
    ("p90_total_64k_q8176", 3, 1.1707189184551994),
)


class M1162CalibratedFp16QkEvidenceTest(unittest.TestCase):
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

    def test_runner_passes_only_the_real_activation_replay_layer(self):
        runner = json.loads(
            (EVIDENCE / "runner_status.json").read_text(encoding="ascii")
        )
        self.assertTrue(runner["qualified"])
        self.assertEqual(runner["source_revision"], SOURCE_REVISION)
        self.assertEqual(
            runner["baseline_extension_sha256"], BASELINE_SHA256
        )
        self.assertEqual(
            runner["candidate_extension_sha256"], CANDIDATE_SHA256
        )
        self.assertTrue(all(runner["lifecycle"].values()))
        authorization = runner["authorization"]
        self.assertTrue(authorization["real_activation_replay_authorized"])
        self.assertFalse(authorization["short_tp4_screen_authorized"])
        self.assertFalse(
            authorization["long_context_or_quality_authorized"]
        )
        self.assertFalse(
            authorization["main_or_yaml_change_authorized"]
        )

    def test_fresh_cells_pass_calibrated_numeric_and_speed_gates(self):
        for case, gpu, expected_speedup in CASES:
            cell = json.loads(
                (EVIDENCE / f"{case}.json").read_text(encoding="ascii")
            )
            self.assertEqual(cell["visible_physical_gpu"], gpu)
            self.assertEqual(cell["source_revision"], SOURCE_REVISION)
            self.assertAlmostEqual(
                cell["timings"]["speedup"], expected_speedup
            )
            self.assertGreaterEqual(cell["timings"]["speedup"], 1.15)
            self.assertTrue(cell["evaluation"]["qualified"])
            self.assertEqual(cell["evaluation"]["reasons"], [])
            candidate = cell["numerical"]["candidate_calibrated"]
            self.assertGreater(
                candidate["candidate_vs_rounded_relative_l2"], 1e-5
            )
            self.assertLess(
                candidate[
                    "relative_l2_error_multiple_over_fp16_rounding"
                ],
                1.001,
            )
            self.assertLess(
                candidate["max_abs_error_multiple_over_fp16_rounding"],
                1.001,
            )
            self.assertLess(
                cell["numerical"]["candidate_lse_relative_l2"], 4e-8
            )
            self.assertEqual(
                cell["numerical"]["candidate_repeat"],
                {"lse_exact": True, "output_exact": True},
            )


if __name__ == "__main__":
    unittest.main()
