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
    / "M1_158_FP16_QK_DEFAULT_AB_20260730"
)
SOURCE_REVISION = "57effecd29d057f220bdf1e9a54175b9c04a7cba"
CANDIDATE_SHA256 = (
    "36e043f138aa87c635178e4aa6a30af710b87c3f3d7c2a3f1838fc0e365bd368"
)
EXPECTED = (
    ("p90_total_16k_q8176", 1.154297125750411),
    ("p90_total_32k_q8176", 1.165879339809783),
    ("p90_total_64k_q8176", 1.172187613922017),
)


class M1158Fp16QkDefaultEvidenceTest(unittest.TestCase):
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

    def test_failed_numeric_gate_blocks_all_continuation(self):
        runner = json.loads(
            (EVIDENCE / "runner_status.json").read_text(encoding="ascii")
        )
        self.assertFalse(runner["qualified"])
        self.assertEqual(runner["source_revision"], SOURCE_REVISION)
        self.assertEqual(
            runner["candidate_extension_sha256"], CANDIDATE_SHA256
        )
        self.assertTrue(all(runner["lifecycle"].values()))
        self.assertEqual(
            runner["authorization"],
            {
                "long_context_or_quality_authorized": False,
                "main_or_yaml_change_authorized": False,
                "short_tp4_screen_authorized": False,
            },
        )

    def test_default_selector_reproduces_tensor_op_result(self):
        for case, expected_speedup in EXPECTED:
            cell = json.loads(
                (EVIDENCE / f"{case}.json").read_text(encoding="ascii")
            )
            self.assertAlmostEqual(
                cell["timings"]["speedup"], expected_speedup
            )
            self.assertFalse(cell["evaluation"]["qualified"])
            candidate = cell["numerical"]["candidate_vs_reference"]
            self.assertTrue(candidate["finite"])
            self.assertGreater(candidate["output_relative_l2"], 1e-5)
            self.assertLess(candidate["output_relative_l2"], 2e-5)
            self.assertLess(candidate["output_max_abs"], 1e-4)


if __name__ == "__main__":
    unittest.main()
