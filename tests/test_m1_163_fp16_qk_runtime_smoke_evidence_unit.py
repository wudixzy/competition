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
    / "M1_163_FP16_QK_RUNTIME_SMOKE_20260730"
)
SOURCE_REVISION = "3e2ccd4dd5f2f19d998e1582efa751c8a89f6728"
EXTENSION_SHA256 = (
    "94ea8fc3862eae7900bfe0decc913774e06a303767feb1ded16592a0b35ce0f3"
)


class M1163Fp16QkRuntimeSmokeEvidenceTest(unittest.TestCase):
    def test_manifest_authenticates_the_privacy_safe_report(self):
        digest, name = (
            (EVIDENCE / "SHA256SUMS")
            .read_text(encoding="ascii")
            .strip()
            .split("  ", 1)
        )
        self.assertEqual(name, "./production_65k_q8176.json")
        self.assertEqual(
            hashlib.sha256(
                (EVIDENCE / name.removeprefix("./")).read_bytes()
            ).hexdigest(),
            digest,
        )

    def test_runtime_module_loads_and_runs_the_fixed_65k_shape(self):
        report = json.loads(
            (EVIDENCE / "production_65k_q8176.json")
            .read_text(encoding="ascii")
        )
        self.assertEqual(report["source_commit"], SOURCE_REVISION)
        self.assertEqual(
            report["extension"]["sha256"], EXTENSION_SHA256)
        self.assertEqual(report["visible_physical_gpu"], 1)
        self.assertEqual(report["context_len"], 65536)
        self.assertEqual(report["query_len"], 8176)
        self.assertTrue(report["numerical"]["finite"])
        self.assertLess(report["numerical"]["lse_relative_l2"], 4e-8)
        self.assertGreater(report["timings"]["speedup"], 2.6)

    def test_legacy_gate_result_does_not_authorize_promotion(self):
        report = json.loads(
            (EVIDENCE / "production_65k_q8176.json")
            .read_text(encoding="ascii")
        )
        self.assertFalse(report["evaluation"]["qualified"])
        self.assertGreater(
            report["numerical"]["output_relative_l2"], 1e-5)
        self.assertLess(
            report["numerical"]["output_relative_l2"], 2e-5)
        self.assertTrue(all(
            value is False
            for value in report["authorization"].values()
        ))


if __name__ == "__main__":
    unittest.main()
