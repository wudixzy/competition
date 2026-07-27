from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs" / "experiments" / "evidence"
    / "M1_65_REQUAL_M1_69_20260728"
)
EXPECTED_SHA256 = {
    "benchmark.json":
        "318aa20bc7799cb601b03059ae91e3772be7fe6583b0955c99dc8e10d8bd87cb",
    "fatal_scan.txt":
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "overall.rc":
        "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "preflight_after.json":
        "05794a4e890c93ded859cfe68a320641f72da4884c1ca1b49d4f3f7959e324d7",
    "preflight_before.json":
        "05794a4e890c93ded859cfe68a320641f72da4884c1ca1b49d4f3f7959e324d7",
    "preflight_comparison.json":
        "5d09848a67c056014aa0410eae6484b0cc273a9a6366e1633410244632e650c9",
    "qualification.json":
        "d4a372cf24798ee885353ed3261b4d71b6d48d1a62576671caf00bcc62ddc637",
    "runner_status.json":
        "e6ecb1bb8f721d45fd7f45e68934cb6f48526242c74bf86629c99d1cc21586d2",
    "runtime_identity.json":
        "b664b8050f79f3f6f8d3cb1a7613e20ca775e2390ae7dba75150dd7e25942081",
    "service_postflight.json":
        "55bca6062f652cb97df087df040e2ce21d9d9d22b39ac28fa55457460b90c2b6",
    "timeout_scan.txt":
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}


def load(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


class M165RequalM169EvidenceTest(unittest.TestCase):

    def test_evidence_hashes_and_manifest_are_bound(self):
        manifest = {}
        for line in (EVIDENCE / "SHA256SUMS").read_text(
                encoding="utf-8").splitlines():
            digest, name = line.split(maxsplit=1)
            manifest[name] = digest
        self.assertEqual(manifest, EXPECTED_SHA256)
        self.assertEqual(
            {path.name for path in EVIDENCE.iterdir()},
            set(EXPECTED_SHA256) | {"SHA256SUMS"},
        )
        for name, expected in EXPECTED_SHA256.items():
            actual = hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, name)

    def test_component_is_bit_exact_and_faster_on_current_overlay(self):
        benchmark = load("benchmark.json")
        qualification = load("qualification.json")
        self.assertEqual(benchmark["sequence"]["steps"], 500)
        self.assertEqual(benchmark["sequence"]["finite_steps"], 500)
        self.assertEqual(benchmark["sequence"]["mapped_exact_steps"], 500)
        self.assertEqual(
            benchmark["sequence"]["normalized_exact_steps"], 500)
        self.assertEqual(
            benchmark["sequence"]["max_mapped_relative_l2"], 0.0)
        self.assertEqual(
            benchmark["sequence"]["max_normalized_relative_l2"], 0.0)
        for fixed in benchmark["fixed"].values():
            self.assertTrue(fixed["mapped"]["exact"])
            self.assertTrue(fixed["normalized"]["exact"])
        timings = benchmark["timings"]
        self.assertGreaterEqual(
            timings["candidate"]["speedup_vs_reference"], 1.25)
        self.assertLess(
            timings["candidate"]["median_ms"],
            timings["reference"]["median_ms"],
        )
        self.assertTrue(qualification["component_qualified"])
        self.assertFalse(qualification["production_promotion_authorized"])
        self.assertEqual(qualification["reasons"], [])

    def test_runner_cleanup_and_gpu_state_qualify(self):
        status = load("runner_status.json")
        comparison = load("preflight_comparison.json")
        postflight = load("service_postflight.json")
        self.assertEqual(status["returncode"], 0)
        self.assertEqual(status["last_stage"], "completed")
        self.assertEqual(
            status["source_revision"],
            "37001edff643d98bf41bf4a52e0a145329003315",
        )
        self.assertTrue(all(value == 0 for value in status["gates"].values()))
        self.assertFalse(status["production_promotion_authorized"])
        self.assertTrue(comparison["qualified"])
        self.assertEqual(
            comparison["stages"][1]["free_memory_drop_from_first_bytes"],
            {"1": 0},
        )
        self.assertTrue(postflight["qualified"])
        self.assertEqual(postflight["settling"]["final_clean_streak"], 3)
        self.assertEqual(postflight["api_server_pids"], [])
        self.assertEqual(postflight["worker_pids"], [])
        self.assertEqual(postflight["gpu_processes"], [])
        self.assertEqual((EVIDENCE / "overall.rc").read_text().strip(), "0")
        self.assertEqual((EVIDENCE / "fatal_scan.txt").read_bytes(), b"")
        self.assertEqual((EVIDENCE / "timeout_scan.txt").read_bytes(), b"")


if __name__ == "__main__":
    unittest.main()
