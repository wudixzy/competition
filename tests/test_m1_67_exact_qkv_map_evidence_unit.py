from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs" / "experiments" / "evidence"
    / "M1_67_EXACT_QKV_MAP"
)
EXPECTED_SHA256 = {
    "benchmark.json":
        "b754a2bab55163837a4d14f748a2427a590c265b3fea5f3fac0a757d0d49711d",
    "qualification.json":
        "3b7f6895823542931350cc6f64fc53959738c0809e089d1b8ad1443a0feec1f2",
    "runner_status.json":
        "7043f2dadd33bfe4dcbd864a754a60f656e6daf13cae2fab5a05cb581c84cad7",
    "runtime_identity.json":
        "5524184e2f64a90c4275b719a3388aef201cd0f81b55f9ff33db01128597e5b0",
    "service_postflight.json":
        "f9dbb8899cb86e3fc185423d1938e45777f13f101062d5e28cf0cb5659b1f144",
    "preflight_comparison.json":
        "5d09848a67c056014aa0410eae6484b0cc273a9a6366e1633410244632e650c9",
}


def load(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


class M167ExactQkvMapEvidenceTests(unittest.TestCase):

    def test_bound_artifact_hashes(self):
        for name, expected in EXPECTED_SHA256.items():
            actual = hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, name)

    def test_numerical_gate_passes_but_absolute_saving_rejects(self):
        benchmark = load("benchmark.json")
        qualification = load("qualification.json")
        self.assertEqual(benchmark["sequence"]["exact_steps"], 1000)
        self.assertEqual(benchmark["sequence"]["finite_steps"], 1000)
        self.assertEqual(benchmark["sequence"]["relative_l2"], 0.0)
        self.assertEqual(benchmark["sequence"]["max_relative_l2"], 0.0)
        self.assertGreater(
            benchmark["timings"]["candidate_speedup"], 1.25)
        self.assertLess(
            benchmark["timings"]["candidate_saving_ms"], 0.02)
        self.assertFalse(qualification["component_qualified"])
        self.assertFalse(
            qualification["production_promotion_authorized"])
        self.assertEqual(
            qualification["reasons"],
            ["q/k/v stage saving is below 0.02 ms/layer"],
        )

    def test_runtime_and_lifecycle_are_bound_and_clean(self):
        runner = load("runner_status.json")
        runtime = load("runtime_identity.json")
        postflight = load("service_postflight.json")
        comparison = load("preflight_comparison.json")
        self.assertEqual(
            runner["source_revision"],
            "79bdd95f69327dd3e165360ad83a0526dc387ccc",
        )
        self.assertEqual(runner["physical_gpu"], 1)
        self.assertTrue(runtime["qualified"])
        self.assertEqual(
            runtime["runtime_tree_sha256"],
            "05d720faf7a6298946b6ab70d0ab73b8e88f0d7b90c7a54cf50c2c19e0273b7b",
        )
        expected_gates = {
            "benchmark": 0,
            "build": 0,
            "cleanup": 0,
            "fatal_scan": 0,
            "preflight_after": 0,
            "preflight_before": 0,
            "preflight_comparison": 0,
            "qualification": 1,
            "runtime_identity": 0,
            "service_postflight": 0,
            "timeout_scan": 0,
        }
        self.assertEqual(runner["gates"], expected_gates)
        self.assertTrue(postflight["qualified"])
        self.assertEqual(postflight["settling"]["final_clean_streak"], 3)
        self.assertTrue(comparison["qualified"])
        for stage in comparison["stages"]:
            self.assertEqual(
                stage["free_memory_drop_from_first_bytes"]["1"], 0)

    def test_rc_files_fail_only_the_frozen_qualification(self):
        expected = {
            "benchmark.rc": "0",
            "build.rc": "0",
            "cleanup.rc": "0",
            "fatal_scan.rc": "0",
            "overall.rc": "1",
            "preflight_after.rc": "0",
            "preflight_before.rc": "0",
            "preflight_comparison.rc": "0",
            "qualification.rc": "1",
            "service_postflight.rc": "0",
            "timeout_scan.rc": "0",
        }
        for name, value in expected.items():
            self.assertEqual(
                (EVIDENCE / name).read_text(encoding="utf-8").strip(),
                value,
                name,
            )


if __name__ == "__main__":
    unittest.main()
