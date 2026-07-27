from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs" / "experiments" / "evidence"
    / "M1_71_HYBRID_MOE_EXACT_TAIL_20260728"
)
MANIFEST_SHA256 = (
    "883608475cda54213e6cbd01e0a0d552c640ef0c477d3ccb511a10ecc95445b8"
)


def load(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


class M171HybridMoeExactTailEvidenceTest(unittest.TestCase):

    def test_manifest_binds_every_evidence_file(self):
        manifest_path = EVIDENCE / "SHA256SUMS"
        self.assertEqual(
            hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            MANIFEST_SHA256,
        )
        manifest = {}
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            digest, relative = line.split(maxsplit=1)
            manifest[relative.removeprefix("./")] = digest
        actual_files = {
            path.relative_to(EVIDENCE).as_posix()
            for path in EVIDENCE.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        }
        self.assertEqual(set(manifest), actual_files)
        for relative, expected in manifest.items():
            actual = hashlib.sha256(
                (EVIDENCE / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, relative)

    def test_performance_passes_but_numerics_reject_candidate(self):
        benchmark = load("benchmark.json")
        qualification = load("qualification.json")
        runner = load("runner_status.json")

        self.assertEqual(
            runner["source_revision"],
            "6b2eac5cabb0d84d7e44bb1edd893c75229026e3",
        )
        self.assertEqual(runner["returncode"], 1)
        self.assertEqual(runner["gates"]["benchmark"], 0)
        self.assertEqual(runner["gates"]["qualification"], 1)
        self.assertFalse(runner["qualified"])

        observed = qualification["observed"]
        limits = qualification["limits"]
        self.assertGreaterEqual(
            observed["fixed_speedup"], limits["speedup"])
        self.assertGreaterEqual(
            observed["routed_speedup"], limits["speedup"])
        self.assertGreaterEqual(
            observed["routed_saving_ms"], limits["saving_ms"])
        self.assertLessEqual(
            observed["direct_w13_relative_l2"], limits["relative_l2"])
        self.assertGreater(
            observed["fixed_relative_l2"], limits["relative_l2"])
        self.assertGreater(
            observed["sequence_relative_l2"], limits["relative_l2"])
        self.assertGreater(
            observed["sequence_max_step_relative_l2"],
            limits["relative_l2"],
        )
        self.assertEqual(observed["sequence_exact_steps"], 50)
        sequence = benchmark["sequence"]["hybrid_exact_tail"]
        self.assertEqual(sequence["steps"], 500)
        self.assertEqual(sequence["finite_steps"], 500)
        self.assertFalse(qualification["component_qualified"])
        self.assertFalse(qualification["production_promotion_authorized"])

    def test_cleanup_and_gpu_state_qualify(self):
        runner = load("runner_status.json")
        postflight = load("service_postflight.json")
        comparison = load("preflight_comparison.json")

        for gate in (
                "cleanup", "fatal_scan", "preflight_after",
                "preflight_before", "preflight_comparison",
                "runtime_pair", "service_postflight", "timeout_scan"):
            self.assertEqual(runner["gates"][gate], 0, gate)
        self.assertTrue(postflight["qualified"])
        self.assertEqual(postflight["api_server_pids"], [])
        self.assertEqual(postflight["worker_pids"], [])
        self.assertEqual(postflight["gpu_processes"], [])
        self.assertEqual(postflight["settling"]["final_clean_streak"], 3)
        self.assertTrue(comparison["qualified"])
        self.assertEqual(
            comparison["stages"][1]["free_memory_drop_from_first_bytes"],
            {"1": 0},
        )
        self.assertEqual((EVIDENCE / "fatal_scan.txt").read_bytes(), b"")
        self.assertEqual((EVIDENCE / "timeout_scan.txt").read_bytes(), b"")
        self.assertEqual((EVIDENCE / "cleanup.rc").read_text().strip(), "0")
        self.assertEqual((EVIDENCE / "overall.rc").read_text().strip(), "1")


if __name__ == "__main__":
    unittest.main()
