from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs" / "experiments" / "evidence"
    / "M1_70_DIAGNOSTIC_HTTP_V2_PORT_CONFLICT_20260728"
)
MANIFEST_SHA256 = (
    "edc09f42e725504be16fb614323e8229f2b92aa5f2f9227998d3686502ccad1c"
)


def load(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


class M170DiagnosticHttpV2PortConflictEvidenceTest(unittest.TestCase):

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

    def test_baseline_passed_and_candidate_was_not_evaluated(self):
        runner = load("runner_status.json")
        baseline = load("baseline_default/status.json")
        baseline_probe = load("baseline_default/probe.json")
        candidate = load("candidate_default/status.json")
        diagnosis = load("diagnosis.json")

        self.assertEqual(
            runner["source_revision"],
            "b51c4227e7b28081030ef4d0a17c2143aa8e051a",
        )
        self.assertEqual(runner["returncode"], 1)
        self.assertFalse(runner["qualified"])
        self.assertTrue(baseline["qualified"])
        self.assertEqual(baseline_probe["case_count"], 8)
        self.assertTrue(all(case["ok"] for case in baseline_probe["cases"]))
        self.assertFalse(candidate["qualified"])
        self.assertEqual(candidate["gates"]["startup"], 1)
        self.assertIsNone(candidate["artifact_sha256"]["probe"])

        self.assertTrue(diagnosis["qualified"])
        self.assertEqual(
            diagnosis["classification"], "runner_arm_port_reuse")
        self.assertTrue(
            diagnosis["candidate"]["address_in_use_marker_present"])
        self.assertFalse(
            diagnosis["candidate"]["model_construction_started"])
        self.assertFalse(
            diagnosis["scope"]["candidate_behavior_evaluated"])
        self.assertTrue(
            all(not value for value in diagnosis["privacy"].values()))

    def test_cleanup_and_gpu_state_remained_valid(self):
        candidate_postflight = load(
            "candidate_default/service_postflight.json")
        candidate_preflight = load(
            "candidate_default/preflight_comparison.json")
        final_postflight = load("final_postflight.json")
        final_preflight = load("final_preflight_comparison.json")
        diagnosis = load("diagnosis.json")

        for postflight in (candidate_postflight, final_postflight):
            self.assertTrue(postflight["qualified"])
            self.assertEqual(postflight["api_server_pids"], [])
            self.assertEqual(postflight["worker_pids"], [])
            self.assertEqual(postflight["gpu_processes"], [])
            self.assertEqual(
                postflight["settling"]["final_clean_streak"], 3)
        for preflight in (candidate_preflight, final_preflight):
            self.assertTrue(preflight["qualified"])
            self.assertEqual(
                preflight["stages"][1]["free_memory_drop_from_first_bytes"],
                {"1": 0},
            )
        self.assertTrue(diagnosis["cleanup"]["fatal_scan_empty"])
        self.assertTrue(diagnosis["cleanup"]["timeout_scan_empty"])
        self.assertEqual((EVIDENCE / "fatal_scan.txt").read_bytes(), b"")
        self.assertEqual((EVIDENCE / "timeout_scan.txt").read_bytes(), b"")


if __name__ == "__main__":
    unittest.main()
