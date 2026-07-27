from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs" / "experiments" / "evidence"
    / "M1_70_DIAGNOSTIC_HTTP_V1_NEGATIVE_20260728"
)
MANIFEST_SHA256 = (
    "a61c7b38e20374168c780a93e09567e8cac306d9c9ae6111987145221dc17b50"
)


def load(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


class M170DiagnosticHttpV1NegativeEvidenceTest(unittest.TestCase):

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

    def test_negative_result_is_a_harness_contract_error(self):
        runner = load("runner_status.json")
        arm = load("baseline_default/status.json")
        probe = load("baseline_default/probe.json")
        access = load("baseline_default/http_access_summary.json")

        self.assertEqual(
            runner["source_revision"],
            "4b78815d504248adfd2059ca613d1422e7fc6d97",
        )
        self.assertEqual(runner["returncode"], 1)
        self.assertFalse(runner["qualified"])
        self.assertEqual(arm["gates"]["probe"], 1)
        self.assertEqual(
            [case["name"] for case in probe["cases"] if not case["ok"]],
            ["single_system_text_parts"],
        )

        status_by_name = {
            request["name"]: request["http_status"]
            for request in access["requests"]
        }
        self.assertEqual(status_by_name["single_system_text_parts"], 200)
        self.assertEqual(status_by_name["multiple_system_text_parts"], 400)
        self.assertTrue(
            access["diagnosis"]["v1_harness_expectation_was_wrong"])
        self.assertTrue(
            access["diagnosis"]["baseline_multi_system_defect_reproduced"])
        self.assertEqual(access["status_counts"], {"200": 5, "400": 2})
        self.assertTrue(
            all(not value for value in access["privacy"].values()))

    def test_runtime_checkpoint_and_cleanup_are_valid(self):
        runtime = load("runtime_pair.json")
        checkpoint = load("checkpoint_verify.json")
        arm_preflight = load("baseline_default/preflight_comparison.json")
        arm_postflight = load("baseline_default/service_postflight.json")
        final_preflight = load("final_preflight_comparison.json")
        final_postflight = load("final_postflight.json")

        self.assertTrue(runtime["qualified"])
        self.assertEqual(
            runtime["observed_runtime_file_delta"],
            ["api_server", "protocol"],
        )
        self.assertTrue(checkpoint["qualified"])
        self.assertEqual(checkpoint["layer_count"], 4)
        self.assertEqual(checkpoint["weight_payload_bytes"], 11345363552)
        for report in (
                arm_preflight, arm_postflight,
                final_preflight, final_postflight):
            self.assertTrue(report["qualified"])
        self.assertEqual(
            arm_preflight["stages"][1]["free_memory_drop_from_first_bytes"],
            {"1": 0},
        )
        self.assertEqual(
            final_preflight["stages"][1]["free_memory_drop_from_first_bytes"],
            {"1": 0},
        )
        for postflight in (arm_postflight, final_postflight):
            self.assertEqual(postflight["api_server_pids"], [])
            self.assertEqual(postflight["worker_pids"], [])
            self.assertEqual(postflight["gpu_processes"], [])
            self.assertEqual(
                postflight["settling"]["final_clean_streak"], 3)
        self.assertEqual((EVIDENCE / "fatal_scan.txt").read_bytes(), b"")
        self.assertEqual((EVIDENCE / "timeout_scan.txt").read_bytes(), b"")
        self.assertEqual(
            (EVIDENCE / "baseline_default" / "fatal_scan.txt").read_bytes(),
            b"",
        )


if __name__ == "__main__":
    unittest.main()
