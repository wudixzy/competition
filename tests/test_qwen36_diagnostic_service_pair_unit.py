#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
EVIDENCE = ROOT / "docs" / "experiments" / "evidence" \
    / "M1_60_DIAGNOSTIC"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import qualify_qwen36_diagnostic_service_pair as qualify  # noqa: E402


def evidence_arguments() -> dict[str, Path]:
    return {
        "--checkpoint-verify": EVIDENCE / "checkpoint" / "verify.json",
        "--tp1-status": EVIDENCE / "tp1" / "status.json",
        "--tp1-api": EVIDENCE / "tp1" / "api_gate.json",
        "--tp1-prefix": EVIDENCE / "tp1" / "prefix_boundary.json",
        "--tp1-runtime": EVIDENCE / "tp1" / "runtime_install.json",
        "--tp1-manifest": EVIDENCE / "tp1" / "checkpoint_manifest.json",
        "--tp1-preflight-before":
            EVIDENCE / "tp1" / "preflight_before.json",
        "--tp1-preflight-after":
            EVIDENCE / "tp1" / "preflight_after.json",
        "--tp2-status": EVIDENCE / "tp2" / "status.json",
        "--tp2-api": EVIDENCE / "tp2" / "api_gate.json",
        "--tp2-prefix": EVIDENCE / "tp2" / "prefix_boundary.json",
        "--tp2-runtime": EVIDENCE / "tp2" / "runtime_install.json",
        "--tp2-manifest":
            EVIDENCE / "checkpoint" / "diagnostic-checkpoint-manifest.json",
        "--tp2-preflight-before":
            EVIDENCE / "tp2" / "preflight_before.json",
        "--tp2-preflight-after":
            EVIDENCE / "tp2" / "preflight_after.json",
        "--tp2-nccl": EVIDENCE / "tp2" / "nccl_before.json",
        "--gdn-broadcast": EVIDENCE / "tp2" / "gdn_action_broadcast.json",
    }


class Qwen36DiagnosticServicePairUnitTest(unittest.TestCase):
    def _run(
        self,
        output: Path,
        overrides: dict[str, Path] | None = None,
    ) -> tuple[int, dict]:
        arguments = evidence_arguments()
        arguments.update(overrides or {})
        argv = ["qualify_qwen36_diagnostic_service_pair.py"]
        for flag, path in arguments.items():
            argv.extend([flag, str(path)])
        argv.extend(["--out", str(output)])
        with mock.patch.object(sys, "argv", argv), \
                contextlib.redirect_stdout(io.StringIO()):
            rc = qualify.main()
        return rc, json.loads(output.read_text(encoding="utf-8"))

    def test_committed_service_pair_qualifies_without_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rc, report = self._run(Path(directory) / "report.json")
        self.assertEqual(rc, 0)
        self.assertTrue(report["qualified"])
        self.assertTrue(report["checkpoint"]["identical_across_tp"])
        self.assertTrue(report["runtime"]["identical_across_tp"])
        self.assertTrue(report["api"]["response_evidence_exact_across_tp"])
        self.assertTrue(report["prefix_cache"]["exact_across_tp"])
        self.assertFalse(report["semantic_quality_evaluated"])
        self.assertFalse(report["production_promotion_authorized"])
        committed = json.loads(
            (EVIDENCE / "service_pair_qualification.json").read_text(
                encoding="utf-8"))
        self.assertEqual(report, committed)

    def test_changed_tp2_output_digest_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = json.loads(
                evidence_arguments()["--tp2-api"].read_text(
                    encoding="utf-8"))
            value["cases"][1]["evidence"]["cold"]["message_sha256"] = "0" * 64
            changed = root / "api.json"
            changed.write_text(json.dumps(value), encoding="utf-8")
            rc, report = self._run(
                root / "report.json", {"--tp2-api": changed})
        self.assertEqual(rc, 1)
        self.assertIn(
            "TP1/TP2 API response structures or digests differ",
            report["reasons"],
        )

    def test_changed_runtime_tree_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = json.loads(
                evidence_arguments()["--tp2-runtime"].read_text(
                    encoding="utf-8"))
            value["runtime_tree_sha256"] = "0" * 64
            changed = root / "runtime.json"
            changed.write_text(json.dumps(value), encoding="utf-8")
            rc, report = self._run(
                root / "report.json", {"--tp2-runtime": changed})
        self.assertEqual(rc, 1)
        self.assertIn("TP1/TP2 runtime trees differ", report["reasons"])

    def test_gpu_memory_leak_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = json.loads(
                evidence_arguments()["--tp2-preflight-after"].read_text(
                    encoding="utf-8"))
            value["results"][0]["free"] -= 4096
            changed = root / "preflight.json"
            changed.write_text(json.dumps(value), encoding="utf-8")
            rc, report = self._run(
                root / "report.json",
                {"--tp2-preflight-after": changed},
            )
        self.assertEqual(rc, 1)
        self.assertTrue(any(
            "memory was not fully restored" in reason
            for reason in report["reasons"]))

    def test_qualifier_keeps_diagnostic_scope(self) -> None:
        source = (
            ROOT / "tests" / "qualify_qwen36_diagnostic_service_pair.py"
        ).read_text(encoding="utf-8")
        for marker in (
            '"semantic_quality_evaluated": False',
            '"full_model_tp4_evaluated": False',
            '"official_performance_evaluated": False',
            '"production_promotion_authorized": False',
            "source_payload_bytes_compared",
            "missing_restore_fail_fast_source_attested",
        ):
            self.assertIn(marker, source)


if __name__ == "__main__":
    unittest.main()
