#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_m1_87_single_gpu_queue.sh"


class M187SingleGpuQueueRunnerUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_runner_has_scoped_lifecycle_and_private_artifacts(self) -> None:
        for marker in (
            "set -Eeuo pipefail",
            "umask 077",
            'if [[ "$RUN_ROOT" != /tmp/* ]]',
            "M1-87 queue refuses a dirty source tree",
            "ACTIVE_CHILD_PID",
            "ACTIVE_CHILD_PGID",
            "ACTIVE_CHILD_STARTTIME",
            "ACTIVE_CHILD_SESSION_TOKEN",
            "read_process_starttime",
            "exec_bi100_session.py",
            "${label}_child_identity.json",
            "bi100_stop_process_group",
            '"$ACTIVE_CHILD_PGID" "$ACTIVE_CHILD_PID" 900 20 \\\n'
            '            "$ACTIVE_CHILD_STARTTIME" \\\n'
            '            "$ACTIVE_CHILD_SESSION_TOKEN"',
            'kill -TERM "$ACTIVE_CHILD_PID"',
            'kill -KILL "$ACTIVE_CHILD_PID"',
            'wait "$ACTIVE_CHILD_PID"',
            "trap finish EXIT",
            "trap '' TERM INT",
            "cleanup_recorded_bi100_sessions.py",
            "service_recovery.json",
            "--settle-timeout-s 90 --clean-samples 3",
            "--kill-after=90s 240s",
            "scan_fatal_logs",
            "scan_timeout_rcs",
            "find \"$RUN_ROOT\" -type f",
            "124|137|143",
        ):
            self.assertIn(marker, self.source)

    def test_runner_orders_functional_then_audit_then_image_ab(self) -> None:
        diagnostic = self.source.index(
            'CURRENT_STAGE=m1_84\n')
        audit = self.source.index(
            'CURRENT_STAGE=interstage_audit\n')
        image = self.source.index(
            'CURRENT_STAGE=m1_86\n')
        completed = self.source.index(
            'CURRENT_STAGE=completed\n')
        self.assertLess(diagnostic, audit)
        self.assertLess(audit, image)
        self.assertLess(image, completed)
        self.assertIn("run_qwen36_diagnostic_gate.sh", self.source)
        self.assertIn("run_m1_86_multi_image_ab.sh", self.source)
        self.assertIn(
            "qualify_m1_87_single_gpu_queue.py", self.source)

    def test_runner_keeps_diagnostic_authority_boundary(self) -> None:
        self.assertNotIn("computility-run.yaml", self.source)
        self.assertNotIn("git push", self.source)
        self.assertNotIn("git checkout", self.source)
        self.assertNotIn("git switch", self.source)
        self.assertNotIn("pkill", self.source)
        self.assertNotIn("killall", self.source)

    def test_invalid_arguments_fail_before_gpu_or_model_access(self) -> None:
        result = subprocess.run(
            ["bash", str(RUNNER), "contains/a/slash", "/tmp/unused-m1-87"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("short non-sensitive label", result.stderr)

        result = subprocess.run(
            ["bash", str(RUNNER), "m1-87", "/tmp/unused-m1-87"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            env={
                "PATH": "/usr/bin:/bin",
                "DIAGNOSTIC_PORT": "8040",
                "MULTI_IMAGE_PORT": "8040",
            },
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("ports must differ", result.stderr)


if __name__ == "__main__":
    unittest.main()
