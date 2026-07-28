#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_m1_88_w13_rounding_guard.sh"


class M188W13RoundingGuardRunnerUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_runner_has_scoped_lifecycle_and_postflight(self) -> None:
        for marker in (
            "set -Eeuo pipefail",
            "umask 077",
            'if [[ "$RUN_ROOT" != /tmp/* ]]',
            "M1-88 runner refuses a dirty source tree",
            "exec_bi100_session.py",
            "read_process_starttime",
            "ACTIVE_SESSION_TOKEN",
            "bi100_stop_process_group",
            '"$ACTIVE_PGID" "$ACTIVE_PID" 60 20',
            'kill -TERM "$ACTIVE_PID"',
            'kill -KILL "$ACTIVE_PID"',
            'wait "$ACTIVE_PID"',
            "trap finish EXIT",
            "cleanup_recorded_bi100_sessions.py",
            "service_postflight_gate.py",
            "bi100_preflight.py",
            "compare_bi100_preflights.py",
            "fatal_scan.rc",
            "timeout_scan.rc",
            "--settle-timeout-s 90 --clean-samples 3",
        ):
            self.assertIn(marker, self.source)

    def test_runner_binds_the_bounded_experiment(self) -> None:
        for marker in (
            "build_corex_moe_w13_rounding_probe.sh",
            "bench_moe_w13_rounding_guard.py",
            "qualify_moe_w13_rounding_guard.py",
            "--seeds 20260716,20260727",
            "--sequence-steps 500",
            '"relative_l2": 1.0e-5',
            '"max_flagged_fraction": 0.05',
            '"max_step_flagged_fraction": 0.10',
            '"sequence_steps_per_seed": 500',
            '"term_grace_s": 60',
        ):
            self.assertIn(marker, self.source)

    def test_runner_keeps_diagnostic_authority_boundary(self) -> None:
        for marker in (
            '"production_runtime_changed": False',
            '"production_promotion_authorized": False',
            '"yaml_change_authorized": False',
            '"main_merge_authorized": False',
        ):
            self.assertIn(marker, self.source)
        for forbidden in (
            "computility-run.yaml",
            "patch_ops.sh",
            "Dockerfile",
            "git push",
            "git checkout",
            "git switch",
            "pkill",
            "killall",
        ):
            self.assertNotIn(forbidden, self.source)

    def test_invalid_gpu_fails_before_runtime_access(self) -> None:
        result = subprocess.run(
            [
                "bash",
                str(RUNNER),
                "not-a-gpu",
                "m1-88-local",
                "/tmp/unused-m1-88",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("non-negative integer", result.stderr)


if __name__ == "__main__":
    unittest.main()
