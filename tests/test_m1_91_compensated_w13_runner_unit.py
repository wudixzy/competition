#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_m1_91_compensated_w13.sh"


class M191CompensatedW13RunnerUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_runner_has_attested_lifecycle_and_postflight(self) -> None:
        for marker in (
            "set -Eeuo pipefail",
            "umask 077",
            'if [[ "$RUN_ROOT" != /tmp/* ]]',
            "M1-91 runner refuses a dirty source tree",
            "exec_bi100_session.py",
            "read_process_starttime",
            "ACTIVE_SESSION_TOKEN",
            "bi100_stop_process_group",
            '"$ACTIVE_PGID" "$ACTIVE_PID" 60 20',
            'kill -TERM "$ACTIVE_PID"',
            'kill -KILL "$ACTIVE_PID"',
            'wait "$ACTIVE_PID"',
            "trap finish EXIT",
            "trap '' INT TERM",
            "cleanup_recorded_bi100_sessions.py",
            "qualify_recorded_session_cleanup.py",
            "scoped_cleanup_clean.rc",
            "source_unchanged.rc",
            "service_postflight_gate.py",
            "bi100_preflight.py",
            "compare_bi100_preflights.py",
            "fatal_scan.rc",
            "timeout_scan.rc",
            "--settle-timeout-s 90 --clean-samples 3",
        ):
            self.assertIn(marker, self.source)

    def test_runner_binds_the_fixed_single_gpu_experiment(self) -> None:
        for marker in (
            "build_corex_moe_compensated_w13.sh",
            "bench_moe_compensated_w13.py",
            "qualify_moe_compensated_w13.py",
            "corex_moe_direct_routed.so",
            "--seeds 20260716,20260727",
            "--sequence-steps 500",
            "--warmup 30",
            "--iterations 300",
            "--repeats 9",
            '"relative_l2": 1.0e-5',
            '"fixed_speedup": 1.5',
            '"routed_speedup": 1.25',
            '"sequence_steps_per_seed": 500',
            '"term_grace_s": 60',
            '"kill_grace_s": 20',
            '"complete_token_scan_required": True',
            '"schema": "bi100-m1-91-compensated-w13-runner-v1"',
        ):
            self.assertIn(marker, self.source)

    def test_runner_binds_benchmark_and_extension_artifacts(self) -> None:
        for marker in (
            "artifact_binding.json",
            "candidate_extension_sha256",
            "direct_extension_sha256",
            "report_sha256",
            "benchmark_extensions.get(\"candidate_sha256\")",
            "benchmark_extensions.get(\"direct_sha256\")",
        ):
            self.assertIn(marker, self.source)

    def test_runner_keeps_probe_authority_boundary(self) -> None:
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
                "m1-91-local",
                "/tmp/unused-m1-91",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("non-negative integer", result.stderr)

    def test_recovery_and_scans_fail_closed(self) -> None:
        self.assertIn(
            '--expected-identity "$RUN_ROOT/build_process_identity.json"',
            self.source,
        )
        self.assertIn(
            '--expected-identity "$RUN_ROOT/benchmark_process_identity.json"',
            self.source,
        )
        self.assertIn("-name '*.stdout'", self.source)
        self.assertIn("-name '*.stderr'", self.source)
        self.assertIn("-name '*.rc'", self.source)
        self.assertIn("124|137|143", self.source)
        self.assertIn("cleanup_clean_rc -ne 0", self.source)
        self.assertIn("source_rc -ne 0", self.source)


if __name__ == "__main__":
    unittest.main()
