from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_m1_99_fused_prefill_service_ab.sh"


class M199FusedPrefillRunnerUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_runner_has_scoped_tp4_lifecycle(self) -> None:
        for marker in (
            "set -Eeuo pipefail",
            "umask 077",
            "exec_bi100_session.py",
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
            "service_postflight_gate.py",
            "--settle-timeout-s 90 --clean-samples 3",
            "bi100_preflight.py",
            "compare_bi100_preflights.py",
            "fatal_scan.rc",
            "timeout_scan.rc",
        ):
            self.assertIn(marker, self.source)

    def test_runner_uses_fixed_three_pair_alternating_design(self) -> None:
        for marker in (
            '"1 control 0"',
            '"1 candidate 1"',
            '"2 candidate 1"',
            '"2 control 0"',
            '"3 control 0"',
            '"3 candidate 1"',
            "--targets 65536,235000 --max-tokens 32",
            "m1-99-pair-${pair}-20260728",
            "compare_m1_99_fused_prefill_paired_ab.py",
            "path=corex_split4",
        ):
            self.assertIn(marker, self.source)

    def test_runner_changes_only_private_fused_selector_between_arms(self):
        self.assertIn(
            'BI100_ATTN_COREX_FUSED_PREFILL="$selector"',
            self.source,
        )
        for marker in (
            "BI100_GDN_CACHE_POLICY=admission64",
            "BI100_GDN_RESTORE_MODE=hybrid64",
            "GDN cache; policy=admission64 restore=hybrid64",
            "BI100_HYBRID_KV_ACCOUNTING=full_attention",
            "BI100_CPU_KV_OFFLOAD=0",
            "BI100_MOE_COREX_DIRECT_ROUTED=1",
            "BI100_GDN_COREX_PACKED_DECODE=1",
        ):
            self.assertIn(marker, self.source)
        self.assertNotIn(
            "GDN cache; policy=admission64 restore=direct",
            self.source,
        )

    def test_runner_keeps_promotion_boundary(self) -> None:
        for marker in (
            '"official_style_replay_authorized": False',
            '"production_promotion_authorized": False',
            '"yaml_change_authorized": False',
            '"main_merge_authorized": False',
            '"official_881_evaluated": False',
            '"full_model_quality_suite_evaluated": False',
        ):
            self.assertIn(marker, self.source)
        for forbidden in (
            "git push",
            "git checkout",
            "git switch",
            "pkill",
            "killall",
        ):
            self.assertNotIn(forbidden, self.source)

    def test_busy_port_probe_is_quiet(self) -> None:
        self.assertIn(
            "python3 - <<'PY' >/dev/null 2>&1\nimport socket",
            self.source,
        )

    def test_invalid_invocation_fails_before_runtime_access(self) -> None:
        result = subprocess.run(
            ["bash", str(RUNNER)],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)


if __name__ == "__main__":
    unittest.main()
