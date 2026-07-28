from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_m1_104_admission64_performance_ab.sh"


class M1104Admission64RunnerUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_shell_syntax_and_usage_are_cpu_only(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(RUNNER)], check=False,
            capture_output=True, text=True, timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(
            ["bash", str(RUNNER)], check=False,
            capture_output=True, text=True, timeout=5,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)

    def test_fixed_tp4_contract_and_alternation(self) -> None:
        for marker in (
            "--gpus 0,1,2,3",
            "--expected-gpus 0,1,2,3",
            "BI100_GDN_CACHE_POLICY=\"$policy\"",
            "BI100_GDN_RESTORE_MODE=direct",
            "BI100_HYBRID_KV_ACCOUNTING=full_attention",
            "BI100_CPU_KV_OFFLOAD=0",
            "BI100_ATTN_COREX_FUSED_PREFILL=0",
            "BI100_CACHE_TRACE=1",
            "BI100_KV_EVICTION_POLICY=lru",
            "BI100_MOE_COREX_DIRECT_ROUTED=1",
            "BI100_GDN_COREX_PACKED_DECODE=1",
            "'1 control fine32'",
            "'1 candidate admission64'",
            "'2 candidate admission64'",
            "'2 control fine32'",
            "'3 control fine32'",
            "'3 candidate admission64'",
            "--policy \"$policy\"",
            "--salt-namespace",
            "compare_m1_104_admission64_paired_ab.py",
        ):
            self.assertIn(marker, self.source)

    def test_attested_scoped_cleanup_and_postflight(self) -> None:
        for marker in (
            "exec_bi100_session.py",
            "ACTIVE_SESSION_TOKEN",
            "bi100_stop_process_group",
            '"$ACTIVE_PGID" "$ACTIVE_PID" 60 20',
            'kill -TERM "$ACTIVE_PID"',
            'kill -KILL "$ACTIVE_PID"',
            'wait "$ACTIVE_PID"',
            "cleanup_recorded_bi100_sessions.py",
            "qualify_recorded_session_cleanup.py",
            "service_postflight_gate.py",
            "compare_bi100_preflights.py",
            "fatal_scan.rc",
            "timeout_scan.rc",
            "source_unchanged.rc",
            "trap finish EXIT",
            "trap '' INT TERM",
        ):
            self.assertIn(marker, self.source)

    def test_only_complete_measurements_reach_comparison(self) -> None:
        self.assertNotIn("if [[ $measurement_rc -eq 1 ]]; then", self.source)
        self.assertIn("measurement.json", self.source)
        self.assertIn('[[ $rc -eq 0 ]] || exit 1', self.source)
        self.assertIn('--control "$RUN_ROOT/pair1_control/measurement.json"', self.source)
        self.assertIn('--candidate "$RUN_ROOT/pair3_candidate/measurement.json"', self.source)
        self.assertIn("write_arm_status", self.source)
        self.assertIn("runner_status.json", self.source)

    def test_promotion_boundary_is_closed(self) -> None:
        for marker in (
            "'full_quality_m1_85_authorized'",
            "'official_style_replay_authorized': False",
            "'production_promotion_authorized': False",
            "'yaml_change_authorized': False",
            "'main_merge_authorized': False",
            "'official_881_evaluated': False",
            "'full_model_quality_suite_evaluated': False",
        ):
            self.assertIn(marker, self.source)
        for forbidden in ("pkill", "killall", "git push", "git checkout", "git switch"):
            self.assertNotIn(forbidden, self.source)


if __name__ == "__main__":
    unittest.main()
