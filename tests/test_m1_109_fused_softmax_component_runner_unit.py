from __future__ import annotations

from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_m1_109_fused_softmax_component_ab.sh"


class M1109FusedSoftmaxComponentRunnerUnitTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_runner_fixes_one_production_case_per_gpu(self) -> None:
        for marker in (
            "production_dense_q8176",
            "production_65k_q8176",
            "production_128k_q8176",
            "production_235k_q5616",
            'for gpu in 0 1 2 3',
            'CUDA_VISIBLE_DEVICES="$GPU"',
            "bench_m1_55_production_prefill.py",
        ):
            self.assertIn(marker, self.source)

    def test_runner_compares_binaries_on_the_same_gpu(self) -> None:
        self.assertIn('"$INSTANCE" "$gpu" "${CASES[$gpu]}"', self.source)
        self.assertIn("median old/new speedup", self.source)
        self.assertIn(
            'printf \'%s\\n\' "$comparison_rc" > "$RUN_ROOT/comparison.rc"',
            self.source,
        )
        self.assertIn('exit "$comparison_rc"', self.source)
        self.assertIn('"tp4_service_experiment_authorized": not reasons',
                      self.source)
        self.assertIn('"main_or_yaml_change_authorized": False', self.source)

    def test_runner_uses_scoped_graceful_cleanup(self) -> None:
        for marker in (
            "exec_bi100_session.py",
            "cleanup_recorded_bi100_sessions.py",
            "qualify_recorded_session_cleanup.py",
            "session_recovery_clean.rc",
            "trap finish EXIT",
            "timeout --foreground --signal=TERM --kill-after=60s",
            "service_postflight_gate.py",
            "bi100_preflight.py",
            "compare_bi100_preflights.py",
            "fatal_scan.rc",
            "timeout_scan.rc",
            "runner_status.json",
        ):
            self.assertIn(marker, self.source)
        self.assertNotIn("setsid ", self.source)
        self.assertNotIn("pkill", self.source)
        self.assertNotIn("killall", self.source)

    def test_embedded_python_is_syntactically_valid(self) -> None:
        blocks = re.findall(
            r"<<'PY'\n(.*?)\nPY", self.source, flags=re.DOTALL)
        self.assertGreaterEqual(len(blocks), 3)
        for index, block in enumerate(blocks):
            compile(block, f"<runner-heredoc-{index}>", "exec")

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
