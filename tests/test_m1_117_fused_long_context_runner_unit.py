from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_m1_85_admission64_quality_ab.sh"
WRAPPER = ROOT / "scripts/run_m1_117_fused_prefill_long_context_ab.sh"


class M1117FusedLongContextRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = RUNNER.read_text(encoding="utf-8")
        cls.wrapper = WRAPPER.read_text(encoding="utf-8")

    def test_shell_syntax(self) -> None:
        subprocess.run(
            ["bash", "-n", str(RUNNER), str(WRAPPER)],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_wrapper_selects_dedicated_private_variant(self) -> None:
        self.assertIn(
            "BI100_QUALITY_AB_VARIANT=m1-117-fused-prefill-long-context",
            self.wrapper,
        )
        self.assertIn("run_m1_85_admission64_quality_ab.sh", self.wrapper)

    def test_runner_uses_two_fresh_long_context_arms(self) -> None:
        self.assertIn("suite=long-context", self.runner)
        self.assertIn(
            "run_arm control admission64 m1-117-control-fused-off",
            self.runner,
        )
        self.assertIn(
            "run_arm candidate admission64 m1-117-candidate-fused-on",
            self.runner,
        )
        self.assertIn(
            "long-context \"$policy\" \"$restore_mode\" \\\n"
            "                \"$fused_prefill\" lru",
            self.runner,
        )

    def test_runner_keeps_complete_strict_comparator(self) -> None:
        self.assertIn(
            "compare_long_context_quality_reports.py", self.runner)
        self.assertIn(
            '"$RUN_ROOT/control/quality_report.json"', self.runner)
        self.assertIn(
            '"$RUN_ROOT/candidate/quality_report.json"', self.runner)
        self.assertIn(
            "[[ $long_context_comparison_rc -eq 0 ]]", self.runner)

    def test_runner_does_not_enable_m1_116_diagnostic_for_m1_117(self) -> None:
        self.assertIn(
            'if [[ "$QUALITY_AB_VARIANT" == \\\n'
            "            m1-116-fused-prefill-adjudication ]]; then",
            self.runner,
        )
        self.assertNotIn(
            "m1-117-fused-prefill-long-context ]]; then\n"
            "        runner_env+=(",
            self.runner,
        )

    def test_status_never_authorizes_promotion(self) -> None:
        self.assertIn('"performance_authorized": False', self.runner)
        self.assertIn(
            '"default_policy_change_authorized": False', self.runner)
        self.assertIn(
            '"production_promotion_authorized": False', self.runner)


if __name__ == "__main__":
    unittest.main()
