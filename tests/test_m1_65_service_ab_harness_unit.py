from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATOR = ROOT / "scripts/run_m1_65_combined_qk_service_ab.sh"
SERVICE_GATE = ROOT / "scripts/run_quality_service_gate.sh"


class M165ServiceAbHarnessTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.orchestrator = ORCHESTRATOR.read_text(encoding="utf-8")
        cls.service_gate = SERVICE_GATE.read_text(encoding="utf-8")

    def test_same_runner_and_profiles_define_the_only_ab_delta(self):
        self.assertIn(
            "run_arm strict-reference m1-65-control", self.orchestrator)
        self.assertIn(
            "run_arm strict-reference-combined-qk m1-65-candidate",
            self.orchestrator,
        )
        self.assertEqual(
            self.orchestrator.count(
                '"$ROOT/scripts/run_quality_service_gate.sh"'),
            1,
        )
        self.assertIn(
            "compare_gdn_combined_qk_service_ab.py", self.orchestrator)

    def test_decode_arm_reuses_fail_closed_service_cleanup(self):
        self.assertIn("functional|long-context|decode)", self.service_gate)
        self.assertIn("gdn_combined_qk_decode_api.py", self.service_gate)
        self.assertIn(
            'bi100_stop_process_group "$ACTIVE_PGID" "$ACTIVE_PID" 60 20',
            self.service_gate,
        )
        self.assertIn("tests/service_postflight_gate.py", self.service_gate)
        self.assertIn("run_preflight after", self.service_gate)
        self.assertIn("scan_fatal_log", self.service_gate)
        self.assertIn("scan_runner_timeouts", self.service_gate)
        self.assertIn("trap 'exit 143' TERM", self.service_gate)
        self.assertIn("trap finish EXIT", self.service_gate)

    def test_orchestrator_manages_only_its_active_child_group(self):
        self.assertIn("setsid env BI100_QUALITY_KERNEL_PROFILE", self.orchestrator)
        self.assertIn('kill -TERM -- "-$ACTIVE_CHILD_PGID"', self.orchestrator)
        self.assertIn("waited < 120", self.orchestrator)
        self.assertIn('kill -KILL -- "-$ACTIVE_CHILD_PGID"', self.orchestrator)
        self.assertIn("wait \"$ACTIVE_CHILD_PID\"", self.orchestrator)
        self.assertNotIn("pkill", self.orchestrator)
        self.assertIn("orchestrator_postflight", self.orchestrator)

    def test_ab_is_private_and_does_not_change_submission_contract(self):
        self.assertIn(
            "A/B output must use a private /tmp path", self.orchestrator)
        self.assertIn("A/B refuses a dirty source tree", self.orchestrator)
        self.assertNotIn("computility-run.yaml", self.orchestrator)
        self.assertNotIn("git push", self.orchestrator)


if __name__ == "__main__":
    unittest.main()
