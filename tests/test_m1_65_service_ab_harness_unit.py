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
        self.assertIn(
            'source "$ROOT/scripts/lib/process_group.sh"',
            self.orchestrator,
        )
        self.assertIn("CHILD_TERM_GRACE_S=900", self.orchestrator)
        self.assertIn(
            'bi100_stop_process_group \\\n'
            '        "$ACTIVE_CHILD_PGID" "$ACTIVE_CHILD_PID"',
            self.orchestrator,
        )
        self.assertNotIn("pkill", self.orchestrator)
        self.assertIn("orchestrator_postflight", self.orchestrator)

    def test_orchestrator_postflight_is_fail_closed_and_complete(self):
        for gate in (
                "orchestrator_cleanup",
                "orchestrator_postflight",
                "orchestrator_preflight_after",
                "orchestrator_fatal_scan",
                "orchestrator_timeout_scan"):
            self.assertIn(f'"{gate}"', self.orchestrator)
            self.assertIn(f"$RUN_ROOT/{gate}.rc", self.orchestrator)
        self.assertIn("tests/service_postflight_gate.py", self.orchestrator)
        self.assertIn("tests/bi100_preflight.py", self.orchestrator)
        self.assertIn("--gpus 0,1,2,3", self.orchestrator)
        self.assertIn("Gloo.*(failed|reset|error)", self.orchestrator)
        self.assertIn(
            "worker.*(died|lost|exited unexpectedly)",
            self.orchestrator,
        )
        self.assertIn("124|137", self.orchestrator)
        self.assertIn("trap 'exit 143' TERM", self.orchestrator)
        self.assertIn("trap 'exit 130' INT", self.orchestrator)
        self.assertIn("trap finish EXIT", self.orchestrator)
        cleanup = self.orchestrator.index("stop_active_child\n")
        process_scan = self.orchestrator.index("run_orchestrator_postflight\n")
        gpu_preflight = self.orchestrator.index("run_orchestrator_preflight\n")
        fatal_scan = self.orchestrator.index("scan_orchestrator_fatal_logs\n")
        timeout_scan = self.orchestrator.index(
            "scan_orchestrator_timeouts\n")
        self.assertLess(cleanup, process_scan)
        self.assertLess(process_scan, gpu_preflight)
        self.assertLess(gpu_preflight, fatal_scan)
        self.assertLess(fatal_scan, timeout_scan)

    def test_ab_is_private_and_does_not_change_submission_contract(self):
        self.assertIn(
            "A/B output must use a private /tmp path", self.orchestrator)
        self.assertIn("A/B refuses a dirty source tree", self.orchestrator)
        self.assertNotIn("computility-run.yaml", self.orchestrator)
        self.assertNotIn("git push", self.orchestrator)


if __name__ == "__main__":
    unittest.main()
