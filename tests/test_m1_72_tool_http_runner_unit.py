from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_m1_72_tool_http_ab.sh"


class M172ToolHttpRunnerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_two_arms_isolate_request_compatibility_delta(self):
        self.assertIn(
            'run_arm baseline "$CONTROL_RUNTIME_SITE_PACKAGES"',
            self.source,
        )
        self.assertIn(
            'run_arm candidate "$CANDIDATE_RUNTIME_SITE_PACKAGES"',
            self.source,
        )
        self.assertIn('"$CONTROL_REVISION" 400 400 0 "$PORT"',
                      self.source)
        self.assertIn('"$CANDIDATE_REVISION" 200 200 1',
                      self.source)
        self.assertIn("--strict-false-expected-status", self.source)
        self.assertIn("--object-history-expected-status", self.source)
        self.assertIn("compare_qwen36_tool_http_ab.py", self.source)

    def test_each_arm_uses_a_preflighted_distinct_port(self):
        self.assertIn("check_port_available()", self.source)
        self.assertIn('sock.bind(("", int(sys.argv[1])))', self.source)
        self.assertIn('"$CONTROL_REVISION" 400 400 0 "$PORT"',
                      self.source)
        self.assertIn('"$((PORT + 1))"', self.source)
        self.assertIn('"port_preflight": rc("port_preflight.rc")',
                      self.source)
        self.assertIn('"arm_ports": {', self.source)

    def test_compute_and_capacity_contract_are_fixed(self):
        self.assertIn("--max-model-len 262144", self.source)
        self.assertIn("--tensor-parallel-size 1", self.source)
        self.assertIn("BI100_MOE_COREX_DIRECT_ROUTED=0", self.source)
        self.assertIn("BI100_GDN_COREX_PACKED_DECODE=0", self.source)
        self.assertIn("BI100_GDN_COMBINED_QK_NORM=0", self.source)
        self.assertIn("BI100_GDN_CACHE_POLICY=fine32", self.source)
        self.assertIn("BI100_GDN_RESTORE_MODE=direct", self.source)
        self.assertIn("verify_m1_72_runtime_pair.py", self.source)
        self.assertIn("verify_qwen36_diagnostic_checkpoint.py", self.source)

    def test_cleanup_is_scoped_graceful_and_fail_closed(self):
        self.assertIn(
            'bi100_stop_process_group "$ACTIVE_PGID" "$ACTIVE_PID" 60 20',
            self.source,
        )
        self.assertNotIn("pkill", self.source)
        self.assertIn("exec setsid", self.source)
        self.assertIn("trap 'exit 143' TERM", self.source)
        self.assertIn("trap 'exit 130' INT", self.source)
        self.assertIn("trap finish EXIT", self.source)
        self.assertIn("service_postflight_gate.py", self.source)
        self.assertIn("--settle-timeout-s 90 --clean-samples 3", self.source)
        self.assertIn('run_preflight "$arm/preflight_after"', self.source)
        self.assertIn('run_preflight "$RUN_ROOT/final_preflight"',
                      self.source)
        self.assertIn("compare_bi100_preflights.py", self.source)
        self.assertIn("worker.*(died|lost|exited unexpectedly)", self.source)
        self.assertIn("Gloo.*(failed|reset|error)", self.source)
        self.assertIn("NCCL.*(failed|abort|error)", self.source)
        self.assertIn("124|137", self.source)

    def test_outputs_stay_private_and_submission_is_untouched(self):
        self.assertIn(
            "RUN_ROOT must use a private /tmp path", self.source)
        self.assertIn(
            "M1-72 runner refuses a dirty source tree", self.source)
        self.assertNotIn("computility-run.yaml", self.source)
        self.assertNotIn("git push", self.source)
        self.assertNotIn("git checkout", self.source)


if __name__ == "__main__":
    unittest.main()
