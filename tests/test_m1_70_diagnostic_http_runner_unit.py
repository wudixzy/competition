from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_m1_70_diagnostic_http_ab.sh"


class M170DiagnosticHttpRunnerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_three_arms_isolate_protocol_and_image_limit_deltas(self):
        self.assertIn(
            'run_arm baseline_default "$CONTROL_RUNTIME_SITE_PACKAGES"',
            self.source,
        )
        self.assertIn(
            'run_arm candidate_default "$CANDIDATE_RUNTIME_SITE_PACKAGES"',
            self.source,
        )
        self.assertIn(
            'run_arm candidate_image2 "$CANDIDATE_RUNTIME_SITE_PACKAGES"',
            self.source,
        )
        self.assertIn('"$CONTROL_REVISION" 400 1 0', self.source)
        self.assertIn('"$CANDIDATE_REVISION" 200 1 1', self.source)
        self.assertIn('"$CANDIDATE_REVISION" 200 2 1', self.source)
        self.assertIn(
            "--multiple-system-parts-expected-status", self.source)
        self.assertIn("--limit-mm-per-prompt image=2", self.source)
        self.assertIn("compare_qwen36_compat_http_ab.py", self.source)

    def test_reference_compute_and_capacity_contract_are_fixed(self):
        self.assertIn("--max-model-len 262144", self.source)
        self.assertIn("--tensor-parallel-size 1", self.source)
        self.assertIn("BI100_MOE_COREX_DIRECT_ROUTED=0", self.source)
        self.assertIn("BI100_GDN_COREX_PACKED_DECODE=0", self.source)
        self.assertIn("BI100_GDN_COMBINED_QK_NORM=0", self.source)
        self.assertIn("BI100_GDN_CACHE_POLICY=fine32", self.source)
        self.assertIn("BI100_GDN_RESTORE_MODE=direct", self.source)
        self.assertIn("verify_m1_70_runtime_pair.py", self.source)
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
        self.assertIn("run_preflight \"$arm/preflight_after\"", self.source)
        self.assertIn("run_preflight \"$RUN_ROOT/final_preflight\"", self.source)
        self.assertIn("compare_bi100_preflights.py", self.source)
        self.assertIn("worker.*(died|lost|exited unexpectedly)", self.source)
        self.assertIn("Gloo.*(failed|reset|error)", self.source)
        self.assertIn("NCCL.*(failed|abort|error)", self.source)
        self.assertIn("124|137", self.source)

    def test_outputs_stay_private_and_submission_is_untouched(self):
        self.assertIn(
            "RUN_ROOT must use a private /tmp path", self.source)
        self.assertIn(
            "M1-70 runner refuses a dirty source tree", self.source)
        self.assertNotIn("computility-run.yaml", self.source)
        self.assertNotIn("git push", self.source)
        self.assertNotIn("git checkout", self.source)


if __name__ == "__main__":
    unittest.main()
