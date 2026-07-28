from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_m1_86_multi_image_ab.sh"


class M186MultiImageRunnerUnitTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_fixed_single_delta_arms(self):
        self.assertIn("run_arm control 1 400", self.source)
        self.assertIn("run_arm candidate 2 200", self.source)
        self.assertIn(
            'COMMAND+=(--limit-mm-per-prompt image=2)', self.source)
        self.assertIn(
            '--expected-two-image-status "$expected_status"', self.source)
        self.assertIn("compare_m1_86_multi_image_ab.py", self.source)
        self.assertEqual(
            self.source.count("--limit-mm-per-prompt"), 1,
            "selector should appear only in the candidate command",
        )

    def test_reference_compute_and_capacity_are_fixed(self):
        self.assertIn("--max-model-len 262144", self.source)
        self.assertIn("--tensor-parallel-size 1", self.source)
        self.assertIn("BI100_MOE_COREX_DIRECT_ROUTED=0", self.source)
        self.assertIn("BI100_GDN_COREX_PACKED_DECODE=0", self.source)
        self.assertIn("BI100_GDN_COMBINED_QK_NORM=0", self.source)
        self.assertIn("BI100_GDN_CACHE_POLICY=fine32", self.source)
        self.assertIn("BI100_GDN_RESTORE_MODE=direct", self.source)
        self.assertIn("BI100_HYBRID_KV_ACCOUNTING=full_attention", self.source)
        self.assertIn("check_startup_capacity.py", self.source)
        self.assertIn(
            "qualify_m1_86_multi_image_trace.py", self.source)
        self.assertIn("--block-size 16", self.source)

    def test_exact_current_overlay_and_checkpoint_are_required(self):
        self.assertIn(
            "verify_qwen36_diagnostic_checkpoint.py", self.source)
        self.assertIn(
            "verify_bare_host_runtime_identity.py", self.source)
        self.assertIn(
            "runtime_overlay_identity.json", self.source)
        self.assertIn(
            "M1-86 runner refuses a dirty source tree", self.source)
        self.assertIn(
            '"runtime_source_files_match": overlay.get("qualified")',
            self.source,
        )

    def test_cleanup_is_scoped_graceful_and_fail_closed(self):
        self.assertIn(
            '"$ACTIVE_PGID" "$ACTIVE_PID" 60 20 \\\n'
            '            "$ACTIVE_STARTTIME" "$ACTIVE_SESSION_TOKEN"',
            self.source,
        )
        self.assertIn("ACTIVE_SESSION_TOKEN", self.source)
        self.assertNotIn("pkill", self.source)
        self.assertIn("exec_bi100_session.py", self.source)
        self.assertIn(
            'value.get("pgid") != expected',
            self.source,
        )
        self.assertIn('value.get("sid") != expected', self.source)
        self.assertIn("if [[ $rc -eq 0 ]]; then", self.source)
        self.assertIn("trap 'exit 143' TERM", self.source)
        self.assertIn("trap 'exit 130' INT", self.source)
        self.assertIn("trap finish EXIT", self.source)
        self.assertIn("trap '' TERM INT", self.source)
        self.assertIn("service_postflight_gate.py", self.source)
        self.assertIn(
            "--settle-timeout-s 90 --clean-samples 3", self.source)
        self.assertIn("compare_bi100_preflights.py", self.source)
        self.assertIn("Gloo.*(failed|reset|error)", self.source)
        self.assertIn("NCCL.*(failed|abort|error)", self.source)
        self.assertIn("124|137|143", self.source)
        self.assertIn("-name '*.rc' -print0", self.source)
        self.assertIn(
            "-name '*.log' -o -name '*.stdout' -o -name '*.stderr'",
            self.source,
        )

    def test_startup_and_contract_gates_are_bounded_and_mandatory(self):
        self.assertIn("wait_http_health.py", self.source)
        self.assertIn('--starttime-ticks "$ACTIVE_STARTTIME"', self.source)
        self.assertIn('--timeout-s "$STARTUP_TIMEOUT_S"', self.source)
        self.assertIn('--out "$arm/startup.json"', self.source)
        self.assertNotIn("startup_deadline=$((SECONDS", self.source)
        self.assertIn(
            '"service_contract": rc("service_contract.rc")',
            self.source,
        )
        self.assertIn(
            '"cache_trace": rc("cache_trace.rc")',
            self.source,
        )
        self.assertIn('--expected-gpu "$GPU_INDEX"', self.source)
        self.assertIn(
            '--control-postflight "$RUN_ROOT/control/service_postflight.json"',
            self.source,
        )

    def test_streaming_4xx_and_private_artifacts_are_mandatory(self):
        self.assertIn("umask 077", self.source)
        self.assertIn("qwen36_multi_image_http_gate.py", self.source)
        self.assertIn("summarize_api_4xx_log.py", self.source)
        self.assertIn(
            "RUN_ROOT must use a private /tmp path", self.source)
        self.assertNotIn("computility-run.yaml", self.source)
        self.assertNotIn("git push", self.source)
        self.assertNotIn("git checkout", self.source)
        self.assertNotIn("git switch", self.source)


if __name__ == "__main__":
    unittest.main()
