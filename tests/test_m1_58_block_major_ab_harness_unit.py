import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
HARNESS = ROOT / "scripts/run_m1_58_block_major_kv_ab.sh"


class M158BlockMajorAbHarnessTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.source = HARNESS.read_text(encoding="utf-8")

    def test_two_arms_differ_only_by_fixed_selector(self):
        self.assertIn("export BI100_CPU_KV_OFFLOAD=1", self.source)
        self.assertIn(
            "export BI100_HYBRID_KV_ACCOUNTING=full_attention",
            self.source,
        )
        self.assertIn("export BI100_GDN_CACHE_POLICY=admission64", self.source)
        self.assertIn("export BI100_GDN_RESTORE_MODE=direct", self.source)
        self.assertIn("export BI100_CACHE_TRACE=0", self.source)
        self.assertIn(
            "export BI100_BLOCK_MAJOR_CPU_KV_TRACE=0", self.source)
        self.assertIn("run_arm control 0", self.source)
        self.assertIn("run_arm candidate 1", self.source)
        self.assertIn(
            'BI100_BLOCK_MAJOR_CPU_KV="$selector"', self.source)
        self.assertNotIn("NUM_GPU_BLOCKS_OVERRIDE=", self.source)

    def test_pressure_workload_and_gates_are_frozen(self):
        for value in (
            "--target-prompt-tokens 65536",
            "--pressure-prompt-tokens 135040 --pressure-count 9",
            "--max-tokens 8 --timeout-s 900",
            "--min-candidate-cached 65504",
            "m158-block-major-fixed-20260726",
        ):
            self.assertIn(value, self.source)
        self.assertIn("hybrid_kv_startup_gate.py", self.source)
        self.assertIn("compare_m1_58_block_major_ab.py", self.source)
        self.assertIn("verify_bare_host_runtime_identity.py", self.source)
        self.assertIn("compare_bi100_preflights.py", self.source)

    def test_identity_and_private_output_are_required(self):
        self.assertIn('if [[ $# -ne 2 ]]', self.source)
        self.assertIn("usage: $0 INSTANCE RUN_ROOT", self.source)
        self.assertIn("INSTANCE=$1", self.source)
        self.assertIn('if [[ "$RUN_ROOT" != /tmp/* ]]', self.source)
        self.assertIn(
            "M1-58 output must stay outside the source repository",
            self.source,
        )
        self.assertIn("M1-58 refuses a dirty source tree", self.source)
        self.assertIn("source_revision.txt", self.source)
        self.assertIn("source_branch.txt", self.source)
        self.assertIn("instance.txt", self.source)
        self.assertNotIn("M1_58_RUN_ROOT", self.source)
        self.assertNotIn("bench_runs/m1_58", self.source)

    def test_service_lifetimes_and_failure_scan_are_fail_closed(self):
        self.assertIn('setsid "$ROOT/launch_service"', self.source)
        self.assertIn(
            'bi100_stop_process_group \\\n'
            '            "$ACTIVE_PGID" "$ACTIVE_PID" 60 20',
            self.source,
        )
        self.assertNotIn('kill -TERM "$ACTIVE_PID"', self.source)
        self.assertNotIn("pkill -9", self.source)
        self.assertIn("refusing to overwrite", self.source)
        self.assertIn("--gpus 0,1,2,3", self.source)
        self.assertIn("Traceback", self.source)
        self.assertIn("Connection reset by peer", self.source)
        self.assertIn("Gloo.*(failed|reset|error)", self.source)
        self.assertIn("NCCL.*(failed|abort|error)", self.source)
        self.assertIn("worker.*(died|lost|exited unexpectedly)", self.source)
        self.assertIn("engine iteration timed out", self.source)
        self.assertLess(
            self.source.index("run_preflight before_control"),
            self.source.index("run_arm control 0"),
        )
        self.assertLess(
            self.source.index("run_preflight after_control"),
            self.source.index("run_arm candidate 1"),
        )
        self.assertLess(
            self.source.index("run_preflight after_candidate"),
            self.source.index("run_offline_gate preflight_comparison"),
        )

    def test_each_arm_and_finalizer_require_postflight(self):
        self.assertIn("service_postflight_gate.py", self.source)
        self.assertIn(
            'run_service_postflight "$output/service_postflight"',
            self.source,
        )
        self.assertIn(
            'run_service_postflight "$RUN_ROOT/final_postflight"',
            self.source,
        )
        self.assertIn("service_postflight.rc", self.source)
        self.assertIn("final_postflight.rc", self.source)
        self.assertIn("preflight_final.json", self.source)
        self.assertIn("final_preflight_comparison.rc", self.source)
        self.assertLess(
            self.source.index('stop_service\n    cleanup_rc=$?'),
            self.source.index(
                'run_service_postflight "$output/service_postflight"'),
        )

    def test_timeout_and_runner_status_are_fail_closed(self):
        self.assertIn("scan_timeout_rcs", self.source)
        self.assertIn("124|137", self.source)
        self.assertIn("timeout_scan.rc", self.source)
        self.assertIn("runner_status.json", self.source)
        self.assertIn(
            '"schema": "bi100-m1-58-block-major-ab-runner-v2"',
            self.source,
        )
        self.assertIn('"production_promotion_authorized": False', self.source)
        self.assertIn("CURRENT_STAGE=complete", self.source)

    def test_fixed_ab_contract_remains_one_variable(self):
        self.assertIn(
            'BI100_RUNTIME_WORKDIR="$output/runtime-workdir"',
            self.source,
        )
        self.assertEqual(self.source.count("--mode candidate"), 1)
        self.assertNotIn("--mode control", self.source)
        self.assertNotIn("BI100_BLOCK_MAJOR_CPU_KV=1", self.source)
        self.assertNotIn("BI100_BLOCK_MAJOR_CPU_KV=0", self.source)


if __name__ == "__main__":
    unittest.main()
