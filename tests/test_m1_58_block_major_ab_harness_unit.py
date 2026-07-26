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

    def test_service_lifetimes_and_failure_scan_are_fail_closed(self):
        self.assertIn('setsid "$ROOT/launch_service"', self.source)
        self.assertIn("bi100_stop_process_group", self.source)
        self.assertIn("refusing to overwrite", self.source)
        self.assertIn("--gpus 0,1,2,3", self.source)
        self.assertIn("Traceback", self.source)
        self.assertIn("Connection reset by peer", self.source)
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
            self.source.index("comparison.json"),
        )


if __name__ == "__main__":
    unittest.main()
