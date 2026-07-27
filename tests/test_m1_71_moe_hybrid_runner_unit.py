from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_m1_71_moe_hybrid_exact_tail_gate.sh"


class M171MoeHybridRunnerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_fixed_component_contract_has_no_parameter_scan(self):
        self.assertIn("bench_moe_direct_routed.py", self.source)
        self.assertIn("qualify_moe_hybrid_exact_tail.py", self.source)
        self.assertIn("--warmup 30 --iterations 300 --repeats 9", self.source)
        self.assertIn("--sequence-steps 500 --seed 20260716", self.source)
        self.assertIn('"relative_l2": 1.0e-5', self.source)
        self.assertIn('"speedup": 1.25', self.source)
        self.assertIn('"saving_ms": 0.02', self.source)
        self.assertNotIn("for tile", self.source)
        self.assertNotIn("for threshold", self.source)

    def test_runtime_identity_and_gpu_gates_are_mandatory(self):
        self.assertIn("verify_m1_70_runtime_pair.py", self.source)
        self.assertIn("run_preflight before", self.source)
        self.assertIn("run_preflight after", self.source)
        self.assertIn("compare_bi100_preflights.py", self.source)
        self.assertIn("service_postflight_gate.py", self.source)
        self.assertIn("--settle-timeout-s 90 --clean-samples 3", self.source)
        for gate in (
                "runtime_pair",
                "benchmark",
                "qualification",
                "cleanup",
                "service_postflight",
                "fatal_scan",
                "timeout_scan",
                "preflight_after",
                "preflight_comparison"):
            self.assertIn(f'"{gate}"', self.source)

    def test_cleanup_is_scoped_and_graceful(self):
        self.assertIn(
            'bi100_stop_process_group "$ACTIVE_PGID" "$ACTIVE_PID" 60 20',
            self.source,
        )
        self.assertIn("setsid timeout", self.source)
        self.assertIn("trap 'exit 143' TERM", self.source)
        self.assertIn("trap 'exit 130' INT", self.source)
        self.assertIn("trap finish EXIT", self.source)
        self.assertNotIn("pkill", self.source)
        self.assertIn("Gloo.*(failed|reset|error)", self.source)
        self.assertIn("NCCL.*(failed|abort|error)", self.source)

    def test_private_component_gate_does_not_touch_submission(self):
        self.assertIn("RUN_ROOT must use a private /tmp path", self.source)
        self.assertIn("M1-71 gate refuses a dirty source tree", self.source)
        self.assertNotIn("computility-run.yaml", self.source)
        self.assertNotIn("git push", self.source)
        self.assertNotIn("MODEL_PATH", self.source)


if __name__ == "__main__":
    unittest.main()
