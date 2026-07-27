from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_moe_pairwise_w13_gate.sh"


class MoePairwiseW13RunnerUnitTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_runner_is_bound_to_one_physical_gpu_and_clean_source(self):
        self.assertIn('export CUDA_VISIBLE_DEVICES="$GPU_INDEX"', self.source)
        self.assertIn("--gpus \"$GPU_INDEX\"", self.source)
        self.assertIn("pairwise W13 gate refuses a dirty source tree",
                      self.source)
        self.assertIn("verify_bare_host_runtime_identity.py", self.source)

    def test_runner_uses_fixed_numerical_and_performance_contract(self):
        self.assertIn("bench_moe_pairwise_w13.py", self.source)
        self.assertIn("--sequence-steps 500", self.source)
        self.assertIn("qualify_moe_pairwise_w13.py", self.source)
        self.assertIn('"relative_l2": 1.0e-5', self.source)
        self.assertIn('"fixed_speedup": 1.5', self.source)
        self.assertIn('"routed_speedup": 1.25', self.source)

    def test_runner_always_runs_postflight_and_writes_status(self):
        self.assertIn("trap finish EXIT", self.source)
        self.assertIn("run_preflight after", self.source)
        self.assertIn("compare_bi100_preflights.py", self.source)
        self.assertIn('--expected-gpus "$GPU_INDEX"', self.source)
        self.assertIn("runner_status.json", self.source)
        self.assertIn('"production_promotion_authorized": False', self.source)

    def test_raw_outputs_stay_outside_repository(self):
        self.assertIn("RUN_ROOT must stay outside the source repository",
                      self.source)
        self.assertIn("RUN_ROOT must use a private /tmp path", self.source)


if __name__ == "__main__":
    unittest.main()
