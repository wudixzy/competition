from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_gdn_combined_qk_norm_gate.sh"


class CombinedQkNormRunnerTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_runner_is_bound_to_exact_overlay_and_one_gpu(self):
        self.assertIn("verify_bare_host_runtime_identity.py", self.source)
        self.assertIn('export CUDA_VISIBLE_DEVICES="$GPU_INDEX"', self.source)
        self.assertIn('--gpus "$GPU_INDEX"', self.source)
        self.assertIn('--expected-gpus "$GPU_INDEX"', self.source)

    def test_runner_uses_fixed_numerical_and_speed_contract(self):
        self.assertIn("bench_gdn_combined_qk_norm.py", self.source)
        self.assertIn("--sequence-steps 500", self.source)
        self.assertIn("qualify_gdn_combined_qk_norm.py", self.source)
        self.assertIn('"relative_l2": 1.0e-5', self.source)
        self.assertIn('"speedup": 1.25', self.source)
        self.assertIn('"saving_ms": 0.02', self.source)
        self.assertIn('"production_promotion_authorized": False', self.source)

    def test_cleanup_is_scoped_and_postflight_is_mandatory(self):
        self.assertIn("setsid timeout", self.source)
        self.assertIn(
            'bi100_stop_process_group "$ACTIVE_PGID" "$ACTIVE_PID" 60 20',
            self.source,
        )
        self.assertIn("service_postflight_gate.py", self.source)
        self.assertIn("run_preflight after", self.source)
        self.assertIn("preflight_comparison.rc", self.source)
        self.assertIn("timeout_scan.rc", self.source)
        self.assertIn("trap finish EXIT", self.source)

    def test_raw_artifacts_stay_outside_repository(self):
        self.assertIn(
            "RUN_ROOT must stay outside the source repository", self.source)
        self.assertIn("RUN_ROOT must use a private /tmp path", self.source)


if __name__ == "__main__":
    unittest.main()
