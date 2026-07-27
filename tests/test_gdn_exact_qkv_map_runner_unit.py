from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_gdn_exact_qkv_map_gate.sh"
EXTENSION = ROOT / "tests" / "corex_gdn_qkv_map_ext.cu"


class ExactQkvMapRunnerTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.runner = RUNNER.read_text(encoding="utf-8")
        cls.extension = EXTENSION.read_text(encoding="utf-8")

    def test_probe_is_fixed_to_one_exact_larger_boundary(self):
        self.assertIn("bench_gdn_exact_qkv_map.py", self.runner)
        self.assertIn("qualify_gdn_exact_qkv_map.py", self.runner)
        self.assertIn("build_corex_gdn_qkv_map.sh", self.runner)
        self.assertIn("qkv_map_kernel", self.extension)
        self.assertIn("__half2float(value[index])", self.extension)
        self.assertNotIn("computility-run.yaml", self.runner)

    def test_every_tracked_process_uses_current_process_group_cleanup(self):
        self.assertIn('source "$ROOT/scripts/lib/process_group.sh"',
                      self.runner)
        self.assertIn("setsid timeout --signal=TERM --kill-after=60s",
                      self.runner)
        self.assertIn(
            '"$ACTIVE_PGID" "$ACTIVE_PID" 60 20', self.runner)
        self.assertNotIn("pkill", self.runner)
        self.assertIn("trap 'exit 143' TERM", self.runner)
        self.assertIn("trap 'exit 130' INT", self.runner)
        self.assertIn("trap finish EXIT", self.runner)

    def test_postflight_is_complete_and_fail_closed(self):
        for gate in (
                "cleanup", "service_postflight", "fatal_scan",
                "timeout_scan", "preflight_after",
                "preflight_comparison"):
            self.assertIn(f'"{gate}": read_rc(', self.runner)
        self.assertIn("tests/service_postflight_gate.py", self.runner)
        self.assertIn("run_preflight after", self.runner)
        self.assertIn("Gloo.*(failed|reset|error)", self.runner)
        self.assertIn(
            "worker.*(died|lost|exited unexpectedly)", self.runner)
        self.assertIn("124|137", self.runner)

    def test_raw_results_are_private_and_source_must_be_clean(self):
        self.assertIn("RUN_ROOT must use a private /tmp path", self.runner)
        self.assertIn("exact q/k/v gate refuses a dirty source tree",
                      self.runner)
        self.assertNotIn("git push", self.runner)


if __name__ == "__main__":
    unittest.main()
