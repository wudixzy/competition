from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "scripts/run_quality_service_gate.sh"


class QualityServiceGateHarnessTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.source = HARNESS.read_text(encoding="utf-8")

    def test_every_run_uses_fresh_service_and_four_gpu_preflights(self):
        self.assertIn('setsid "$ROOT/launch_service"', self.source)
        self.assertIn('run_preflight before', self.source)
        self.assertIn('run_preflight after', self.source)
        self.assertIn('--gpus 0,1,2,3', self.source)
        self.assertIn('bi100_stop_process_group', self.source)

    def test_quality_contract_is_attested_before_start(self):
        identity = self.source.index("verify_bare_host_runtime_identity.py")
        contract = self.source.index("build_quality_runtime_contract.py")
        allocator = self.source.index("prefix_namespace_fork_gate.py")
        broadcast = self.source.index("gdn_action_broadcast_gate.py")
        service = self.source.index('setsid "$ROOT/launch_service"')
        self.assertLess(identity, contract)
        self.assertLess(contract, allocator)
        self.assertLess(allocator, broadcast)
        self.assertLess(broadcast, service)
        self.assertIn('--expected-cache-trace 1', self.source)

    def test_harness_preserves_model_capability_contract(self):
        self.assertIn('export BI100_HYBRID_KV_ACCOUNTING=full_attention',
                      self.source)
        self.assertIn('export BI100_GDN_ALLOW_NAN_ZERO=0', self.source)
        self.assertIn('export BI100_GDN_FINITE_CHECK=0', self.source)
        self.assertNotIn("--quantization", self.source)
        self.assertNotIn("--speculative-model", self.source)
        self.assertNotIn("computility-run.yaml", self.source)

        functional = (ROOT / "scripts/run_quality_functional_gate.sh").read_text(
            encoding="utf-8")
        self.assertIn("--allow-bare-engine-n2-skip", functional)

    def test_raw_run_artifacts_cannot_enter_repository(self):
        self.assertIn(
            "quality run output must stay outside the source repository",
            self.source,
        )
        self.assertIn("quality run output must use a private /tmp path",
                      self.source)

    def test_fatal_scan_covers_known_worker_failures(self):
        for marker in (
                "CUDA error", "SIGSEGV", "out of memory",
                "worker process.*died", "Gloo.*failed"):
            self.assertIn(marker, self.source)


if __name__ == "__main__":
    unittest.main()
