from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_ifeval_service_gate.sh"


class IFEvalServiceGateTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_runtime_and_evaluator_source_identities_are_separate(self):
        self.assertIn("EXPECTED_RUNTIME_REVISION", self.text)
        self.assertIn("runtime_source_revision.txt", self.text)
        self.assertIn("evaluator_source_revision.txt", self.text)
        self.assertIn("runtime source revision differs", self.text)

    def test_lifecycle_keeps_all_safety_gates(self):
        for marker in (
                "verify_bare_host_runtime_identity.py",
                "build_quality_runtime_contract.py",
                "prefix_namespace_fork_gate.py",
                "gdn_action_broadcast_gate.py",
                "bi100_preflight.py",
                "hybrid_kv_startup_gate.py",
                "compare_bi100_preflights.py",
                "scan_fatal_log",
                'unlink "$RUN_ROOT/ifeval.checkpoint.json"',
                "bi100_stop_process_group"):
            self.assertIn(marker, self.text)

    def test_ifeval_runner_uses_fixed_runtime_contract(self):
        self.assertIn("tests/ifeval_quality_api.py", self.text)
        self.assertIn("--runtime-contract", self.text)
        self.assertIn('--progress "$RUN_ROOT/ifeval_progress.json"', self.text)
        self.assertIn("--gdn-cache-policy", self.text)
        self.assertIn("--kv-eviction-policy", self.text)
        self.assertIn("43200s", self.text)

    def test_evaluator_dependencies_do_not_enter_service_pythonpath(self):
        service_line = next(
            line for line in self.text.splitlines()
            if line.startswith("export PYTHONPATH="))
        self.assertNotIn("IFEVAL_ENV", service_line)
        self.assertNotIn("google_ifeval", service_line)
        self.assertIn("IFEVAL_PYTHONPATH=", self.text)
        self.assertIn('NLTK_DATA="$IFEVAL_ENV/nltk_data"', self.text)


if __name__ == "__main__":
    unittest.main()
