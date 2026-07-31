from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_m1_169_tail64_nofinal_tp1_ab.sh"
BENCH = ROOT / "tests" / "bench_m1_104_admission64_policy_matrix.py"


class M1169Tail64NoFinalRunnerUnitTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNNER.read_text(encoding="utf-8")
        cls.bench = BENCH.read_text(encoding="utf-8")

    def test_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(RUNNER)], check=True)

    def test_candidate_is_accepted_by_fixed_matrix(self) -> None:
        self.assertIn(
            'choices=("fine32", "admission64", "tail64_nofinal", "off")',
            self.bench,
        )

    def test_tp1_screen_preserves_production_shape_contract(self) -> None:
        for fragment in (
            "--max-model-len 262144",
            "--tensor-parallel-size 1",
            "--max-num-batched-tokens 8192",
            "BI100_GDN_RESTORE_MODE=hybrid64",
            "BI100_ATTN_COREX_FUSED_PREFILL=0",
            "CANDIDATE_POLICY=${CANDIDATE_POLICY:-tail64_nofinal}",
            "ARM_ORDER=${ARM_ORDER:-admission64,$CANDIDATE_POLICY}",
        ):
            self.assertIn(fragment, self.source)

    def test_cleanup_is_scoped_and_promotion_stays_closed(self) -> None:
        self.assertIn("bi100_stop_process_group", self.source)
        self.assertIn(" 60 20 \\", self.source)
        self.assertNotIn("pkill", self.source)
        self.assertIn('"tp4_evaluated": False', self.source)
        self.assertIn('"production_promotion_authorized": False', self.source)

    def test_startup_poll_is_quiet_and_stage_is_recorded(self) -> None:
        self.assertIn("health >/dev/null 2>&1 && return 0", self.source)
        self.assertIn('printf \'%s\\n\' "$CURRENT_STAGE" > "$RUN_ROOT/stage.txt"',
                      self.source)
        self.assertIn("CURRENT_STAGE=complete\nwrite_stage", self.source)

    def test_service_cannot_import_repository_shadow_package(self) -> None:
        start = self.source.index("start_service()")
        stop = self.source.index("scan_fatal()")
        service = self.source[start:stop]
        self.assertLess(service.index("cd /tmp"), service.index("exec env"))
        self.assertIn("verify_service_import_contract()", self.source)
        self.assertIn('api_path.is_relative_to(site)', self.source)
        self.assertIn('cli_path.is_relative_to(site)', self.source)
        self.assertIn('"--reasoning-parser" in options', self.source)
        self.assertIn('"qwen3_coder" in ToolParserManager.tool_parsers',
                      self.source)
        self.assertIn('"service_import_contract": rc(', self.source)

    def test_policy_contract_cannot_import_repository_shadow_package(self) -> None:
        start = self.source.index("verify_policy_contract()")
        stop = self.source.index("run_arm()")
        policy = self.source[start:stop]
        self.assertLess(policy.index("cd /tmp"), policy.index("python3 -"))
        self.assertIn('> "$arm/policy_contract.stdout"', self.source)
        self.assertIn('2> "$arm/policy_contract.stderr"', self.source)

    def test_policy_contract_uses_runtime_overlay_introspection(self) -> None:
        self.assertIn("verify_policy_contract()", self.source)
        self.assertIn("from vllm.gdn_prefix import (", self.source)
        self.assertIn("gdn_cache_policy_from_env", self.source)
        self.assertIn("gdn_restore_mode_from_env", self.source)
        self.assertIn('observed == expected and restore == "hybrid64"',
                      self.source)
        self.assertNotIn('[BI100] GDN cache; policy=$policy', self.source)

    def test_both_complete_arms_reach_privacy_safe_comparison(self) -> None:
        self.assertIn("compare_m1_169_tail64_nofinal_tp1.py", self.source)
        self.assertIn(
            '--control "$RUN_ROOT/admission64/measurement.json"',
            self.source,
        )
        self.assertIn(
            '--candidate "$RUN_ROOT/tail64_nofinal/measurement.json"',
            self.source,
        )
        self.assertIn('"comparison": rc("comparison.rc")', self.source)

    def test_arm_records_each_gate_and_waits_for_port_release(self) -> None:
        self.assertIn("wait_port_free()", self.source)
        self.assertIn("for _ in $(seq 1 120)", self.source)
        self.assertIn("socket.SO_REUSEADDR", self.source)
        self.assertIn("except OSError:", self.source)
        for name in (
            "startup.rc",
            "policy_contract.rc",
            "measurement.rc",
            "health_after.rc",
            "cleanup.rc",
            "port_free.rc",
        ):
            self.assertIn(name, self.source)

    def test_optional_profile_override_is_diagnostic_and_recorded(self) -> None:
        self.assertIn("NUM_GPU_BLOCKS_OVERRIDE=${NUM_GPU_BLOCKS_OVERRIDE:-}",
                      self.source)
        self.assertIn('--num-gpu-blocks-override "$NUM_GPU_BLOCKS_OVERRIDE"',
                      self.source)
        self.assertIn('"num_gpu_blocks_override": (', self.source)
        self.assertIn('"production_promotion_authorized": False', self.source)


if __name__ == "__main__":
    unittest.main()
