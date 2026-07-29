from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "scripts/run_quality_service_gate.sh"


class QualityServiceGateHarnessTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.source = HARNESS.read_text(encoding="utf-8")

    def test_every_run_uses_fresh_service_and_four_gpu_preflights(self):
        self.assertIn("exec_bi100_session.py", self.source)
        self.assertIn(
            '"$RUN_ROOT/process_group_identity.json"', self.source)
        self.assertIn('"$ROOT/launch_service"', self.source)
        self.assertIn('run_preflight before', self.source)
        self.assertIn('run_preflight after', self.source)
        self.assertIn('--gpus 0,1,2,3', self.source)
        self.assertIn('bi100_stop_process_group', self.source)

    def test_quality_contract_is_attested_before_start(self):
        identity = self.source.index("verify_bare_host_runtime_identity.py")
        contract = self.source.index("build_quality_runtime_contract.py")
        allocator = self.source.index("prefix_namespace_fork_gate.py")
        broadcast = self.source.index("gdn_action_broadcast_gate.py")
        service = self.source.index('"$ROOT/launch_service"')
        self.assertLess(identity, contract)
        self.assertLess(contract, allocator)
        self.assertLess(allocator, broadcast)
        self.assertLess(broadcast, service)
        self.assertIn('--expected-cache-trace 1', self.source)
        self.assertIn('tests/agent_workload_matrix.py', self.source)
        self.assertIn('agent_workload.rc', self.source)

    def test_contract_smoke_runs_only_the_two_regression_cases(self):
        self.assertIn(
            "functional|long-context|decode|contract-smoke|ifeval",
            self.source,
        )
        self.assertIn("--case max_tokens_1", self.source)
        self.assertIn("--case stream_forced_terminal", self.source)
        self.assertIn(
            "SUITE must be functional, long-context, decode, "
            "contract-smoke, ifeval, or teacher-forced",
            self.source,
        )

    def test_current_hybrid64_candidate_is_an_explicit_quality_mode(self):
        self.assertIn("direct|hybrid64|aligned)", self.source)
        self.assertIn(
            '[[ "$RESTORE_MODE" == hybrid64 '
            '&& "$POLICY" != admission64 ]]',
            self.source,
        )
        builder = (
            ROOT / "tests/build_quality_runtime_contract.py"
        ).read_text(encoding="utf-8")
        startup = (
            ROOT / "tests/hybrid_kv_startup_gate.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'choices=("direct", "hybrid64", "aligned")',
            builder,
        )
        self.assertIn(
            'choices=("direct", "hybrid64", "aligned")',
            startup,
        )

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
        self.assertNotIn("--allow-bare-engine-n2-skip", functional)

    def test_strict_reference_profile_disables_rejected_pair_only(self):
        self.assertIn(
            "BI100_QUALITY_KERNEL_PROFILE:-submission", self.source)
        self.assertIn("strict-reference)", self.source)
        self.assertIn("strict-reference-combined-qk)", self.source)
        self.assertIn("MOE_DIRECT=0", self.source)
        self.assertIn("GDN_PACKED=0", self.source)
        self.assertIn("GDN_COMBINED_QK=1", self.source)
        self.assertIn(
            'export BI100_MOE_COREX_DIRECT_ROUTED="$MOE_DIRECT"',
            self.source,
        )
        self.assertIn(
            'export BI100_GDN_COREX_PACKED_DECODE="$GDN_PACKED"',
            self.source,
        )
        self.assertIn(
            'export BI100_GDN_COMBINED_QK_NORM="$GDN_COMBINED_QK"',
            self.source,
        )
        self.assertIn('--kernel-profile "$KERNEL_PROFILE"', self.source)
        self.assertIn(
            '--expected-kernel-profile "$KERNEL_PROFILE"', self.source)

    def test_raw_run_artifacts_cannot_enter_repository(self):
        self.assertIn(
            "quality run output must stay outside the source repository",
            self.source,
        )
        self.assertIn("quality run output must use a private /tmp path",
                      self.source)

    def test_ifeval_environment_is_bound_to_an_approved_manifest(self):
        self.assertIn(
            "BI100_IFEVAL_MANIFEST:-$ROOT/quality/external/"
            "google_ifeval/manifest.v1.json",
            self.source,
        )
        self.assertIn(
            "01c7e9dd4aafc11b5e2505fec2c3c71c"
            "53d8d27992ab40445638e97404440107",
            self.source,
        )
        self.assertIn(
            'value.get("manifest_sha256") != sys.argv[2]',
            self.source,
        )
        self.assertIn('--manifest "$IFEVAL_MANIFEST"', self.source)

    def test_fatal_scan_covers_known_worker_failures(self):
        for marker in (
                "CUDA error", "SIGSEGV", "out of memory",
                "worker.*(died|lost|exited unexpectedly)",
                "Gloo.*(failed|reset|error)",
                "NCCL.*(failed|abort|error)",
                "Connection reset by peer", "Timeout(Error|Expired)"):
            self.assertIn(marker, self.source)

    def test_cleanup_has_grace_period_and_fail_closed_postflight(self):
        self.assertIn(
            '"$ACTIVE_PGID" "$ACTIVE_PID" 60 20 \\',
            self.source,
        )
        self.assertIn(
            '"$ACTIVE_STARTTIME" "$ACTIVE_SESSION_TOKEN" || rc=$?',
            self.source,
        )
        self.assertIn('return "$rc"', self.source)
        self.assertIn("trap - EXIT", self.source)
        self.assertIn("trap '' TERM INT", self.source)
        self.assertIn(
            "cleanup_recorded_bi100_sessions.py", self.source)
        self.assertIn(
            "qualify_recorded_session_cleanup.py", self.source)
        self.assertIn("tests/service_postflight_gate.py", self.source)
        self.assertIn("--settle-timeout-s 30 --clean-samples 3",
                      self.source)
        self.assertIn('"service_postflight": read_rc(', self.source)
        self.assertIn('"api_4xx_attribution": read_rc(', self.source)
        self.assertIn("tests/summarize_api_4xx_log.py", self.source)
        self.assertIn('"timeout_scan": read_rc(', self.source)
        cleanup = self.source.index("stop_service\n")
        recovery = self.source.index("recover_service_session\n")
        process_scan = self.source.index("run_service_postflight\n")
        gpu_preflight = self.source.index("run_preflight after")
        self.assertLess(cleanup, recovery)
        self.assertLess(recovery, process_scan)
        self.assertLess(cleanup, process_scan)
        self.assertLess(process_scan, gpu_preflight)
        self.assertNotIn("pkill", self.source)
        for gate in (
                "process_group", "cleanup", "service_recovery",
                "service_recovery_clean", "service_postflight", "fatal_scan",
                "api_4xx_attribution",
                "timeout_scan", "preflight_after",
                "preflight_comparison"):
            self.assertIn(f'"{gate}": read_rc(', self.source)


if __name__ == "__main__":
    unittest.main()
