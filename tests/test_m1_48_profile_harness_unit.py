from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "scripts" / "run_m1_48_prefill_profile.sh"
INSTALLER = ROOT / "scripts" / "install_bi100_bare_host_runtime.sh"
PATCH_OPS = ROOT / "qwen3_6_scripts" / "patch_ops.sh"


class M148ProfileHarnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = HARNESS.read_text(encoding="utf-8")

    def test_m1_49_qualification_is_a_hard_precondition(self):
        prerequisite = self.source.index(
            'require_zero_rc "$M1_49_LONG_DIR/qualification.rc"')
        first_preflight = self.source.index("run_preflight before_control")
        self.assertLess(prerequisite, first_preflight)
        self.assertIn(
            "bi100-m1-49-long-context-qualification-v1", self.source)

    def test_control_and_profile_share_the_fixed_runtime_contract(self):
        for line in (
            "BI100_HYBRID_KV_ACCOUNTING=full_attention",
            "BI100_GDN_CACHE_POLICY=admission64",
            "BI100_GDN_RESTORE_MODE=direct",
            "BI100_CPU_KV_OFFLOAD=0",
            "BI100_ATTN_COREX_FUSED_PREFILL=0",
            "BI100_PROFILE_MODE=event",
            "BI100_PROFILE_INCLUDE_STARTUP=0",
        ):
            self.assertIn(line, self.source)
        self.assertIn("run_arm control 0", self.source)
        self.assertIn("run_arm profile 1", self.source)
        self.assertIn('--run-id "$RUN_ID" --mode "$arm"', self.source)

    def test_process_cleanup_and_gpu_leak_gates_are_mandatory(self):
        self.assertIn('setsid "$ROOT/launch_service"', self.source)
        self.assertIn('kill -TERM -- "-$ACTIVE_PGID"', self.source)
        self.assertIn('kill -KILL -- "-$ACTIVE_PGID"', self.source)
        for stage in ("before_control", "after_control", "after_profile"):
            self.assertIn(f"run_preflight {stage}", self.source)
        self.assertIn("--max-free-memory-drop-bytes 1073741824", self.source)
        self.assertIn("fatal_scan \"$arm\"", self.source)
        cleanup = self.source.index("verify_prequalification_cleanup\n")
        qualification = self.source.index("run_gate qualification 300")
        self.assertLess(cleanup, qualification)
        self.assertIn("--prequalification-cleanup", self.source)
        self.assertIn('rm -f "$RUN_ROOT/qualification.json"', self.source)

    def test_source_cleanliness_excludes_only_generated_evidence(self):
        self.assertIn("':(exclude)bench_runs/**'", self.source)
        self.assertIn("--untracked-files=all", self.source)

    def test_profile_is_descriptive_and_uses_no_old_amdahl_constant(self):
        self.assertIn("summarize_prefill_path_profile.py", self.source)
        self.assertIn("qualify_m1_48_prefill_profile.py", self.source)
        self.assertNotIn("2.5770191951430728", self.source)
        self.assertNotIn("observed-service-improvement", self.source)

    def test_runtime_overlay_covers_every_profiled_file(self):
        installer = INSTALLER.read_text(encoding="utf-8")
        patch_ops = PATCH_OPS.read_text(encoding="utf-8")
        for name in (
            '"bi100_profile"',
            '"paged_attention"',
            '"xformers_backend"',
        ):
            self.assertIn(name, installer)
        self.assertIn("python3 ./patch_xformers_profile.py", patch_ops)
        self.assertIn(
            "python3 ./patch_worker_startup_profile_guard.py", patch_ops)
        self.assertIn("startup_profile_guard_patch", installer)
        self.assertIn("BI100_RUNTIME_INSTALL_REPORT", self.source)

    def test_harness_never_changes_submission_or_visibility(self):
        self.assertNotIn("computility-run.yaml", self.source)
        self.assertNotIn("git push", self.source)
        self.assertNotIn("visibility", self.source.lower())


if __name__ == "__main__":
    unittest.main()
