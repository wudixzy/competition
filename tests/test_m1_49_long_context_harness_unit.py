from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_m1_49_long_context_gates.sh"
CLEANUP = ROOT / "scripts" / "lib" / "process_group.sh"


class M149LongContextHarnessTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SCRIPT.read_text(encoding="utf-8")
        cls.cleanup = CLEANUP.read_text(encoding="utf-8")

    def test_requires_qualified_fixed_ab_before_service_start(self):
        self.assertIn('require_zero_rc "$AB_DIR/overall.rc"', self.source)
        self.assertIn('require_zero_rc "$AB_DIR/comparison.rc"', self.source)
        self.assertIn(
            'require_zero_rc "$AB_DIR/preflight_comparison_final.rc"',
            self.source,
        )
        self.assertIn(
            'report.get("qualified") is not True', self.source)
        self.assertLess(
            self.source.index('require_zero_rc "$AB_DIR/overall.rc"'),
            self.source.index('setsid "$ROOT/launch_service"'),
        )

    def test_candidate_runtime_contract_is_frozen(self):
        fixed_exports = (
            "BI100_HYBRID_KV_ACCOUNTING=full_attention",
            "BI100_GDN_CACHE_POLICY=admission64",
            "BI100_GDN_RESTORE_MODE=direct",
            "BI100_CPU_KV_OFFLOAD=0",
            "BI100_ATTN_COREX_FUSED_PREFILL=0",
            "BI100_CACHE_TRACE=0",
        )
        for value in fixed_exports:
            self.assertIn(value, self.source)
        self.assertNotIn("BI100_HYBRID_KV_ACCOUNTING=legacy40", self.source)
        self.assertIn("--mode full_attention", self.source)
        self.assertIn("--max-model-len 262144", self.source)

    def test_long_context_and_multimodal_gates_are_fixed(self):
        self.assertIn("--target-prompt-tokens 131000 --max-tokens 256", self.source)
        self.assertIn("--equivalence-mode exact", self.source)
        self.assertIn("--target-prompt-tokens 235000 --max-tokens 1000", self.source)
        self.assertIn("--equivalence-mode warm-repeat", self.source)
        self.assertIn("--target-prompt-tokens 262000 --max-tokens 16", self.source)
        self.assertIn("multimodal_prefix_isolation_api.py", self.source)
        self.assertIn("qualify_multimodal_prefix_isolation.py", self.source)
        self.assertIn("qualify_m1_49_long_context.py", self.source)

    def test_service_lifetime_is_guarded_by_gpu_preflights_and_group_cleanup(self):
        self.assertIn("run_preflight before_long", self.source)
        self.assertIn("run_preflight after_long", self.source)
        self.assertIn("compare_bi100_preflights.py", self.source)
        self.assertIn(
            "--max-free-memory-drop-bytes 1073741824", self.source)
        self.assertIn('setsid "$ROOT/launch_service"', self.source)
        self.assertIn(
            'source "$ROOT/scripts/lib/process_group.sh"', self.source)
        self.assertIn(
            'bi100_stop_process_group "$ACTIVE_PGID" "$ACTIVE_PID"',
            self.source,
        )
        self.assertIn('kill -TERM -- "-$pgid"', self.cleanup)
        self.assertIn('kill -KILL -- "-$pgid"', self.cleanup)
        self.assertIn('substr($3, 1, 1) == "Z"', self.cleanup)
        self.assertIn('printf \'%s\\n\' "$cleanup_rc"', self.source)


if __name__ == "__main__":
    unittest.main()
