from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_m1_136_fused_prefill_shadow.sh"


class M1136FusedPrefillShadowRunnerTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SCRIPT.read_text(encoding="utf-8")

    def test_fixed_tp4_and_context_contract(self) -> None:
        for fragment in (
            "--gpus 0,1,2,3",
            "--targets 65536,131072",
            "BI100_ATTN_COREX_FUSED_PREFILL=1",
            "BI100_ATTN_COREX_FUSED_PREFILL_SHADOW=1",
            "BI100_ATTN_COREX_FUSED_PREFILL_SHADOW_CONTEXTS=49152,114688",
            "BI100_ATTN_COREX_FUSED_PREFILL_SHADOW_MAX_CALLS_PER_CONTEXT=2",
            "max_model_len\": 262144",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.source)

    def test_shadow_is_qualified_before_scoped_cleanup(self) -> None:
        self.assertLess(
            self.source.index("trap finish EXIT"),
            self.source.index("CURRENT_STAGE=shadow_qualification"),
        )
        self.assertLess(
            self.source.index("CURRENT_STAGE=shadow_qualification"),
            self.source.index("CURRENT_STAGE=complete"),
        )
        self.assertIn("qualify_fused_prefill_shadow.py", self.source)

    def test_cleanup_uses_term_grace_and_recorded_identity(self) -> None:
        for fragment in (
            '"$ACTIVE_PGID" "$ACTIVE_PID" 60 20',
            "cleanup_recorded_bi100_sessions.py",
            "qualify_recorded_session_cleanup.py",
            "service_postflight_gate.py",
            "compare_bi100_preflights.py",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.source)

    def test_runner_cannot_authorize_main_or_yaml(self) -> None:
        self.assertIn('"production_promotion_authorized": False', self.source)
        self.assertIn('"yaml_change_authorized": False', self.source)
        self.assertIn('"main_merge_authorized": False', self.source)


if __name__ == "__main__":
    unittest.main()
