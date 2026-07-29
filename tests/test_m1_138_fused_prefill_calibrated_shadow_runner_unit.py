from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "run_m1_138_fused_prefill_calibrated_shadow.sh"
RUNNER = ROOT / "scripts" / "run_m1_136_fused_prefill_shadow.sh"


class M1138FusedPrefillCalibratedShadowRunnerTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.wrapper = WRAPPER.read_text(encoding="utf-8")
        cls.runner = RUNNER.read_text(encoding="utf-8")

    def test_wrapper_selects_only_calibrated_variant(self) -> None:
        self.assertIn(
            "export BI100_FUSED_PREFILL_SHADOW_VARIANT=calibrated",
            self.wrapper,
        )
        self.assertIn(
            'exec "$ROOT/scripts/run_m1_136_fused_prefill_shadow.sh"',
            self.wrapper,
        )

    def test_calibrated_contract_is_used_by_qualification(self) -> None:
        for fragment in (
            'QUALIFIER="$ROOT/tests/'
            'qualify_fused_prefill_calibrated_shadow.py"',
            'CONTRACT="$ROOT/quality/'
            'fused_prefill_numeric_adjudication.v1.json"',
            'python3 "$QUALIFIER"',
            '--contract "$CONTRACT"',
            "bi100-m1-138-fused-prefill-calibrated-shadow-runner-v1",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.runner)

    def test_finite_failure_records_but_nonfinite_still_aborts(self) -> None:
        self.assertIn("FAILURE_ACTION=record", self.runner)
        self.assertIn("NUMERIC_MODE=calibrated", self.runner)
        self.assertIn(
            "BI100_ATTN_COREX_FUSED_PREFILL_SHADOW_FAILURE_ACTION="
            '"$FAILURE_ACTION"',
            self.runner,
        )


if __name__ == "__main__":
    unittest.main()
