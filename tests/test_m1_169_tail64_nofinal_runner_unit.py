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
            'choices=("fine32", "admission64", "tail64_nofinal")',
            self.bench,
        )

    def test_tp1_screen_preserves_production_shape_contract(self) -> None:
        for fragment in (
            "--max-model-len 262144",
            "--tensor-parallel-size 1",
            "--max-num-batched-tokens 8192",
            "BI100_GDN_RESTORE_MODE=hybrid64",
            "BI100_ATTN_COREX_FUSED_PREFILL=0",
            "admission64,tail64_nofinal",
            "tail64_nofinal,admission64",
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


if __name__ == "__main__":
    unittest.main()
