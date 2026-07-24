from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "requalify_m1_48_prefill_profile.sh"


class M148ProfileRequalificationHarnessTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SCRIPT.read_text(encoding="utf-8")

    def test_accepts_only_the_exact_query_head_contract_failure(self):
        self.assertIn('require_nonzero_rc "$SOURCE_RUN/profile_summary.rc"',
                      self.source)
        self.assertIn('require_nonzero_rc "$SOURCE_RUN/overall.rc"',
                      self.source)
        self.assertIn("len(reasons) != 116", self.source)
        self.assertIn(
            "source failure is not the exact TP4 query-head contract bug",
            self.source,
        )
        self.assertIn("for rank in range(4)", self.source)
        self.assertIn("for offset in range(29)", self.source)

    def test_original_evidence_is_read_only_and_cleanup_is_rechecked(self):
        self.assertNotIn('> "$SOURCE_RUN/', self.source)
        self.assertIn(
            'bi100_process_group_count "$pgid" live', self.source)
        self.assertIn('sock.bind(("127.0.0.1", 8000))', self.source)
        self.assertIn("run_gate post_source_preflight 150", self.source)

    def test_recomputes_with_the_fixed_model_geometry(self):
        self.assertIn("summarize_prefill_path_profile.py", self.source)
        self.assertIn("--expected-processes 4", self.source)
        self.assertIn("--num-attention-heads 16", self.source)
        self.assertIn("qualify_m1_48_prefill_profile.py", self.source)
        self.assertIn('printf \'%s\\n\' 0 > "$RUN_ROOT/', self.source)

    def test_requires_all_successful_runtime_gates(self):
        for gate in (
            "cleanup",
            "runtime_identity",
            "preflight_before_control",
            "control_startup_gate",
            "control_service",
            "control/fatal_scan",
            "preflight_after_control",
            "profile_startup_gate",
            "profile_service",
            "profile/fatal_scan",
            "preflight_after_profile",
            "preflight_comparison",
        ):
            self.assertIn(gate, self.source)


if __name__ == "__main__":
    unittest.main()
