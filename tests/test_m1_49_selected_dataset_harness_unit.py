from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_m1_49_selected_dataset.sh"


class M149SelectedDatasetHarnessTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SCRIPT.read_text(encoding="utf-8")

    def test_qualified_long_context_is_a_hard_precondition(self):
        self.assertIn(
            'require_zero_rc "$M1_49_LONG_DIR/overall.rc"', self.source)
        self.assertIn(
            'require_zero_rc "$M1_49_LONG_DIR/cleanup.rc"', self.source)
        self.assertIn(
            'require_zero_rc "$M1_49_LONG_DIR/qualification.rc"',
            self.source,
        )
        self.assertIn(
            "hybrid-kv-capacity-correctness-not-prefill-speed",
            self.source,
        )

    def test_candidate_runtime_and_dataset_are_fixed(self):
        for value in (
            "BI100_HYBRID_KV_ACCOUNTING=full_attention",
            "BI100_GDN_CACHE_POLICY=admission64",
            "BI100_GDN_RESTORE_MODE=direct",
            "BI100_ATTN_COREX_FUSED_PREFILL=0",
        ):
            self.assertIn(value, self.source)
        self.assertIn(
            "TRACE_MODE=${M1_49_SELECTED_TRACE_MODE:-0}", self.source)
        self.assertIn('export BI100_CACHE_TRACE="$TRACE_MODE"', self.source)
        self.assertIn(
            "M1_49_SELECTED_TRACE_MODE must be 0 or 1", self.source)
        self.assertIn(
            "dac6afc77621b51dbc09cfa046c008a1e51a779bb771edcb27cb6a686f8884c8",
            self.source,
        )
        self.assertIn("[4, 4, 3, 2]", self.source)
        self.assertIn("--max-tokens 256", self.source)
        self.assertIn("--seed 20260713", self.source)

    def test_replay_is_first_api_workload_and_is_safely_qualified(self):
        self.assertNotIn("smoke_api.py", self.source)
        replay = self.source.index('run_gate replay 12600')
        qualification = self.source.index(
            'run_offline_gate qualification 60')
        self.assertLess(replay, qualification)
        self.assertIn("replay_selected_dataset.py", self.source)
        self.assertIn("qualify_selected_dataset_replay.py", self.source)
        self.assertIn('rm -f "$RUN_ROOT/qualification.json"', self.source)

    def test_service_has_preflights_and_process_group_cleanup(self):
        self.assertIn("run_preflight before_selected", self.source)
        self.assertIn("run_preflight after_selected", self.source)
        self.assertIn("compare_bi100_preflights.py", self.source)
        self.assertIn("run_offline_gate preflight_comparison", self.source)
        self.assertIn('setsid "$ROOT/launch_service"', self.source)
        self.assertIn(
            'bi100_stop_process_group "$ACTIVE_PGID" "$ACTIVE_PID"',
            self.source,
        )
        self.assertIn("fatal_scan", self.source)

    def test_trace_smoke_is_explicit_and_cannot_qualify_as_881(self):
        self.assertIn('if [[ "$TRACE_MODE" == 1 ]]', self.source)
        self.assertIn("--expected-requests 13", self.source)
        self.assertNotIn("--qualification-trace", self.source)
        self.assertIn("qualify_selected_dataset_trace_smoke.py", self.source)
        self.assertIn('--expected-cache-trace "$TRACE_MODE"', self.source)
        self.assertIn(
            'prior_service["cache_trace"] = "<diagnostic>"', self.source)
        self.assertIn("verify_bare_host_runtime_identity.py", self.source)
        self.assertIn(
            'prior_service["runtime_site_packages"] = "<attested-overlay>"',
            self.source,
        )


if __name__ == "__main__":
    unittest.main()
