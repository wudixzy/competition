from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SERVICE = (
    ROOT / "scripts/run_quality_service_gate.sh"
).read_text(encoding="utf-8")
OUTER = (
    ROOT / "scripts/run_m1_85_admission64_quality_ab.sh"
).read_text(encoding="utf-8")
INSTALLER = (
    ROOT / "scripts/install_bi100_bare_host_runtime.sh"
).read_text(encoding="utf-8")
PATCH_OPS = (
    ROOT / "qwen3_6_scripts/patch_ops.sh"
).read_text(encoding="utf-8")
PROTOCOL = (
    ROOT / "qwen3_6_scripts/protocol.py"
).read_text(encoding="utf-8")
TOKENIZATION = (
    ROOT / "qwen3_6_scripts/serving_tokenization.py"
).read_text(encoding="utf-8")
SERVING_CHAT = (
    ROOT / "qwen3_6_scripts/serving_chat.py"
).read_text(encoding="utf-8")
COLLECTOR = (
    ROOT / "tests/teacher_forced_topk_api.py"
).read_text(encoding="utf-8")
WRAPPER = (
    ROOT / "scripts/run_m1_132_teacher_forced_ab.sh"
).read_text(encoding="utf-8")
REPEAT_WRAPPER = (
    ROOT / "scripts/run_m1_134_teacher_forced_control_repeat.sh"
).read_text(encoding="utf-8")
PRIVATE_ARTIFACTS = (
    ROOT / "scripts/lib/private_artifacts.sh"
).read_text(encoding="utf-8")


class M1132TeacherForcedRunnerTests(unittest.TestCase):

    def test_wrapper_selects_private_variant(self) -> None:
        self.assertIn(
            "BI100_QUALITY_AB_VARIANT="
            "m1-132-fused-prefill-teacher-forced",
            WRAPPER,
        )
        self.assertIn(
            'exec "$ROOT/scripts/run_m1_85_admission64_quality_ab.sh" "$@"',
            WRAPPER,
        )

    def test_control_repeat_wrapper_keeps_both_arms_fused_off(self) -> None:
        self.assertIn(
            "BI100_QUALITY_AB_VARIANT="
            "m1-134-teacher-forced-control-repeat",
            REPEAT_WRAPPER,
        )
        self.assertIn("m1-134-control-a-fused-off", OUTER)
        self.assertIn("m1-134-control-b-fused-off", OUTER)
        self.assertIn(
            "teacher_forced_comparison_mode=control-repeat",
            OUTER,
        )

    def test_service_runs_fixed_collector(self) -> None:
        self.assertIn(
            "functional|long-context|decode|contract-smoke|ifeval|"
            "teacher-forced",
            SERVICE,
        )
        self.assertIn("tests/teacher_forced_topk_api.py", SERVICE)
        self.assertIn("--timeout-s 3600", SERVICE)
        self.assertIn(
            '--out "$RUN_ROOT/teacher_forced_observation.json"',
            SERVICE,
        )

    def test_hmac_key_is_removed_before_service_start(self) -> None:
        unset_index = SERVICE.index("unset BI100_TEACHER_FORCED_HMAC_KEY")
        launch_index = SERVICE.index('"$ROOT/launch_service"')
        self.assertLess(unset_index, launch_index)
        self.assertNotIn(
            "env BI100_TEACHER_FORCED_HMAC_KEY=",
            OUTER,
        )
        self.assertNotIn(
            "runner_env+=(BI100_TEACHER_FORCED_HMAC_KEY=",
            OUTER,
        )

    def test_outer_uses_one_key_and_compares_both_arms(self) -> None:
        self.assertIn("secrets.token_hex(32)", OUTER)
        self.assertIn(
            "tests/compare_teacher_forced_logprobs.py",
            OUTER,
        )
        self.assertIn(
            'quality/layered_quality_gate.v1.json',
            OUTER,
        )
        self.assertIn(
            '[[ $teacher_forced_comparison_rc -eq 0 ]]',
            OUTER,
        )

    def test_outer_trap_deletes_private_observations(self) -> None:
        finish_index = OUTER.index("finish() {")
        cleanup_index = OUTER.index(
            'remove_teacher_forced_observations "$RUN_ROOT"', finish_index)
        status_index = OUTER.index('write_status "$rc"', finish_index)
        self.assertLess(finish_index, cleanup_index)
        self.assertLess(cleanup_index, status_index)
        self.assertIn(
            '"$run_root/control"/'
            '.teacher_forced_observation.json.*.tmp',
            PRIVATE_ARTIFACTS,
        )
        self.assertIn("private_observation_cleanup.rc", OUTER)
        self.assertIn(
            'source "$ROOT/scripts/lib/private_artifacts.sh"', OUTER)

    def test_server_token_identity_path_is_installed_and_used(self) -> None:
        self.assertIn(
            "chat_template_kwargs: Optional[Dict[str, Any]]",
            PROTOCOL,
        )
        self.assertIn("request.chat_template_kwargs or {}", TOKENIZATION)
        self.assertIn("cp ./serving_tokenization.py", PATCH_OPS)
        self.assertIn(
            'root / "qwen3_6_scripts/serving_tokenization.py"',
            INSTALLER,
        )
        self.assertIn('"/tokenize"', COLLECTOR)
        self.assertIn(
            "server_token_ids != local_token_ids",
            COLLECTOR,
        )
        self.assertIn(
            '"bi100_prompt_logprobs_sample_positions": sampled',
            COLLECTOR,
        )
        self.assertIn(
            "request.bi100_prompt_logprobs_sample_positions",
            SERVING_CHAT,
        )
        self.assertIn(
            "row if position in selected else None",
            SERVING_CHAT,
        )
        self.assertIn(
            "num_cached_tokens not in (None, 0)",
            SERVING_CHAT,
        )
        self.assertIn(
            'details.get("cached_tokens", 0) != 0',
            COLLECTOR,
        )

    def test_no_private_observation_digest_is_persisted(self) -> None:
        self.assertNotIn("teacher_forced_observation_sha256", OUTER)
        self.assertIn(
            "teacher_forced_comparison_sha256",
            OUTER,
        )


if __name__ == "__main__":
    unittest.main()
