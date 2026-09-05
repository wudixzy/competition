from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import m1_180_capability_distribution_api as workload
import run_m1_180_three_arm_adjudication as orchestrator


class M1180HarnessTests(unittest.TestCase):

    @staticmethod
    def _reasoning_response(reasoning: str, content: str) -> dict:
        return {
            "choices": [{"message": {"content": content,
                                       "reasoning_content": reasoning},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 4,
                      "total_tokens": 24,
                      "prompt_tokens_details": {"cached_tokens": 0}},
        }

    def test_frozen_population_has_ten_cases_per_stratum(self) -> None:
        self.assertEqual(len(workload.CODE_CASES), 10)
        self.assertEqual(len(workload.REASONING_CASES), 10)
        self.assertEqual(len(workload.TOOL_VALUES), 10)
        self.assertEqual(len(workload.STRUCTURED_VALUES), 10)
        self.assertEqual(len(workload.COLORS), 10)
        self.assertEqual(len(workload.LONG_TARGETS), 10)
        self.assertEqual(workload.SMOKE_PER_STRATUM, 4)

    def test_smoke_baseline_only_is_detected_without_raw_content(self) -> None:
        case = {"case_id": "code_00", "stage": "smoke", "pass": True}
        reference = {"capability": {"cases": [case]}}
        candidate = [{**case, "pass": False}]
        self.assertEqual(
            workload.smoke_regressions([reference], candidate), ["code_00"])

    def test_three_arm_commands_bind_selector_and_variant(self) -> None:
        args = SimpleNamespace(
            instance="i", run_root=Path("/tmp/run"), pair_id="p",
            session_preflight=Path("/tmp/preflight.json"),
            m1_109_extension=Path("/tmp/m109.so"),
            m1_109_sha256="a" * 64,
            m1_162_extension=Path("/tmp/m162.so"),
            m1_162_sha256="b" * 64,
        )
        fused = orchestrator.arm_command(args, "fused_off")
        m109 = orchestrator.arm_command(args, "m1_109")
        m162 = orchestrator.arm_command(args, "m1_162")
        self.assertIn("fused_off", fused)
        self.assertNotIn("--fused-variant", fused)
        self.assertIn("m1_109_fp32_qk", m109)
        self.assertIn("m1_162_fp16_qk", m162)
        self.assertIn("--reference-fused-off", m162)
        self.assertIn("--reference-m1-109", m162)

    def test_candidate_service_adapts_in_one_workload_process(self) -> None:
        source = Path(workload.__file__).read_text(encoding="utf-8")
        self.assertIn("cases = run_cases(client, tokenizer, 0, SMOKE_PER_STRATUM)",
                      source)
        self.assertIn("if extended:", source)
        self.assertIn("SMOKE_PER_STRATUM, FULL_PER_STRATUM", source)
        self.assertNotIn("retain-raw-responses", source)

    def test_reasoning_prompt_does_not_contain_expected_answer(self) -> None:
        captured = {}

        def response(_client, payload, timeout=900):
            captured["prompt"] = payload["messages"][0]["content"]
            return self._reasoning_response("worked independently", "FINAL=91"), 1.0

        with mock.patch.object(workload, "_response", side_effect=response):
            result = workload._run_reasoning(object(), 0)
        self.assertTrue(result["pass"])
        self.assertTrue(result["reasoning_protocol_valid"])
        self.assertNotIn("91", captured["prompt"])
        self.assertNotIn("FINAL=91", captured["prompt"])

    def test_reasoning_protocol_requires_reasoning_and_content(self) -> None:
        with mock.patch.object(
                workload, "_response",
                return_value=(self._reasoning_response("", "FINAL=91"), 1.0)):
            result = workload._run_reasoning(object(), 0)
        self.assertFalse(result["reasoning_protocol_valid"])
        self.assertFalse(result["pass"])

    def test_multimodal_color_uses_whole_normalized_terms(self) -> None:
        self.assertTrue(workload._matches_color_answer(
            "The dominant color is red.", ("red", "红")))
        self.assertTrue(workload._matches_color_answer("红色", ("red", "红")))
        self.assertFalse(workload._matches_color_answer(
            "The region is predominant.", ("red", "红")))


if __name__ == "__main__":
    unittest.main()
