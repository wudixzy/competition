#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import qwen36_diagnostic_api as api  # noqa: E402


class Qwen36DiagnosticApiUnitTest(unittest.TestCase):
    def test_response_summary_is_privacy_safe(self) -> None:
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "private generated text",
                    "reasoning_content": "private reasoning",
                    "tool_calls": [],
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 8},
            },
        }
        summary = api._response_summary(response)
        encoded = json.dumps(summary)
        self.assertNotIn("private generated text", encoded)
        self.assertNotIn("private reasoning", encoded)
        self.assertEqual(summary["cached_tokens"], 8)
        self.assertEqual(summary["completion_tokens"], 3)
        self.assertEqual(len(summary["message_sha256"]), 64)

    def test_png_fixture_is_embedded_and_valid(self) -> None:
        value = api._solid_png_data_url((255, 0, 0))
        prefix = "data:image/png;base64,"
        self.assertTrue(value.startswith(prefix))
        decoded = base64.b64decode(value[len(prefix):])
        self.assertTrue(decoded.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(decoded.endswith(b"IEND\xaeB`\x82"))

    def test_non_200_response_uses_digest_in_error(self) -> None:
        with mock.patch.object(
                api, "_request_json", return_value=(500, {"secret": "body"})):
            with self.assertRaisesRegex(
                    AssertionError, r"response_sha256=[0-9a-f]{64}"):
                api._post_chat(
                    "http://127.0.0.1:8000",
                    {"model": "llm"},
                    timeout_s=1,
                )

    def test_response_requires_generated_tokens(self) -> None:
        response = {
            "choices": [{
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "length",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 0},
        }
        with self.assertRaisesRegex(
                AssertionError, "generated no completion tokens"):
            api._response_summary(response)


class Qwen36DiagnosticHarnessStaticTest(unittest.TestCase):
    def test_harness_is_diagnostic_only_and_keeps_capacity(self) -> None:
        harness = (
            ROOT / "scripts" / "run_qwen36_diagnostic_gate.sh"
        ).read_text(encoding="utf-8")
        for marker in (
            "--max-model-len 262144",
            "verify_qwen36_diagnostic_checkpoint.py",
            "BI100_DIAGNOSTIC_LAYER_TRACE=1",
            "BI100_HYBRID_KV_ACCOUNTING=full_attention",
            "prefix_boundary_api.py",
            "qwen36_diagnostic_api.py",
            "qwen36_quality_contract_diagnostic.py",
            "quality_contract_gate.json",
            '"quality_contract": read_rc("quality_contract_gate.rc")',
            "qwen36_compat_http_gate.py",
            "--multiple-system-parts-expected-status 200",
            "--image-limit 1",
            "compat_http_gate.json",
            '"compat_http": read_rc("compat_http_gate.rc")',
            "qwen36_tool_http_gate.py",
            "--strict-false-expected-status 200",
            "--object-history-expected-status 200",
            "tool_http_gate.json",
            '"tool_http": read_rc("tool_http_gate.rc")',
            "bi100_stop_process_group",
            '"$ACTIVE_PGID" "$ACTIVE_PID" 60 20',
            "service_postflight_gate.py",
            "scan_timeout_rcs",
            "perform_postflight",
            "qwen36-diagnostic-cleanup-v1",
            "cleanup_status.json",
            "trap 'exit 143' TERM",
            "production_promotion_authorized",
            "unset CUDA_VISIBLE_DEVICES",
        ):
            self.assertIn(marker, harness)
        self.assertNotIn("computility-run.yaml", harness)
        self.assertNotIn("git push", harness)

    def test_layer_trace_is_opt_in_and_post_moe(self) -> None:
        model = (
            ROOT / "qwen3_6_scripts" / "qwen3_5.py"
        ).read_text(encoding="utf-8")
        env_index = model.index('os.getenv("BI100_DIAGNOSTIC_LAYER_TRACE")')
        moe_index = model.index('with bi100_timer("layer.moe")', env_index)
        completed_index = model.index("stage=completed", moe_index)
        self.assertLess(moe_index, completed_index)
        yaml = (ROOT / "computility-run.yaml").read_text(encoding="utf-8")
        self.assertNotIn("BI100_DIAGNOSTIC_LAYER_TRACE", yaml)

    def test_submission_preflight_forbids_trace_leak(self) -> None:
        preflight = (
            ROOT / "tests" / "submission_preflight.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"BI100_DIAGNOSTIC_LAYER_TRACE"', preflight)


if __name__ == "__main__":
    unittest.main()
