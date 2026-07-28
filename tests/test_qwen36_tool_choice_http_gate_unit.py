from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

SPEC = importlib.util.spec_from_file_location(
    "qwen36_tool_choice_http_gate",
    TESTS / "qwen36_tool_choice_http_gate.py",
)
GATE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GATE)


def tool_response(*, city: str = "Beijing") -> dict:
    return {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "reasoning_content": "",
                "tool_calls": [{
                    "id": "call_synthetic",
                    "type": "function",
                    "function": {
                        "name": GATE.TOOL_NAME,
                        "arguments": json.dumps(
                            {"city": city}, separators=(",", ":")),
                    },
                }],
            },
        }],
        "usage": {
            "prompt_tokens": 40,
            "completion_tokens": 8,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
    }


def tool_stream(*, city: str = "Beijing") -> dict:
    return {
        "chunks": 3,
        "done": 1,
        "usage_blocks": 1,
        "usage": {
            "prompt_tokens": 40,
            "completion_tokens": 8,
            "total_tokens": 48,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
        "content": "",
        "reasoning_content": "",
        "finish_reasons": ["tool_calls"],
        "tool_calls": [{
            "name": GATE.TOOL_NAME,
            "arguments": {"city": city},
        }],
    }


def run_report(
    *,
    failed_mode: str | None = None,
    drift_mode: str | None = None,
) -> dict:
    def request(method, url, payload=None, *, timeout_s):
        del timeout_s
        if method == "GET" and url.endswith("/health"):
            return 200, {}
        if method != "POST" or payload is None:
            raise AssertionError("unexpected request")
        mode = ("omitted" if "tool_choice" not in payload
                else "auto" if payload["tool_choice"] == "auto"
                else "named")
        if failed_mode == f"{mode}_nonstream":
            return 400, {"error": {"message": "synthetic failure"}}
        city = "北京" if drift_mode == f"{mode}_nonstream" else "Beijing"
        return 200, tool_response(city=city)

    def stream(base, payload, timeout_s):
        del base, timeout_s
        mode = ("omitted" if "tool_choice" not in payload
                else "auto" if payload["tool_choice"] == "auto"
                else "named")
        if failed_mode == f"{mode}_stream":
            return 400, {}
        city = "北京" if drift_mode == f"{mode}_stream" else "Beijing"
        return 200, tool_stream(city=city)

    with mock.patch.object(GATE, "_request_json", side_effect=request):
        report = GATE.run_gate(
            "http://127.0.0.1:8000",
            Path("/diagnostic-model"),
            30.0,
            request_stream=stream,
        )
    return report


class Qwen36ToolChoiceHttpGateTest(unittest.TestCase):

    def test_all_valid_modes_and_transports_qualify(self):
        report = run_report()
        self.assertTrue(report["qualified"], report)
        self.assertEqual(report["case_count"], 7)
        self.assertTrue(all(report["checks"].values()))
        self.assertFalse(report["strict_true_evaluated"])
        self.assertFalse(report["required_tool_choice_evaluated"])
        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn(GATE.USER_TEXT, serialized)
        self.assertNotIn(GATE.TOOL_NAME, serialized)
        self.assertNotIn("Beijing", serialized)

    def test_each_valid_mode_must_return_http_200(self):
        for mode in (
            "omitted_nonstream",
            "auto_nonstream",
            "named_nonstream",
            "omitted_stream",
            "auto_stream",
            "named_stream",
        ):
            with self.subTest(mode=mode):
                report = run_report(failed_mode=mode)
                self.assertFalse(report["qualified"])
                failed = [
                    case["name"] for case in report["cases"]
                    if not case["ok"]
                ]
                self.assertEqual(failed, [f"tool_choice_{mode}"])

    def test_semantic_drift_fails_closed(self):
        report = run_report(drift_mode="named_stream")
        self.assertFalse(report["qualified"])
        failed = [
            case["name"] for case in report["cases"] if not case["ok"]
        ]
        self.assertEqual(failed, [])
        self.assertFalse(
            report["checks"]["nonstream_stream_semantics_exact"])
        self.assertTrue(report["checks"]["omitted_auto_semantics_exact"])

    def test_invalid_nonstream_tool_shape_fails(self):
        response = tool_response()
        response["choices"][0]["finish_reason"] = "stop"
        with self.assertRaisesRegex(
                AssertionError, "did not finish as tool_calls"):
            GATE._tool_semantics_from_response(response)

    def test_invalid_stream_usage_fails(self):
        stream = tool_stream()
        stream["usage"]["total_tokens"] = 49
        with self.assertRaisesRegex(
                AssertionError, "total_tokens is inconsistent"):
            GATE._tool_semantics_from_stream(stream)

    def test_report_does_not_depend_on_call_ids(self):
        left = tool_response()
        right = copy.deepcopy(left)
        right["choices"][0]["message"]["tool_calls"][0]["id"] = "other"
        self.assertEqual(
            GATE._tool_semantics_from_response(left)["semantic_sha256"],
            GATE._tool_semantics_from_response(right)["semantic_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
