from __future__ import annotations

import importlib.util
import json
import copy
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

SPEC = importlib.util.spec_from_file_location(
    "qwen36_tool_http_gate",
    TESTS / "qwen36_tool_http_gate.py",
)
GATE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GATE)

COMPARE_SPEC = importlib.util.spec_from_file_location(
    "compare_qwen36_tool_http_ab",
    TESTS / "compare_qwen36_tool_http_ab.py",
)
COMPARE = importlib.util.module_from_spec(COMPARE_SPEC)
assert COMPARE_SPEC.loader is not None
COMPARE_SPEC.loader.exec_module(COMPARE)


def fake_request(
    *,
    strict_false_supported: bool,
    object_history_supported: bool,
    object_output_drift: bool = False,
):
    def request(method, url, payload=None, *, timeout_s):
        del timeout_s
        if method == "GET" and url.endswith("/v1/models"):
            return 200, {
                "data": [{"id": "llm", "max_model_len": 262144}],
            }
        if method == "GET" and url.endswith("/health"):
            return 200, {}
        if method != "POST" or payload is None:
            raise AssertionError("unexpected request")

        tool = payload["tools"][0]["function"]
        strict = tool.get("strict")
        if strict is True:
            return 400, {"error": {"type": "strict_true"}}
        if strict is False and not strict_false_supported:
            return 400, {"error": {"type": "strict_false"}}
        if payload.get("tool_choice") == "required":
            return 400, {"error": {"type": "required"}}

        arguments = None
        for message in payload["messages"]:
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                arguments = tool_calls[0]["function"]["arguments"]
                break
        if arguments == "{invalid":
            return 400, {"error": {"type": "invalid_json"}}
        if isinstance(arguments, dict) and not object_history_supported:
            return 400, {"error": {"type": "object_arguments"}}

        content = (
            "changed-object-output"
            if object_output_drift and isinstance(arguments, dict)
            else "synthetic-tool-output"
        )
        return 200, {
            "choices": [{
                "finish_reason": "length",
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": None,
                    "tool_calls": [],
                },
            }],
            "usage": {
                "prompt_tokens": 48,
                "completion_tokens": 8,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }

    return request


def run_report(
    strict_false_supported: bool,
    object_history_supported: bool,
    *,
    object_output_drift: bool = False,
) -> dict:
    with mock.patch.object(
            GATE,
            "_request_json",
            side_effect=fake_request(
                strict_false_supported=strict_false_supported,
                object_history_supported=object_history_supported,
                object_output_drift=object_output_drift)):
        return GATE.run_gate(
            "http://127.0.0.1:8000",
            Path("/diagnostic-model"),
            30.0,
            200 if strict_false_supported else 400,
            200 if object_history_supported else 400,
        )


def attribution() -> dict:
    return {
        "schema": COMPARE.ATTRIBUTION_SCHEMA,
        "qualified": True,
        "complete": True,
        "chat_4xx_access_count": 3,
        "attributed_count": 3,
        "by_reason": {
            "invalid_tool_arguments_json": 1,
            "request_validation_tool_strict": 1,
            "unsupported_tool_choice_required": 1,
        },
        "privacy": {
            "contains_multimodal_url_or_bytes": False,
            "contains_raw_log_lines": False,
            "contains_request_content": False,
            "contains_response_content": False,
            "contains_tool_schema": False,
        },
    }


class Qwen36ToolHttpGateTest(unittest.TestCase):

    def test_baseline_reproduces_two_compatibility_400s(self):
        report = run_report(False, False)
        self.assertTrue(report["qualified"], report)
        self.assertEqual(report["case_count"], 9)
        cases = {row["name"]: row for row in report["cases"]}
        self.assertEqual(
            cases["function_tool_strict_false"]["evidence"][
                "http_status"],
            400,
        )
        self.assertEqual(
            cases["tool_arguments_json_object"]["evidence"][
                "http_status"],
            400,
        )
        self.assertEqual(
            cases["post_4xx_health"]["evidence"]["http_status"], 200)

    def test_candidate_accepts_equivalent_forms_exactly(self):
        report = run_report(True, True)
        self.assertTrue(report["qualified"], report)
        cases = {row["name"]: row for row in report["cases"]}
        strict = cases["function_tool_strict_false"]["evidence"]
        history = cases["tool_arguments_json_object"]["evidence"]
        self.assertEqual(strict["http_status"], 200)
        self.assertTrue(strict["default_generation_exact"])
        self.assertEqual(history["http_status"], 200)
        self.assertTrue(history["string_generation_exact"])
        for name in (
                "tool_arguments_invalid_json_400",
                "function_tool_strict_true_400",
                "tool_choice_required_400"):
            self.assertEqual(
                cases[name]["evidence"]["http_status"], 400)

        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn(GATE.USER_TEXT, serialized)
        self.assertNotIn("synthetic_lookup", serialized)
        self.assertNotIn('{"key":"synthetic"}', serialized)

    def test_equivalent_object_output_drift_fails_closed(self):
        report = run_report(
            True,
            True,
            object_output_drift=True,
        )
        self.assertFalse(report["qualified"])
        failed = [case for case in report["cases"] if not case["ok"]]
        self.assertEqual(
            [case["name"] for case in failed],
            ["tool_arguments_json_object"],
        )

    def test_ab_comparison_qualifies_without_promotion(self):
        result = COMPARE.compare(
            run_report(False, False),
            run_report(True, True),
            attribution(),
        )
        self.assertTrue(result["qualified"], result)
        self.assertTrue(
            result["checks"]["strict_false_http_fix_qualified"])
        self.assertTrue(
            result["checks"]["object_history_http_fix_qualified"])
        self.assertFalse(result["production_promotion_authorized"])

    def test_unclassified_4xx_or_output_drift_rejects_comparison(self):
        baseline = run_report(False, False)
        candidate = run_report(True, True)
        bad_attribution = attribution()
        bad_attribution["by_reason"]["unclassified_chat_error"] = 1
        result = COMPARE.compare(
            baseline,
            candidate,
            bad_attribution,
        )
        self.assertFalse(result["qualified"])

        changed = copy.deepcopy(candidate)
        cases = {case["name"]: case for case in changed["cases"]}
        cases["function_tool_default"]["evidence"][
            "message_sha256"] = "0" * 64
        result = COMPARE.compare(baseline, changed, attribution())
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "function_tool_default changed" in reason
            for reason in result["reasons"]))

        wrong_reason = attribution()
        wrong_reason["by_reason"] = {
            "request_validation_tool_parameters": 1,
            "request_validation_tool_strict": 1,
            "unsupported_tool_choice_required": 1,
        }
        result = COMPARE.compare(baseline, candidate, wrong_reason)
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "reason set differs" in reason
            for reason in result["reasons"]))


if __name__ == "__main__":
    unittest.main()
