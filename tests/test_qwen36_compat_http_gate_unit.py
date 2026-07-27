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

GATE_SPEC = importlib.util.spec_from_file_location(
    "qwen36_compat_http_gate",
    TESTS / "qwen36_compat_http_gate.py",
)
GATE = importlib.util.module_from_spec(GATE_SPEC)
assert GATE_SPEC.loader is not None
GATE_SPEC.loader.exec_module(GATE)

COMPARE_SPEC = importlib.util.spec_from_file_location(
    "compare_qwen36_compat_http_ab",
    TESTS / "compare_qwen36_compat_http_ab.py",
)
COMPARE = importlib.util.module_from_spec(COMPARE_SPEC)
assert COMPARE_SPEC.loader is not None
COMPARE_SPEC.loader.exec_module(COMPARE)


def fake_request(*, system_parts_supported: bool, image_limit: int):
    def request(method, url, payload=None, *, timeout_s):
        del timeout_s
        if method == "GET" and url.endswith("/v1/models"):
            return 200, {
                "data": [{
                    "id": "llm",
                    "max_model_len": 262144,
                }],
            }
        if method == "GET" and url.endswith("/health"):
            return 200, {}
        if method != "POST" or payload is None:
            raise AssertionError("unexpected request")

        messages = payload["messages"]
        system_parts = any(
            message.get("role") == "system"
            and isinstance(message.get("content"), list)
            for message in messages
        )
        system_count = sum(
            message.get("role") == "system" for message in messages)
        if (system_parts and system_count > 1
                and not system_parts_supported):
            return 400, {"error": {"type": "TemplateError"}}

        image_count = sum(
            1
            for message in messages
            for part in (
                message.get("content")
                if isinstance(message.get("content"), list) else ()
            )
            if part.get("type") == "image_url"
        )
        if image_count > image_limit:
            return 400, {"error": {"type": "image_count_limit"}}

        content = (
            f"synthetic-image-output-{image_count}"
            if image_count else "synthetic-system-output"
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
                "prompt_tokens": 40 + image_count,
                "completion_tokens": 8,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }

    return request


def run_report(system_parts_supported: bool, image_limit: int) -> dict:
    expected = 200 if system_parts_supported else 400
    with mock.patch.object(
            GATE,
            "_request_json",
            side_effect=fake_request(
                system_parts_supported=system_parts_supported,
                image_limit=image_limit)):
        return GATE.run_gate(
            "http://127.0.0.1:8000",
            Path("/diagnostic-model"),
            30.0,
            expected,
            image_limit,
        )


def attribution(image_count: int) -> dict:
    return {
        "schema": COMPARE.ATTRIBUTION_SCHEMA,
        "qualified": True,
        "chat_4xx_access_count": 1,
        "by_reason": {"image_count_limit": 1},
        "request_shapes": [{
            "images": image_count,
            "image_data": image_count,
            "image_remote": 0,
            "image_other": 0,
            "system_part_msgs": 0,
            "system_text_parts": 0,
            "system_other_parts": 0,
        }],
    }


class Qwen36CompatHttpGateTest(unittest.TestCase):

    def test_baseline_accepts_single_parts_but_reproduces_multi_system_400(self):
        report = run_report(False, 1)
        self.assertTrue(report["qualified"], report)
        self.assertEqual(report["case_count"], 8)
        cases = {row["name"]: row for row in report["cases"]}
        self.assertEqual(
            cases["single_system_text_parts"]["evidence"]["http_status"],
            200,
        )
        self.assertTrue(
            cases["single_system_text_parts"]["evidence"][
                "canonical_generation_exact"])
        self.assertEqual(
            cases["multiple_system_text_parts"]["evidence"]["http_status"],
            400,
        )
        self.assertEqual(
            cases["post_4xx_health"]["evidence"]["http_status"], 200)
        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn("data:image", serialized)
        self.assertNotIn(GATE.SYSTEM_TEXT, serialized)

    def test_candidate_system_and_two_image_replays_are_exact(self):
        report = run_report(True, 2)
        self.assertTrue(report["qualified"], report)
        cases = {row["name"]: row for row in report["cases"]}
        for name in (
                "single_system_text_parts",
                "multiple_system_text_parts"):
            evidence = cases[name]["evidence"]
            self.assertEqual(evidence["http_status"], 200)
            self.assertTrue(evidence["canonical_generation_exact"])
        replay = cases["image_at_limit_replay"]["evidence"]
        self.assertEqual(replay["image_count"], 2)
        self.assertTrue(replay["exact_generation_match"])
        self.assertEqual(
            cases["over_limit_image_400"]["evidence"]["image_count"], 3)

    def test_three_arm_comparison_qualifies_without_promotion(self):
        baseline = run_report(False, 1)
        candidate_default = run_report(True, 1)
        candidate_image2 = run_report(True, 2)
        result = COMPARE.compare(
            baseline,
            candidate_default,
            candidate_image2,
            attribution(2),
            attribution(3),
        )
        self.assertTrue(result["qualified"], result)
        self.assertTrue(
            result["checks"]["system_text_parts_http_fix_qualified"])
        self.assertTrue(
            result["checks"]["explicit_image_two_structural_qualified"])
        self.assertFalse(result["default_image_limit_change_authorized"])
        self.assertFalse(result["production_promotion_authorized"])

    def test_output_drift_rejects_comparison(self):
        baseline = run_report(False, 1)
        candidate_default = run_report(True, 1)
        candidate_image2 = run_report(True, 2)
        changed = copy.deepcopy(candidate_default)
        cases = {row["name"]: row for row in changed["cases"]}
        cases["one_image"]["evidence"]["message_sha256"] = "0" * 64
        result = COMPARE.compare(
            baseline,
            changed,
            candidate_image2,
            attribution(2),
            attribution(3),
        )
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "one-image output differs" in reason
            for reason in result["reasons"]))

    def test_unclassified_or_wrong_image_4xx_rejects_comparison(self):
        baseline = run_report(False, 1)
        candidate_default = run_report(True, 1)
        candidate_image2 = run_report(True, 2)
        bad = attribution(2)
        bad["by_reason"] = {"unclassified_chat_error": 1}
        result = COMPARE.compare(
            baseline,
            candidate_default,
            candidate_image2,
            bad,
            attribution(3),
        )
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "image_count_limit" in reason for reason in result["reasons"]))


if __name__ == "__main__":
    unittest.main()
