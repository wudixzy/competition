from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))
SCRIPT = ROOT / "tests/quality_gate_api.py"
SPEC = importlib.util.spec_from_file_location("quality_gate_api", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class StreamingClient:
    def stream(self, payload):
        return 200, {
            "chunks": 7,
            "done": 1,
            "usage_blocks": 1,
            "event_span_s": 0.5,
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 4,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
            "content": (
                "红色\n橙色\n黄色\n绿色\n蓝色\n"
                "紫色\n黑色\n白色\n灰色\n粉色"
            ),
            "reasoning_content": "",
            "finish_reasons": ["stop"],
            "tool_calls": [],
        }


class ResponseClient:

    def __init__(self, finish_reason="length", completion_tokens=1):
        self.finish_reason = finish_reason
        self.completion_tokens = completion_tokens

    def post(self, payload, timeout=None):
        return 200, {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "llm",
            "choices": [{
                "index": 0,
                "finish_reason": self.finish_reason,
                "message": {"role": "assistant", "content": "A"},
            }],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": self.completion_tokens,
                "total_tokens": 3 + self.completion_tokens,
            },
        }


class ToolClient:

    def __init__(self, city="北京"):
        self.city = city
        self.payloads = []

    def post(self, payload, timeout=None):
        self.payloads.append(payload)
        return 200, {
            "id": "chatcmpl-tool-test",
            "object": "chat.completion",
            "created": 1,
            "model": "llm",
            "choices": [{
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call-test",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": self.city}),
                        },
                    }],
                },
            }],
            "usage": {
                "prompt_tokens": 32,
                "completion_tokens": 8,
                "total_tokens": 40,
            },
        }


class QualityGateApiTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.manifest_sha = MODULE._load_manifest(
            ROOT / "quality/official_metrics_manifest.v1.json")

    def test_handlers_exactly_cover_frozen_manifest(self):
        case_ids = {case["id"] for case in self.manifest["cases"]}
        self.assertEqual(set(MODULE.HANDLERS), case_ids)
        self.assertEqual(len(case_ids), 53)
        self.assertEqual(len(self.manifest_sha), 64)

    def test_tier_selection_is_stable(self):
        self.assertEqual(
            len(MODULE._selected_cases(self.manifest, "quick", [])), 30)
        self.assertEqual(
            len(MODULE._selected_cases(self.manifest, "full", [])), 52)
        self.assertEqual(
            len(MODULE._selected_cases(self.manifest, "extended", [])), 53)

    def test_progress_exposes_only_fixed_case_identity(self):
        case = {
            "ordinal": 53,
            "id": "exact_output_truncation",
            "private_prompt": "must not enter progress",
        }
        progress = MODULE._progress("running", 52, case)
        self.assertEqual(progress, {
            "state": "running",
            "completed_cases": 52,
            "active_ordinal": 53,
            "active_id": "exact_output_truncation",
        })
        self.assertNotIn("private_prompt", json.dumps(progress))

    def test_streaming_usage_is_validated_and_only_digest_is_retained(self):
        config = object()
        observation = MODULE._streaming_usage(StreamingClient(), config)
        self.assertEqual(observation["completion_tokens"], [4])
        self.assertTrue(observation["facts"]["completion_tokens_positive"])
        self.assertEqual(len(observation["semantic_output_sha256"]), 64)
        serialized = json.dumps(observation, ensure_ascii=False)
        self.assertNotIn("红色", serialized)
        self.assertNotIn("绿色", serialized)

    def test_tool_arguments_are_parsed_before_digest(self):
        message = {
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city":"北京"}',
                },
            }],
        }
        self.assertEqual(MODULE._normalized_tool_calls(message), [{
            "name": "get_weather", "arguments": {"city": "北京"},
        }])

    def test_auto_and_named_tool_cases_preserve_disabled_thinking(self):
        named = ToolClient()
        named_observation = MODULE._forced_tool(named, object())
        self.assertFalse(named.payloads[0]["thinking"])
        self.assertEqual(
            named_observation["facts"]["tool_choice_mode"], "named")

        auto = ToolClient()
        auto_observation = MODULE._auto_tool(auto, object())
        self.assertFalse(auto.payloads[0]["thinking"])
        self.assertEqual(auto.payloads[0]["tool_choice"], "auto")
        self.assertEqual(
            auto_observation["facts"]["tool_choice_mode"], "auto")

        with self.assertRaisesRegex(MODULE.CaseFailure, "city argument"):
            MODULE._auto_tool(ToolClient(city="上海"), object())

    def test_observation_does_not_retain_raw_semantic_value(self):
        secret = "raw-model-output-must-not-be-retained"
        observation = MODULE._observation(
            [(200, {"choices": [], "usage": {"completion_tokens": 1}})],
            [{"content": secret}],
        )
        self.assertNotIn(secret, json.dumps(observation))
        self.assertEqual(len(observation["semantic_output_sha256"]), 64)

    def test_strict_sse_parser_requires_final_usage_then_done(self):
        def chunk(choices, usage=None):
            value = {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "llm",
                "choices": choices,
            }
            if usage is not None:
                value["usage"] = usage
            return f"data: {json.dumps(value)}\n\n"

        body = "".join([
            chunk([{"index": 0, "delta": {"role": "assistant"},
                    "finish_reason": None}]),
            chunk([{"index": 0, "delta": {"content": "A"},
                    "finish_reason": None}]),
            chunk([{"index": 0, "delta": {}, "finish_reason": "stop"}]),
            chunk([], {
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "total_tokens": 4,
            }),
            "data: [DONE]\n\n",
        ]).encode()
        parsed = MODULE._parse_sse_payload(body)
        self.assertEqual(parsed["content"], "A")
        self.assertEqual(parsed["usage"]["completion_tokens"], 1)

        wrong_order = body.replace(
            b"data: [DONE]\n\n", b"data: [DONE]\n\ndata: late\n\n")
        with self.assertRaisesRegex(MODULE.CaseFailure, "does not end"):
            MODULE._parse_sse_payload(wrong_order)

    def test_strict_sse_parser_assembles_fragmented_tool_identity(self):
        def chunk(choices, usage=None):
            value = {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "llm",
                "choices": choices,
            }
            if usage is not None:
                value["usage"] = usage
            return f"data: {json.dumps(value)}\n\n"

        body = "".join([
            chunk([{
                "index": 0,
                "delta": {"role": "assistant", "tool_calls": [{
                    "index": 0,
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "arguments": "{\"command\":",
                    },
                }]},
                "finish_reason": None,
            }]),
            chunk([{
                "index": 0,
                "delta": {"tool_calls": [{
                    "index": 0,
                    "function": {"arguments": "\"pwd\"}"},
                }]},
                "finish_reason": None,
            }]),
            chunk([{
                "index": 0,
                "delta": {},
                "finish_reason": "tool_calls",
            }]),
            chunk([], {
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "total_tokens": 7,
            }),
            "data: [DONE]\n\n",
        ]).encode()

        parsed = MODULE._parse_sse_payload(body)
        self.assertEqual(parsed["finish_reasons"], ["tool_calls"])
        self.assertEqual(parsed["tool_calls"], [{
            "name": "terminal",
            "arguments": {"command": "pwd"},
        }])

        missing_identity = body.replace(
            b'"id": "call-1", "type": "function", ', b"", 1)
        with self.assertRaisesRegex(MODULE.CaseFailure, "identity is invalid"):
            MODULE._parse_sse_payload(missing_identity)

        conflicting_identity = body.replace(
            b'"index": 0, "function": {"arguments": "\\\"pwd\\\"}"}',
            b'"index": 0, "id": "call-2", '
            b'"function": {"arguments": "\\\"pwd\\\"}"}',
            1,
        )
        with self.assertRaisesRegex(MODULE.CaseFailure, "identity changed"):
            MODULE._parse_sse_payload(conflicting_identity)

    def test_response_schema_rejects_missing_usage(self):
        response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "llm",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "A"},
            }],
        }
        with self.assertRaisesRegex(MODULE.CaseFailure, "usage is missing"):
            MODULE._validate_response_schema(response)

    def test_response_schema_requires_exact_usage_sum(self):
        response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "llm",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "A"},
            }],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "total_tokens": 5,
            },
        }
        with self.assertRaisesRegex(MODULE.CaseFailure, "inconsistent"):
            MODULE._validate_response_schema(response)

    def test_max_tokens_one_accepts_length_and_enforces_usage_cap(self):
        observation = MODULE._max_tokens_case(
            ResponseClient(), object(), 1)
        self.assertEqual(observation["finish_reasons"], ["length"])
        self.assertTrue(observation["facts"]["completion_within_limit"])

        with self.assertRaisesRegex(MODULE.CaseFailure, "usage is invalid"):
            MODULE._max_tokens_case(
                ResponseClient(completion_tokens=2), object(), 1)

    def test_manifest_file_hash_is_frozen(self):
        value = json.loads((
            ROOT / "quality/official_metrics_manifest.v1.json"
        ).read_text(encoding="utf-8"))
        value["cases"][0]["comparison"] = "contract"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(
                    MODULE.CaseFailure, "file identity is invalid"):
                MODULE._load_manifest(path)


if __name__ == "__main__":
    unittest.main()
