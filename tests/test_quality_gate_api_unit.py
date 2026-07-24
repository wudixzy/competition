from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
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
