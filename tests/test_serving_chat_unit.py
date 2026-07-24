import ast
import json
import pathlib
import types
import unittest
from typing import Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVING_CHAT = ROOT / "qwen3_6_scripts" / "serving_chat.py"
SERVING_CHAT_SOURCE = SERVING_CHAT.read_text()
CHAT_UTILS = ROOT / "qwen3_6_scripts" / "chat_utils.py"
CHAT_UTILS_SOURCE = CHAT_UTILS.read_text()
QWEN_MODEL = ROOT / "qwen3_6_scripts" / "qwen3_5.py"
QWEN_MODEL_SOURCE = QWEN_MODEL.read_text()


def _load_serialize_tool_arguments():
    tree = ast.parse(SERVING_CHAT.read_text(), filename=str(SERVING_CHAT))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_serialize_tool_arguments")
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"json": json}
    exec(compile(module, str(SERVING_CHAT), "exec"), namespace)
    return namespace["_serialize_tool_arguments"]


def _load_named_tool_delta_payload():
    tree = ast.parse(SERVING_CHAT.read_text(), filename=str(SERVING_CHAT))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_named_tool_delta_payload")
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"Dict": dict}
    exec(compile(module, str(SERVING_CHAT), "exec"), namespace)
    return namespace["_named_tool_delta_payload"]


def _load_named_tool_argument_helpers():
    tree = ast.parse(SERVING_CHAT.read_text(), filename=str(SERVING_CHAT))
    names = {
        "_serialize_tool_arguments",
        "_tool_arguments_are_json_object",
        "_select_named_tool_arguments",
    }
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "json": json,
        "List": list,
        "Optional": Optional,
        "ToolCall": object,
    }
    exec(compile(module, str(SERVING_CHAT), "exec"), namespace)
    return namespace


def _load_chat_placeholder_method():
    tree = ast.parse(CHAT_UTILS_SOURCE, filename=str(CHAT_UTILS))
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "BaseMultiModalItemTracker")
    function = next(
        node for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_placeholder_str")
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"ModalityStr": str, "Optional": Optional}
    exec(compile(module, str(CHAT_UTILS), "exec"), namespace)
    return namespace["_placeholder_str"]


class ServingChatUnitTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.serialize = staticmethod(_load_serialize_tool_arguments())
        cls.named_delta = staticmethod(_load_named_tool_delta_payload())
        helpers = _load_named_tool_argument_helpers()
        cls.select_named_arguments = staticmethod(
            helpers["_select_named_tool_arguments"])

    def test_tool_arguments_string_is_not_double_json_encoded(self):
        arguments = '{"city": "上海", "unit": "c"}'
        self.assertEqual(self.serialize(arguments), arguments)

    def test_tool_arguments_structured_values_are_json_encoded(self):
        self.assertEqual(json.loads(self.serialize({"city": "上海"})),
                         {"city": "上海"})
        self.assertEqual(json.loads(self.serialize(["a", "b"])), ["a", "b"])

    def test_tool_arguments_none_defaults_to_empty_object(self):
        self.assertEqual(self.serialize(None), "{}")

    def test_named_tool_calls_use_tool_finish_reason(self):
        self.assertIn("named_tool_called = False", SERVING_CHAT_SOURCE)
        self.assertIn("named_tool_called = True", SERVING_CHAT_SOURCE)
        self.assertIn(
            "auto_tools_called or named_tool_called", SERVING_CHAT_SOURCE)
        self.assertIn(
            "auto_tools_called or tool_choice_function_name",
            SERVING_CHAT_SOURCE,
        )

    def test_named_stream_tool_identity_is_stable_and_sent_once(self):
        first = self.named_delta(
            "terminal", '{"command":', 0, "call-stable", True)
        following = self.named_delta(
            "terminal", '"pwd"}', 0, "call-stable", False)

        self.assertEqual(first, {
            "id": "call-stable",
            "type": "function",
            "index": 0,
            "function": {
                "name": "terminal",
                "arguments": '{"command":',
            },
        })
        self.assertEqual(following, {
            "index": 0,
            "function": {"arguments": '"pwd"}'},
        })
        self.assertIn("named_tool_call_ids = (", SERVING_CHAT_SOURCE)
        self.assertIn("previous_num_tokens[i] == 0", SERVING_CHAT_SOURCE)

    def test_named_nonstream_keeps_valid_raw_json_exactly(self):
        raw = '{ "key" : "TOOLS-731" }'
        parsed = [types.SimpleNamespace(function=types.SimpleNamespace(
            name="lookup_quality_marker", arguments='{"key":"other"}'))]
        self.assertEqual(
            self.select_named_arguments(
                raw, "lookup_quality_marker", parsed),
            raw,
        )

    def test_named_nonstream_recovers_unique_same_name_parser_call(self):
        parsed = [types.SimpleNamespace(function=types.SimpleNamespace(
            name="report_agent_marker",
            arguments={"key": "AGENT-235K-731", "ordinal": 235000},
        ))]
        selected = self.select_named_arguments(
            "<tool_call>not-json</tool_call>",
            "report_agent_marker",
            parsed,
        )
        self.assertEqual(json.loads(selected), {
            "key": "AGENT-235K-731",
            "ordinal": 235000,
        })

    def test_named_nonstream_rejects_ambiguous_or_wrong_parser_call(self):
        raw = "<tool_call>not-json</tool_call>"
        wrong = [types.SimpleNamespace(function=types.SimpleNamespace(
            name="other", arguments='{"key":"x"}'))]
        ambiguous = wrong + wrong
        invalid = [types.SimpleNamespace(function=types.SimpleNamespace(
            name="report_agent_marker", arguments="not-json"))]
        for parsed in (wrong, ambiguous, invalid, None):
            with self.subTest(parsed=parsed):
                self.assertEqual(
                    self.select_named_arguments(
                        raw, "report_agent_marker", parsed),
                    raw,
                )

    def test_named_nonstream_invokes_parser_only_for_invalid_json(self):
        branch = SERVING_CHAT_SOURCE.index(
            "parsed_named_tool_calls: Optional[List[ToolCall]]")
        guard = SERVING_CHAT_SOURCE.index(
            "not _tool_arguments_are_json_object(output_text)", branch)
        parser_call = SERVING_CHAT_SOURCE.index(
            ".extract_tool_calls(", guard)
        selector = SERVING_CHAT_SOURCE.index(
            "named_arguments = _select_named_tool_arguments", parser_call)
        self.assertLess(branch, guard)
        self.assertLess(guard, parser_call)
        self.assertLess(parser_call, selector)

    def test_empty_messages_are_rejected_before_async_work(self):
        guard = 'if not request.messages:'
        guard_pos = SERVING_CHAT_SOURCE.index(guard)
        self.assertIn("messages must contain at least one message",
                      SERVING_CHAT_SOURCE)
        for later_operation in [
                "await self._check_model(request)",
                "await self.engine_client.get_tokenizer(lora_request)",
                "parse_chat_messages_futures(",
        ]:
            self.assertLess(
                guard_pos, SERVING_CHAT_SOURCE.index(later_operation))

    def test_qwen36_image_placeholder_uses_native_vision_tokens(self):
        placeholder_str = _load_chat_placeholder_method()
        tracker = types.SimpleNamespace(
            _model_config=types.SimpleNamespace(
                hf_config=types.SimpleNamespace(model_type="qwen3_5_moe")),
            _tokenizer=None,
        )
        self.assertEqual(
            placeholder_str(tracker, "image", 1),
            "<|vision_start|><|image_pad|><|vision_end|>",
        )

    def test_qwen36_cached_image_tokens_use_visual_suffix(self):
        self.assertIn("if num_placeholders:", QWEN_MODEL_SOURCE)
        self.assertIn("image_embeds[-num_placeholders:]", QWEN_MODEL_SOURCE)
        self.assertNotIn(
            "image token count ({num_placeholders}) does not match",
            QWEN_MODEL_SOURCE,
        )


if __name__ == "__main__":
    unittest.main()
