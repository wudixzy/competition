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

    def test_tool_arguments_string_is_not_double_json_encoded(self):
        arguments = '{"city": "上海", "unit": "c"}'
        self.assertEqual(self.serialize(arguments), arguments)

    def test_tool_arguments_structured_values_are_json_encoded(self):
        self.assertEqual(json.loads(self.serialize({"city": "上海"})),
                         {"city": "上海"})
        self.assertEqual(json.loads(self.serialize(["a", "b"])), ["a", "b"])

    def test_tool_arguments_none_defaults_to_empty_object(self):
        self.assertEqual(self.serialize(None), "{}")

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
