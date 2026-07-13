import ast
import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVING_CHAT = ROOT / "qwen3_6_scripts" / "serving_chat.py"
SERVING_CHAT_SOURCE = SERVING_CHAT.read_text()


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


if __name__ == "__main__":
    unittest.main()
