import ast
import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHAT_UTILS = ROOT / "qwen3_6_scripts" / "chat_utils.py"


def _load_postprocess():
    tree = ast.parse(CHAT_UTILS.read_text())
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_postprocess_messages"
    ]
    if len(functions) != 1:
        raise AssertionError("_postprocess_messages is missing or ambiguous")
    module = ast.fix_missing_locations(
        ast.Module(body=functions, type_ignores=[]))
    namespace = {
        "json": json,
        "List": list,
        "ConversationMessage": dict,
    }
    exec(compile(module, str(CHAT_UTILS), "exec"), namespace)
    return namespace["_postprocess_messages"]


def _messages(arguments):
    return [{
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_synthetic",
            "type": "function",
            "function": {
                "name": "lookup",
                "arguments": arguments,
            },
        }],
    }]


class ChatUtilsToolHistoryUnitTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.postprocess = staticmethod(_load_postprocess())

    def test_json_object_string_is_decoded(self):
        messages = _messages('{"key":"synthetic"}')
        self.postprocess(messages)
        self.assertEqual(
            messages[0]["tool_calls"][0]["function"]["arguments"],
            {"key": "synthetic"},
        )

    def test_predecoded_object_is_preserved(self):
        arguments = {"key": "synthetic"}
        messages = _messages(arguments)
        self.postprocess(messages)
        self.assertIs(
            messages[0]["tool_calls"][0]["function"]["arguments"],
            arguments,
        )

    def test_invalid_json_and_non_objects_fail_closed(self):
        for arguments in ("{invalid", "[]", [], 3, None):
            with self.subTest(arguments=arguments):
                with self.assertRaises((TypeError, ValueError,
                                        json.JSONDecodeError)):
                    self.postprocess(_messages(arguments))


if __name__ == "__main__":
    unittest.main()
