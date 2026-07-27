import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tests" / "qualify_chat_request_compat.py"
SPEC = importlib.util.spec_from_file_location(
    "qualify_chat_request_compat", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class QualifyChatRequestCompatUnitTest(unittest.TestCase):

    def test_tool_history_pairs_only_change_argument_representation(self):
        as_string = MODULE._tool_history('{"key":"synthetic"}')
        as_object = MODULE._tool_history({"key": "synthetic"})
        string_function = as_string["messages"][1]["tool_calls"][0]["function"]
        object_function = as_object["messages"][1]["tool_calls"][0]["function"]
        self.assertEqual(
            {**string_function, "arguments": None},
            {**object_function, "arguments": None},
        )
        self.assertIsInstance(string_function["arguments"], str)
        self.assertIsInstance(object_function["arguments"], dict)

    def test_strict_pairs_only_change_noop_field(self):
        omitted = MODULE._strict_payload(None)
        explicit_false = MODULE._strict_payload(False)
        omitted_function = omitted["tools"][0]["function"]
        false_function = explicit_false["tools"][0]["function"]
        self.assertNotIn("strict", omitted_function)
        self.assertIs(false_function["strict"], False)
        self.assertEqual(
            omitted_function,
            {
                key: value for key, value in false_function.items()
                if key != "strict"
            },
        )

    def test_system_text_part_pair_has_one_canonical_meaning(self):
        parts = MODULE._system_parts_payload(normalized=False)
        normalized = MODULE._system_parts_payload(normalized=True)
        self.assertEqual(
            normalized["messages"][0]["content"],
            "synthetic rule A1\nsynthetic rule A2\n\nsynthetic rule B",
        )
        self.assertIsInstance(parts["messages"][1]["content"], list)
        self.assertEqual(
            [message["role"] for message in parts["messages"]],
            ["user", "system", "system"],
        )
        self.assertEqual(
            [message["role"] for message in normalized["messages"]],
            ["system", "user"],
        )

    def test_report_source_contains_no_raw_tokenizer_output(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn("contains_prompt_or_response_text", source)
        self.assertNotIn("decode(", source)
        self.assertNotIn("\"tokens\": tokens", source)


if __name__ == "__main__":
    unittest.main()
