import ast
import pathlib
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
API_SERVER = ROOT / "qwen3_6_scripts/api_server.py"
HELPERS = {
    "_bi100_field",
    "_bi100_image_source_kind",
    "_bi100_scalar",
    "_bi100_tool_choice_kind",
    "_bi100_chat_4xx_reason",
    "_bi100_chat_request_shape",
    "_bi100_validation_reason",
}


def _load_helpers():
    tree = ast.parse(API_SERVER.read_text())
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in HELPERS
    ]
    if {node.name for node in functions} != HELPERS:
        raise AssertionError("4xx telemetry helpers are incomplete")
    module = ast.fix_missing_locations(
        ast.Module(body=functions, type_ignores=[]))
    namespace = {}
    exec(compile(module, str(API_SERVER), "exec"), namespace)
    return namespace


class Api4xxTelemetryTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        helpers = _load_helpers()
        cls.reason = staticmethod(helpers["_bi100_chat_4xx_reason"])
        cls.image_source_kind = staticmethod(
            helpers["_bi100_image_source_kind"])
        cls.shape = staticmethod(helpers["_bi100_chat_request_shape"])
        cls.validation_reason = staticmethod(
            helpers["_bi100_validation_reason"])

    def test_known_errors_have_fixed_reason_codes(self):
        self.assertEqual(
            self.reason("messages must contain at least one message"),
            "empty_messages",
        )
        self.assertEqual(
            self.reason("n=2 exceeds max_num_seqs=1. Use n<=1 or omit n."),
            "n_exceeds_max_num_seqs",
        )
        self.assertEqual(
            self.reason('tool_choice = "required" is not supported!'),
            "unsupported_tool_choice_required",
        )
        self.assertEqual(
            self.reason("Tool call arguments are not valid JSON."),
            "invalid_tool_arguments_json",
        )
        self.assertEqual(
            self.reason(
                "Tool call arguments must decode to a JSON object."),
            "invalid_tool_arguments_type",
        )
        self.assertEqual(
            self.reason(
                "At most 1 image(s) may be provided in one request."),
            "image_count_limit",
        )
        self.assertEqual(
            self.reason(
                "You set image=1 (or defaulted to 1) in "
                "`--limit-mm-per-prompt`, but found 2 items "
                "in the same prompt."),
            "image_count_limit",
        )
        self.assertEqual(
            self.reason("Unknown model type: qwen3_5_moe"),
            "image_model_type_unsupported",
        )

    def test_image_source_kinds_are_bounded(self):
        self.assertEqual(
            self.image_source_kind("data:image/png;base64,private"), "data")
        self.assertEqual(
            self.image_source_kind("https://private.example/image"), "remote")
        self.assertEqual(
            self.image_source_kind("file:///private/image"), "other")
        self.assertEqual(self.image_source_kind(None), "other")

    def test_unknown_error_does_not_enter_reason_code(self):
        sensitive = "template failed for private prompt contents"
        reason = self.reason(sensitive)
        self.assertEqual(reason, "unclassified_chat_error")
        self.assertNotIn("private", reason)

    def test_request_shape_retains_only_non_sensitive_counts(self):
        request = types.SimpleNamespace(
            messages=[
                {"role": "system", "content": "private system prompt"},
                types.SimpleNamespace(
                    role="user",
                    content=[
                        {"type": "text", "text": "private user prompt"},
                        {"type": "image_url",
                         "image_url": {
                             "url": "data:image/png;base64,private",
                         }},
                        {"type": "image_url",
                         "image_url": {
                             "url": "https://private.example/image",
                         }},
                        {"type": "image",
                         "private": "already parsed image"},
                    ],
                ),
            ],
            tools=[{"function": {"name": "private_tool"}}],
            tool_choice={
                "type": "function",
                "function": {"name": "private_tool"},
            },
            stream=True,
            n=2,
        )
        self.assertEqual(self.shape(request), {
            "message_count": 2,
            "system_count": 1,
            "system_part_message_count": 0,
            "system_text_part_count": 0,
            "system_other_part_count": 0,
            "tool_count": 1,
            "tool_message_count": 0,
            "assistant_tool_message_count": 0,
            "strict_false_count": 0,
            "strict_true_count": 0,
            "tool_choice_kind": "named",
            "image_count": 3,
            "image_data_count": 1,
            "image_remote_count": 1,
            "image_other_count": 1,
            "has_image": True,
            "stream": True,
            "n": 2,
        })

    def test_shape_counts_tool_history_and_strict_without_values(self):
        shape = self.shape({
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"private": "value"}],
                },
                {
                    "role": "tool",
                    "tool_call_id": "private-id",
                    "content": "private result",
                },
            ],
            "tools": [
                {"function": {"strict": False, "private": "schema"}},
                {"function": {"strict": True, "private": "schema"}},
            ],
            "tool_choice": "required",
        })
        self.assertEqual(shape["tool_message_count"], 1)
        self.assertEqual(shape["assistant_tool_message_count"], 1)
        self.assertEqual(shape["strict_false_count"], 1)
        self.assertEqual(shape["strict_true_count"], 1)
        self.assertEqual(shape["tool_choice_kind"], "required")
        self.assertNotIn("private", repr(shape))

    def test_shape_counts_system_parts_without_text(self):
        shape = self.shape({
            "messages": [{
                "role": "system",
                "content": [
                    {"type": "text", "text": "private"},
                    {"type": "other", "private": "value"},
                ],
            }],
        })
        self.assertEqual(shape["system_part_message_count"], 1)
        self.assertEqual(shape["system_text_part_count"], 1)
        self.assertEqual(shape["system_other_part_count"], 1)
        self.assertNotIn("private", repr(shape))

    def test_validation_errors_have_bounded_specific_reason_codes(self):
        cases = (
            (("body", "tools", 0, "function", "strict"),
             "request_validation_tool_strict"),
            (("body", "tools", 0, "function", "parameters"),
             "request_validation_tool_parameters"),
            (("body", "tool_choice"),
             "request_validation_tool_choice"),
            (("body", "messages", 1, "tool_call_id"),
             "request_validation_message_tool_call_id"),
            (("body", "messages", 1, "tool_calls"),
             "request_validation_message_tool_calls"),
            (("body", "messages", 1, "content"),
             "request_validation_message_content"),
        )
        for location, expected in cases:
            with self.subTest(location=location):
                self.assertEqual(
                    self.validation_reason([{
                        "loc": location,
                        "input": "private",
                    }]),
                    expected,
                )

    def test_runtime_logs_reason_without_raw_error_message(self):
        source = API_SERVER.read_text()
        self.assertIn("[BI100 4XX] endpoint=chat", source)
        self.assertIn("[BI100 4XX] endpoint=request_validation", source)
        self.assertIn("_bi100_validation_reason(validation_errors)", source)
        self.assertIn("_bi100_chat_request_shape(body)", source)
        self.assertIn("_bi100_log_chat_4xx(request, generator)", source)
        self.assertIn("images=%d image_data=%d image_remote=%d", source)
        self.assertNotIn("[BI100 4XX] message=", source)


if __name__ == "__main__":
    unittest.main()
