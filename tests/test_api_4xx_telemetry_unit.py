import ast
import pathlib
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
API_SERVER = ROOT / "qwen3_6_scripts/api_server.py"
HELPERS = {
    "_bi100_field",
    "_bi100_scalar",
    "_bi100_chat_4xx_reason",
    "_bi100_chat_request_shape",
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
        cls.shape = staticmethod(helpers["_bi100_chat_request_shape"])

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
                         "image_url": {"url": "private image bytes"}},
                    ],
                ),
            ],
            tools=[{"function": {"name": "private_tool"}}],
            stream=True,
            n=2,
        )
        self.assertEqual(self.shape(request), {
            "message_count": 2,
            "system_count": 1,
            "tool_count": 1,
            "has_image": True,
            "stream": True,
            "n": 2,
        })

    def test_runtime_logs_reason_without_raw_error_message(self):
        source = API_SERVER.read_text()
        self.assertIn("[BI100 4XX] endpoint=chat", source)
        self.assertIn("reason=request_validation", source)
        self.assertIn("_bi100_log_chat_4xx(request, generator)", source)
        self.assertNotIn("[BI100 4XX] message=", source)


if __name__ == "__main__":
    unittest.main()
