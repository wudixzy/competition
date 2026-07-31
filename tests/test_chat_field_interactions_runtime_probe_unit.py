import ast
import asyncio
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "tests" / "probe_chat_field_interactions_runtime.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "chat_field_interactions_runtime_probe_unit",
        PROBE,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ChatFieldInteractionsRuntimeProbeUnitTest(unittest.TestCase):

    def test_probe_covers_alias_dependency_conflict_and_fail_closed_cases(self):
        source = PROBE.read_text(encoding="utf-8")
        self.assertIsNotNone(ast.parse(source))
        for case in (
            "completion_budget_precedence",
            "thinking_precedence",
            "empty_stream_options_without_stream",
            "top_logprobs_zero_is_noop",
            "positive_top_logprobs_requires_logprobs",
            "null_tool_choice",
            "malformed_tool_choice",
            "missing_response_schema",
            "valid_response_schema",
            "multiple_output_constraints",
            "continue_uses_default_generation_prompt",
            "legacy_function_call_fail_closed",
            "legacy_functions_fail_closed",
            "responses_max_output_tokens_fail_closed",
            "reasoning_effort_fail_closed",
            "malformed_logprob_type",
        ):
            self.assertIn(f'"name": "{case}"', source)
        self.assertIn('"http_500_count"', source)
        self.assertIn('"synthetic_only": True', source)
        self.assertNotIn("import httpx", source)

    def test_probe_records_only_bounded_protocol_facts(self):
        source = PROBE.read_text(encoding="utf-8")
        for forbidden in (
            '"messages": request.messages',
            '"tools": request.tools',
            '"body": response_body',
            '"payload": case["payload"]',
        ):
            self.assertNotIn(forbidden, source)

    def test_stdlib_asgi_client_handles_non_object_json(self):
        probe = _load_probe()

        async def app(scope, receive, send):
            request = await receive()
            self.assertEqual(scope["path"], "/probe")
            self.assertEqual(request["body"], b"[]")
            await send({
                "type": "http.response.start",
                "status": 400,
                "headers": [],
            })
            await send({
                "type": "http.response.body",
                "body": b"bad request",
                "more_body": False,
            })

        status, body = asyncio.run(
            probe._asgi_post_json(app, "/probe", []))
        self.assertEqual(status, 400)
        self.assertEqual(body, b"bad request")


if __name__ == "__main__":
    unittest.main()
