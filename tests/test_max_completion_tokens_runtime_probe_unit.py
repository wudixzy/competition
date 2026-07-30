import ast
import asyncio
import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBE = ROOT / "tests" / "probe_max_completion_tokens_runtime.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "max_completion_tokens_runtime_probe_unit", PROBE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MaxCompletionTokensRuntimeProbeUnitTest(unittest.TestCase):

    def test_probe_is_synthetic_and_covers_required_paths(self):
        source = PROBE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIsNotNone(tree)
        for case in (
            "completion_only_nonstream",
            "completion_only_stream",
            "completion_with_tools",
            "completion_with_multimodal",
            "completion_with_reasoning_switch",
            "legacy_only",
            "both_new_field_precedes",
            "invalid_completion_type",
            "invalid_completion_boundary",
            "unrelated_unknown_field",
        ):
            self.assertIn(f'"name": "{case}"', source)
        self.assertIn("_asgi_post_json(", source)
        self.assertNotIn("import httpx", source)
        self.assertIn("request.to_sampling_params(4096)", source)
        self.assertIn('"synthetic_only": True', source)

    def test_probe_reports_only_bounded_request_shape_facts(self):
        source = PROBE.read_text(encoding="utf-8")
        self.assertIn('"has_tools": bool(request.tools)', source)
        self.assertIn('"has_multimodal": _contains_multimodal_message(',
                      source)
        self.assertNotIn('"messages": request.messages', source)
        self.assertNotIn('"tools": request.tools', source)

    def test_stdlib_asgi_client_posts_json_and_collects_response(self):
        probe = _load_probe()

        async def app(scope, receive, send):
            request = await receive()
            self.assertEqual(scope["path"], "/probe")
            self.assertIn(b'"value":7', request["body"])
            await send({
                "type": "http.response.start",
                "status": 201,
                "headers": [],
            })
            await send({
                "type": "http.response.body",
                "body": b"ok",
                "more_body": False,
            })

        status, body = asyncio.run(
            probe._asgi_post_json(app, "/probe", {"value": 7}))
        self.assertEqual(status, 201)
        self.assertEqual(body, b"ok")


if __name__ == "__main__":
    unittest.main()
