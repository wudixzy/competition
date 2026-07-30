import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBE = ROOT / "tests" / "probe_max_completion_tokens_runtime.py"


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
        self.assertIn("httpx.ASGITransport", source)
        self.assertIn("request.to_sampling_params(4096)", source)
        self.assertIn('"synthetic_only": True', source)

    def test_probe_reports_only_bounded_request_shape_facts(self):
        source = PROBE.read_text(encoding="utf-8")
        self.assertIn('"has_tools": bool(request.tools)', source)
        self.assertIn('"has_multimodal": _contains_multimodal_message(',
                      source)
        self.assertNotIn('"messages": request.messages', source)
        self.assertNotIn('"tools": request.tools', source)


if __name__ == "__main__":
    unittest.main()
