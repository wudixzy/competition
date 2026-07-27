import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "tests" / "probe_chat_request_compat.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "probe_chat_request_compat_unit", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ChatRequestCompatProbeUnitTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.probe = _load_probe()

    def test_matrix_has_fixed_unique_synthetic_cases(self):
        names = [case["name"] for case in self.probe.CASES]
        self.assertEqual(len(names), len(set(names)))
        self.assertGreaterEqual(len(names), 10)
        self.assertIn("function_tool_strict_false", names)
        self.assertIn("function_tool_strict_true", names)
        self.assertIn("tool_choice_required", names)
        self.assertNotIn("evaluation", repr(self.probe.CASES).lower())

    def test_semantic_fail_closed_expectations_are_explicit(self):
        expectations = {
            case["name"]: case["expected"] for case in self.probe.CASES
        }
        self.assertEqual(
            expectations["function_tool_strict_false"], "accept")
        self.assertEqual(
            expectations["function_tool_strict_true"], "reject")
        self.assertEqual(expectations["tool_choice_required"], "reject")

    def test_bounded_errors_never_include_input_or_context(self):
        class FakeValidationError(Exception):

            def errors(self, **kwargs):
                self.kwargs = kwargs
                return [{
                    "loc": ("body", "tools", 0, "function", "strict"),
                    "type": "value_error",
                    "input": "private",
                    "ctx": {"private": "value"},
                }]

        error = FakeValidationError()
        bounded = self.probe._bounded_errors(error)
        self.assertFalse(error.kwargs["include_input"])
        self.assertFalse(error.kwargs["include_context"])
        self.assertEqual(bounded, [{
            "location": ["body", "tools", 0, "function", "strict"],
            "type": "value_error",
        }])


if __name__ == "__main__":
    unittest.main()
