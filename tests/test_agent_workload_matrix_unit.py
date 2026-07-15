import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tests" / "agent_workload_matrix.py"
SPEC = importlib.util.spec_from_file_location("agent_workload_matrix", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class AgentWorkloadMatrixUnitTest(unittest.TestCase):

    def test_matrix_covers_agent_contracts(self):
        cases = MODULE.build_cases()
        self.assertEqual(len(cases), 9)
        self.assertEqual(
            cases["auto_terminal"]["expected_tool"], "terminal")
        self.assertEqual(
            len(cases["large_tool_schema"]["payload"]["tools"]), 92)
        self.assertGreaterEqual(
            len(cases["long_history"]["payload"]["messages"]), 42)
        self.assertIn(
            "tool", {
                message["role"]
                for message in cases["tool_result_roundtrip"]["payload"]["messages"]
            })

    def test_argument_parser_accepts_string_and_object(self):
        self.assertEqual(MODULE.parse_arguments('{"value": 7}'), {"value": 7})
        self.assertEqual(MODULE.parse_arguments({"value": 7}), {"value": 7})


if __name__ == "__main__":
    unittest.main()
