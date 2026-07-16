import pathlib
import sys
import unittest

TESTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
import submission_preflight


ROOT = TESTS.parent


class SubmissionPreflightTest(unittest.TestCase):

    def test_current_submission_passes_every_check(self):
        results = submission_preflight.run_checks(ROOT)
        failures = [item for item in results if not item["ok"]]
        self.assertEqual(failures, [])

    def test_run_config_parser_preserves_exact_command_order(self):
        text = (ROOT / "computility-run.yaml").read_text(encoding="utf-8")
        concurrency, command, environment = \
            submission_preflight.parse_run_config(text)
        self.assertEqual(concurrency, 1)
        self.assertEqual(command, submission_preflight.EXPECTED_COMMAND)
        self.assertEqual(environment, submission_preflight.EXPECTED_ENV)

    def test_run_config_parser_rejects_duplicate_environment_name(self):
        text = """\
concurrency: 1
command:
    - python3
env:
    - name: A
      value: 1
    - name: A
      value: 2
"""
        with self.assertRaisesRegex(ValueError, "duplicate"):
            submission_preflight.parse_run_config(text)


if __name__ == "__main__":
    unittest.main()
