import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "tests" / "audit_chat_request_compat_fields.py"


def _load_audit():
    spec = importlib.util.spec_from_file_location(
        "chat_request_compat_field_audit_unit", AUDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ChatRequestCompatFieldAuditUnitTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.audit_module = _load_audit()

    def test_field_compatibility_contract_is_complete_and_strict(self):
        report = self.audit_module.audit(ROOT)
        self.assertTrue(report["qualified"], report["reasons"])
        self.assertEqual(report["strict_extra_policy"], "forbid")
        self.assertIn(
            "max_completion_tokens", report["lossless_alias_fields"])

    def test_non_alias_fields_remain_fail_closed(self):
        report = self.audit_module.audit(ROOT)
        self.assertIn(
            "reasoning_effort",
            report["review_required_non_alias_fields"],
        )
        self.assertIn(
            "max_output_tokens",
            report["review_required_non_alias_fields"],
        )

    def test_pinned_openai_fields_are_exhaustively_classified(self):
        report = self.audit_module.audit(ROOT)
        self.assertTrue(report["qualified"], report["reasons"])
        for field in (
            "functions",
            "reasoning_effort",
            "service_tier",
            "prompt_cache_key",
            "web_search_options",
        ):
            self.assertIn(field, report["classified_openai_only_fields"])


if __name__ == "__main__":
    unittest.main()
