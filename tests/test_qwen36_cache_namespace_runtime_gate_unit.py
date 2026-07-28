from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tests" / "qwen36_cache_namespace_runtime_gate.py"
SPEC = importlib.util.spec_from_file_location(
    "qwen36_cache_namespace_runtime_gate", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CacheNamespaceRuntimeGateUnitTest(unittest.TestCase):

    def test_v3_contract_covers_empty_mapping_and_request_swap(self):
        self.assertEqual(
            MODULE.SCHEMA, "qwen36-cache-namespace-runtime-gate-v3")
        self.assertEqual(MODULE.VERSION, 3)
        self.assertIn(
            "empty_multimodal_matches_text", MODULE.REQUIRED_CHECKS)
        self.assertIn(
            "request_swap_preserves_namespace", MODULE.REQUIRED_CHECKS)
        self.assertNotIn(
            "empty_multimodal_separated_from_text",
            MODULE.REQUIRED_CHECKS,
        )

    def test_exact_success_shape_qualifies(self):
        checks = {name: True for name in MODULE.REQUIRED_CHECKS}
        qualified, reasons = MODULE.qualify_checks(checks, {})
        self.assertTrue(qualified)
        self.assertEqual(reasons, [])

    def test_missing_extra_or_failed_check_rejects(self):
        checks = {name: True for name in MODULE.REQUIRED_CHECKS}
        checks.pop(MODULE.REQUIRED_CHECKS[-1])
        qualified, reasons = MODULE.qualify_checks(checks, {})
        self.assertFalse(qualified)
        self.assertIn(
            "runtime check order or identity differs", reasons)
        self.assertIn(
            "runtime check failed: request_swap_preserves_namespace",
            reasons,
        )

        checks = {name: True for name in MODULE.REQUIRED_CHECKS}
        checks["unexpected"] = True
        qualified, reasons = MODULE.qualify_checks(checks, {})
        self.assertFalse(qualified)
        self.assertIn(
            "runtime check order or identity differs", reasons)

        checks = {name: True for name in MODULE.REQUIRED_CHECKS}
        checks["different_palette_isolated"] = False
        qualified, reasons = MODULE.qualify_checks(checks, {})
        self.assertFalse(qualified)
        self.assertIn(
            "runtime check failed: different_palette_isolated", reasons)

    def test_any_runtime_exception_rejects_without_message(self):
        checks = {name: True for name in MODULE.REQUIRED_CHECKS}
        qualified, reasons = MODULE.qualify_checks(
            checks, {"same_palette_stable": "OSError"})
        self.assertFalse(qualified)
        self.assertIn(
            "one or more runtime checks raised an exception", reasons)

    def test_atomic_report_write_round_trips(self):
        value = {
            "schema": MODULE.SCHEMA,
            "version": MODULE.VERSION,
            "qualified": False,
            "reasons": ["synthetic"],
            "privacy": {
                "contains_image_bytes": False,
                "contains_namespace_digest": False,
                "contains_request_id": False,
                "contains_prompt_or_output": False,
                "contains_credentials": False,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "gate.json"
            MODULE._atomic_write(output, value)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")), value)
            self.assertEqual(
                list(output.parent.glob(f".{output.name}.*.tmp")), [])

    def test_initialization_failure_is_redacted_and_fail_closed(self):
        report = MODULE.failure_report(
            Path("/tmp/runtime"),
            "not-a-revision-or-a-secret",
            "ImportError",
        )
        self.assertFalse(report["qualified"])
        self.assertIsNone(report["source_revision"])
        self.assertEqual(
            report["error_types"], {"initialization": "ImportError"})
        rendered = json.dumps(report, sort_keys=True)
        self.assertNotIn("not-a-revision-or-a-secret", rendered)
        self.assertNotIn("traceback", rendered.lower())
        self.assertFalse(
            report["privacy"]["contains_exception_message"])

    def test_revision_contract_requires_full_lowercase_digest(self):
        self.assertTrue(MODULE._valid_revision("a" * 40))
        self.assertTrue(MODULE._valid_revision("b" * 64))
        self.assertFalse(MODULE._valid_revision("A" * 40))
        self.assertFalse(MODULE._valid_revision("a" * 39))
        self.assertFalse(MODULE._valid_revision("token"))

    def test_report_source_never_serializes_sensitive_payloads(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        for field in (
                '"contains_image_bytes": False',
                '"contains_namespace_digest": False',
                '"contains_request_id": False',
                '"contains_prompt_or_output": False',
                '"contains_credentials": False'):
            self.assertIn(field, source)
        self.assertNotIn('"image_bytes":', source)
        self.assertNotIn('"namespace_digest":', source)


if __name__ == "__main__":
    unittest.main()
