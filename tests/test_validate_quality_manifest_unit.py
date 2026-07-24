from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "quality/official_metrics_manifest.v1.json"
SCRIPT = ROOT / "tests/validate_quality_manifest.py"
SPEC = importlib.util.spec_from_file_location("quality_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class QualityManifestTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.value = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_frozen_manifest_is_valid(self):
        self.assertEqual(MODULE.validate(self.value), [])
        self.assertEqual(len(self.value["cases"]), 53)
        self.assertEqual(self.value["promotion_tier"], "extended")
        self.assertEqual(self.value["allowed_skips"], {"direct": ["n_2"]})
        self.assertEqual(
            hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            MODULE.EXPECTED_MANIFEST_SHA256,
        )

    def test_operator_source_is_bound_when_available(self):
        self.assertEqual(MODULE.validate_source(ROOT.parent / "指标集合"), [])

    def test_missing_case_fails(self):
        value = copy.deepcopy(self.value)
        value["cases"].pop()
        self.assertIn(
            "manifest must contain 53 cases",
            MODULE.validate(value),
        )

    def test_duplicate_id_and_weaker_promotion_tier_fail(self):
        value = copy.deepcopy(self.value)
        value["cases"][1]["id"] = value["cases"][0]["id"]
        value["promotion_tier"] = "quick"
        reasons = MODULE.validate(value)
        self.assertIn("case ids must be unique", reasons)
        self.assertIn("promotion tier must be extended", reasons)


if __name__ == "__main__":
    unittest.main()
