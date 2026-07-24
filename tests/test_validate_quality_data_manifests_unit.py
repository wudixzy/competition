from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests/validate_quality_data_manifests.py"
SPEC = importlib.util.spec_from_file_location("quality_data_manifests", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class QualityDataManifestTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.provenance = json.loads((
            ROOT / "quality/source_provenance.v1.json"
        ).read_text(encoding="utf-8"))
        cls.matrix = json.loads((
            ROOT / "quality/long_context_matrix.v2.json"
        ).read_text(encoding="utf-8"))

    def test_frozen_manifests_are_valid(self):
        self.assertEqual(MODULE.validate_provenance(self.provenance), [])
        self.assertEqual(MODULE.validate_matrix(self.matrix), [])
        self.assertEqual(len(self.provenance["sources"]), 7)
        self.assertEqual(len(self.matrix["cases"]), 12)
        self.assertEqual(
            hashlib.sha256((
                ROOT / "quality/long_context_matrix.v2.json"
            ).read_bytes()).hexdigest(),
            MODULE.EXPECTED_MATRIX_SHA256,
        )

    def test_operator_files_match_frozen_identity(self):
        metrics = ROOT.parent / "指标集合"
        workload = ROOT.parent / "数据集特征/数据集特征.pdf"
        if not metrics.is_file() or not workload.is_file():
            self.skipTest("external operator source files are unavailable")
        self.assertEqual(
            MODULE.validate_operator_files(metrics, workload), [])

    def test_floating_external_revision_fails(self):
        value = copy.deepcopy(self.provenance)
        source = next(
            row for row in value["sources"] if row["id"] == "google_ifeval")
        source["revision"] = "main"
        self.assertIn(
            "external candidate contract differs for google_ifeval",
            MODULE.validate_provenance(value),
        )

    def test_deferred_license_cannot_be_marked_redistributable(self):
        value = copy.deepcopy(self.provenance)
        source = next(
            row for row in value["sources"]
            if row["id"] == "swe_bench_verified")
        source["redistribution_allowed"] = True
        self.assertIn(
            "SWE-bench must remain deferred pending license review",
            MODULE.validate_provenance(value),
        )

    def test_context_overflow_and_missing_capability_fail(self):
        value = copy.deepcopy(self.matrix)
        value["cases"][-1]["max_tokens"] = 257
        value["cases"] = [
            {**case, "capabilities": [
                item for item in case["capabilities"] if item != "multimodal"
            ]}
            for case in value["cases"]
        ]
        reasons = MODULE.validate_matrix(value)
        self.assertIn("matrix case 12 token budget is invalid", reasons)
        self.assertIn(
            "matrix does not cover every required capability", reasons)

    def test_235k_large_output_budget_is_frozen(self):
        value = copy.deepcopy(self.matrix)
        case = next(
            row for row in value["cases"]
            if row["id"] == "235k_agent_large_output_budget")
        case["max_tokens"] = 1024
        self.assertIn(
            "235K Agent case must retain the official large output budget",
            MODULE.validate_matrix(value),
        )


if __name__ == "__main__":
    unittest.main()
