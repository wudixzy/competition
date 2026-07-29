from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/freeze_ifeval_power_subset.py"
SPEC = importlib.util.spec_from_file_location(
    "freeze_ifeval_power_subset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
EXTERNAL = ROOT / "quality/external/google_ifeval"
MANIFEST = EXTERNAL / "manifest.power149.v2.json"
SUBSET = EXTERNAL / "subset.power149.v2.jsonl"
MANIFEST_SHA256 = (
    "01c7e9dd4aafc11b5e2505fec2c3c71c53d8d27992ab40445638e97404440107"
)
SUBSET_SHA256 = (
    "14dee74f7fc65768d326140367b31b57cce24d59e76bd0098b94d2730eef22e2"
)


class FreezeIFEvalPowerSubsetTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.rows = [
            json.loads(line)
            for line in SUBSET.read_text(encoding="utf-8").splitlines()
        ]

    def test_source_and_selection_reproduce_frozen_power_subset(self) -> None:
        source = MODULE.BASE.load_source(MODULE.SOURCE_PATH)
        selected = MODULE.select_rows(source)
        self.assertEqual(selected, self.rows)
        self.assertEqual(len(selected), 149)
        self.assertEqual(
            [row["key"] for row in selected],
            self.manifest["selection"]["selected_keys_in_request_order"],
        )

    def test_all_instruction_ids_have_predeclared_coverage(self) -> None:
        counts = collections.Counter(
            item for row in self.rows for item in row["instruction_id_list"])
        self.assertEqual(
            set(counts), MODULE.BASE.EXPECTED_INSTRUCTION_IDS)
        self.assertGreaterEqual(min(counts.values()), 10)
        self.assertEqual(
            dict(sorted(counts.items())),
            self.manifest["selection"]["instruction_counts"],
        )

    def test_snapshot_and_generator_identities_are_frozen(self) -> None:
        self.assertEqual(
            hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            MANIFEST_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(SUBSET.read_bytes()).hexdigest(),
            SUBSET_SHA256,
        )
        generator = ROOT / self.manifest["generator"]["path"]
        self.assertEqual(
            hashlib.sha256(generator.read_bytes()).hexdigest(),
            self.manifest["generator"]["sha256"],
        )

    def test_statistical_contract_was_fixed_before_observation(self) -> None:
        self.assertTrue(
            self.manifest["selection"]["frozen_before_candidate_observation"])
        self.assertEqual(self.manifest["statistical_contract"], {
            "bootstrap_samples": 20000,
            "bootstrap_seed": 20260729,
            "confidence": 0.95,
            "margin_change_after_results_forbidden": True,
            "minimum_zero_regression_pairs": 149,
            "noninferiority_margin": 0.02,
            "sample_count": 149,
            "sample_selection_after_results_forbidden": True,
        })
        self.assertFalse(
            self.manifest["scoring"]["overall_promotion_authorized"])


if __name__ == "__main__":
    unittest.main()
