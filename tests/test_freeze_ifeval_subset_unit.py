from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/freeze_ifeval_subset.py"
SPEC = importlib.util.spec_from_file_location("freeze_ifeval_subset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
EXTERNAL = ROOT / "quality/external/google_ifeval"


class FreezeIFEvalSubsetTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((
            EXTERNAL / "manifest.v1.json").read_text(encoding="utf-8"))
        cls.rows = [json.loads(line) for line in (
            EXTERNAL / "subset.v1.jsonl").read_text(
                encoding="utf-8").splitlines()]

    def test_source_and_selection_reproduce_frozen_subset(self):
        source = MODULE.load_source(
            EXTERNAL / "source/ifeval_input_data.jsonl")
        selected = MODULE.select_rows(source)
        self.assertEqual(selected, self.rows)
        self.assertEqual(
            [row["key"] for row in selected],
            self.manifest["selection"]["selected_keys_in_request_order"],
        )
        self.assertEqual(len(selected), 64)

    def test_every_instruction_id_has_four_or_more_examples(self):
        counts = collections.Counter(
            item for row in self.rows for item in row["instruction_id_list"])
        self.assertEqual(set(counts), MODULE.EXPECTED_INSTRUCTION_IDS)
        self.assertGreaterEqual(min(counts.values()), 4)
        self.assertEqual(
            dict(sorted(counts.items())),
            self.manifest["selection"]["instruction_counts"],
        )

    def test_manifest_and_subset_identities_are_frozen(self):
        self.assertEqual(
            hashlib.sha256((EXTERNAL / "manifest.v1.json").read_bytes())
            .hexdigest(),
            "07ec4efb5fe7afaacb55723c1d53be4c2f58c840bbd6a54bf944e15cfbca1855",
        )
        self.assertEqual(
            hashlib.sha256((EXTERNAL / "subset.v1.jsonl").read_bytes())
            .hexdigest(),
            "bdb2e4ec0b0fd19b89c55ebb9ed49e17361706c923ddedeeab429f669e4bdb78",
        )
        self.assertFalse(
            self.manifest["evaluator"]
            ["dataset_difference_from_evaluator_repo"]["selected"])

    def test_evaluator_and_offline_distributions_match_manifest(self):
        evaluator = self.manifest["evaluator"]
        self.assertEqual(
            evaluator["revision"],
            "e6890f85757dd84e27ca6df2dd30651dafad28e0",
        )
        for group, prefix in (
                (evaluator["vendored_files"], EXTERNAL),
                (self.manifest["offline_environment"]
                 ["distribution_artifacts"],
                 EXTERNAL / "wheelhouse")):
            for item in group:
                path = prefix / item["path"]
                self.assertEqual(path.stat().st_size, item["bytes"])
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    item["sha256"],
                )


if __name__ == "__main__":
    unittest.main()
