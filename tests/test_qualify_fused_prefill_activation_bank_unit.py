from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests" / "qualify_fused_prefill_activation_bank.py"
SPEC = importlib.util.spec_from_file_location("qualify_bank", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
CONTRACT = json.loads(
    (ROOT / "quality" / "experiment_funnel.v1.json").read_text())


class QualifyActivationBankTest(unittest.TestCase):

    def build_manifest(self, root: Path, rank: int, buckets, ordinals):
        records = []
        for bucket in buckets:
            for ordinal in ordinals:
                filename = f"rank-{rank}-{bucket}-{ordinal}.pt"
                case = root / filename
                case.write_bytes(f"{rank}:{bucket}:{ordinal}".encode())
                records.append({
                    "file": filename,
                    "size_bytes": case.stat().st_size,
                    "sha256": MODULE._sha256(case),
                    "bucket_min_context_tokens": bucket,
                    "call_ordinal": ordinal,
                })
        value = {
            "schema": MODULE.MANIFEST_SCHEMA,
            "version": 1,
            "run_id": "m1-140",
            "rank": rank,
            "source_revision": "a" * 40,
            "runtime_identity": "overlay",
            "producer": "baseline-pytorch-fallback",
            "synthetic_prompt_attestation": "synthetic-exact-prompt-v1",
            "record_count": len(records),
            "records": records,
            "privacy": {
                "raw_activation_files_may_be_committed": False,
            },
        }
        path = root / f"rank-{rank}.manifest.json"
        path.write_text(json.dumps(value), encoding="ascii")
        return path, value

    def test_full_matrix_qualifies_without_authorizing_short_tp4(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = [
                self.build_manifest(
                    root, rank, (24576, 57344, 122880), (0, 4, 9))
                for rank in range(4)
            ]
            result = MODULE.qualify(
                manifests,
                CONTRACT,
                profile="qualification",
                run_id="m1-140",
                source_revision="a" * 40,
                runtime_identity="overlay",
            )
        self.assertTrue(result["qualified"], result)
        self.assertEqual(result["case_count"], 36)
        self.assertFalse(result["authorization"]["short_tp4_authorized"])


if __name__ == "__main__":
    unittest.main()
