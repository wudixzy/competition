from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests" / "capture_fused_prefill_activation_service.py"
SPEC = importlib.util.spec_from_file_location("capture_activation_service", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(SCRIPT.parent))
try:
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path.remove(str(SCRIPT.parent))


class CaptureFusedPrefillActivationServiceTest(unittest.TestCase):

    @staticmethod
    def _manifest(records):
        return {
            "schema": "bi100-fused-prefill-activation-bank-v2",
            "version": 2,
            "run_id": "capture",
            "rank": 0,
            "source_revision": "a" * 40,
            "runtime_identity": "overlay",
            "source_artifact_sha256": "b" * 64,
            "model_identity": {
                "name": "Qwen3.6-35B-A3B",
                "config_sha256": "c" * 64,
            },
            "tokenizer_identity": {"sha256": "d" * 64},
            "instance": "unit",
            "captured_at_utc": "2026-09-04T00:00:00+00:00",
            "producer": "baseline-pytorch-fallback",
            "synthetic_prompt_attestation": "synthetic-exact-prompt-v1",
            "selection": MODULE.CAPTURE_SELECTION,
            "capture_topology": MODULE.CAPTURE_TOPOLOGY,
            "record_count": len(records),
            "records": records,
            "privacy": MODULE.CAPTURE_PRIVACY,
        }

    @staticmethod
    def _record():
        return {
            "bucket_min_context_tokens": 24576,
            "call_ordinal": 0,
            "layer_index": 3,
            "context_tokens": 24576,
            "query_length": 32,
            "file": "case.pt",
            "sha256": "e" * 64,
            "size_bytes": 42,
            "compact_physical_blocks": 1,
            "logical_blocks": 1536,
            "block_table": {
                "shape": [1536],
                "sha256": "f" * 64,
                "logical_order": (
                    "preserved_after_first_occurrence_compaction"),
            },
            "head_mapping": MODULE.CAPTURE_HEAD_MAPPING,
            "tensors": {
                "query": {
                    "shape": [32, 16, 256], "dtype": "torch.float16"},
                "key": {
                    "shape": [32, 2, 256], "dtype": "torch.float16"},
                "value": {
                    "shape": [32, 2, 256], "dtype": "torch.float16"},
                "key_cache": {
                    "shape": [1, 2, 32, 16, 8],
                    "dtype": "torch.float16"},
                "value_cache": {
                    "shape": [1, 2, 256, 16],
                    "dtype": "torch.float16"},
                "block_table": {
                    "shape": [1536], "dtype": "torch.int32"},
            },
        }

    def test_manifest_cells_fail_closed(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            path = Path(temporary) / "rank-0.manifest.json"
            with self.assertRaisesRegex(RuntimeError, "missing or corrupt"):
                MODULE._captured_cells(path)
            path.write_text("not-json\n", encoding="ascii")
            with self.assertRaisesRegex(RuntimeError, "missing or corrupt"):
                MODULE._captured_cells(path)
            path.write_text(json.dumps({"records": []}), encoding="ascii")
            with self.assertRaisesRegex(RuntimeError, "contract differs"):
                MODULE._captured_cells(path)

    def test_manifest_cells_require_identity_and_unique_cells(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            path = Path(temporary) / "rank-0.manifest.json"
            record = self._record()
            path.write_text(
                json.dumps(self._manifest([record])), encoding="ascii")
            self.assertEqual(
                MODULE._captured_cells(path, "capture"), {(24576, 0)})
            with self.assertRaisesRegex(RuntimeError, "contract differs"):
                MODULE._captured_cells(path, "different-run")
            record["sha256"] = "not-a-digest"
            path.write_text(
                json.dumps(self._manifest([record])), encoding="ascii")
            with self.assertRaisesRegex(RuntimeError, "identity differs"):
                MODULE._captured_cells(path, "capture")
            record = self._record()
            record["tensors"]["query"]["shape"][1] = 15
            path.write_text(
                json.dumps(self._manifest([record])), encoding="ascii")
            with self.assertRaisesRegex(RuntimeError, "identity differs"):
                MODULE._captured_cells(path, "capture")

    def test_frozen_targets_map_to_distinct_buckets(self):
        self.assertEqual(MODULE.TARGET_BUCKETS, {
            32768: 24576,
            65536: 57344,
            131072: 122880,
        })


if __name__ == "__main__":
    unittest.main()
