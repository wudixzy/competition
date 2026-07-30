from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests" / "replay_fused_prefill_activation.py"
SPEC = importlib.util.spec_from_file_location("replay_activation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _tensor_metadata() -> dict:
    return {
        "query": {"shape": [8176, 4, 256], "dtype": "torch.float16"},
        "key": {"shape": [8176, 1, 256], "dtype": "torch.float16"},
        "value": {"shape": [8176, 1, 256], "dtype": "torch.float16"},
        "key_cache": {
            "shape": [1536, 1, 32, 16, 8],
            "dtype": "torch.float16",
        },
        "value_cache": {
            "shape": [1536, 1, 256, 16],
            "dtype": "torch.float16",
        },
        "block_table": {"shape": [1536], "dtype": "torch.int32"},
    }


class ReplayActivationContractTest(unittest.TestCase):

    def _bank(self, root: Path) -> tuple[Path, str, str]:
        root.chmod(0o700)
        case = root / "rank-0.bucket-24576.ordinal-0.pt"
        case.write_bytes(b"private activation placeholder")
        case.chmod(0o600)
        case_sha = hashlib.sha256(case.read_bytes()).hexdigest()
        source = "a" * 40
        runtime = "runtime-tree-sha256"
        manifest = {
            "schema": MODULE.BANK_SCHEMA,
            "version": 1,
            "run_id": "capture-run",
            "rank": 0,
            "source_revision": source,
            "runtime_identity": runtime,
            "producer": "baseline-pytorch-fallback",
            "synthetic_prompt_attestation": "synthetic-exact-prompt-v1",
            "selection": dict(MODULE.CAPTURE_SELECTION),
            "record_count": 1,
            "records": [{
                "bucket_min_context_tokens": 24576,
                "call_ordinal": 0,
                "context_tokens": 24576,
                "query_length": 8176,
                "file": case.name,
                "sha256": case_sha,
                "size_bytes": case.stat().st_size,
                "compact_physical_blocks": 1536,
                "logical_blocks": 1536,
                "tensors": _tensor_metadata(),
            }],
            "privacy": dict(MODULE.CAPTURE_PRIVACY),
        }
        path = root / "rank-0.manifest.json"
        path.write_text(
            json.dumps(manifest, sort_keys=True) + "\n",
            encoding="ascii",
        )
        path.chmod(0o600)
        return path, source, runtime

    def test_private_manifest_contract_is_accepted(self):
        with tempfile.TemporaryDirectory(
                prefix="m1-140-bank-", dir="/tmp") as temporary:
            path, source, runtime = self._bank(Path(temporary))
            manifest, cases = MODULE.validate_bank(
                path,
                expected_capture_source_revision=source,
                expected_runtime_identity=runtime,
            )
            self.assertEqual(manifest["rank"], 0)
            self.assertEqual(len(cases), 1)

    def test_group_readable_case_is_rejected(self):
        with tempfile.TemporaryDirectory(
                prefix="m1-140-bank-", dir="/tmp") as temporary:
            root = Path(temporary)
            path, source, runtime = self._bank(root)
            (root / "rank-0.bucket-24576.ordinal-0.pt").chmod(0o640)
            with self.assertRaisesRegex(
                    ValueError, "activation bank case identity differs"):
                MODULE.validate_bank(
                    path,
                    expected_capture_source_revision=source,
                    expected_runtime_identity=runtime,
                )

    def test_ordered_frozen_subset_supports_smoke_replay(self):
        with tempfile.TemporaryDirectory(
                prefix="m1-140-bank-", dir="/tmp") as temporary:
            path, source, runtime = self._bank(Path(temporary))
            manifest = json.loads(path.read_text(encoding="ascii"))
            manifest["selection"] = {
                "context_buckets": [24576],
                "full_attention_call_ordinals": [0],
            }
            path.write_text(
                json.dumps(manifest, sort_keys=True) + "\n",
                encoding="ascii",
            )
            path.chmod(0o600)
            _, cases = MODULE.validate_bank(
                path,
                expected_capture_source_revision=source,
                expected_runtime_identity=runtime,
            )
            self.assertEqual(len(cases), 1)

    def test_atomic_json_rejects_nonstandard_numbers(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            with self.assertRaises(ValueError):
                MODULE._atomic_json(path, {"metric": math.nan})
            self.assertFalse(path.exists())
            self.assertEqual(
                [name for name in os.listdir(temporary)],
                [],
            )


if __name__ == "__main__":
    unittest.main()
