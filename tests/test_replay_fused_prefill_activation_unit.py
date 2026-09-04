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

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "torch is unavailable in the local unit-test environment",
    )
    def test_memory_bounded_reference_matches_materialized_gqa(self):
        import torch

        torch.manual_seed(20260904)
        for query_len in (3, 16):
            for block_table in (
                torch.tensor([0, 1], dtype=torch.int32),
                torch.tensor([1, 0], dtype=torch.int32),
            ):
                query = torch.randn(query_len, 8, 256, dtype=torch.float16)
                key_new = torch.randn(query_len, 2, 256, dtype=torch.float16)
                value_new = torch.randn(query_len, 2, 256, dtype=torch.float16)
                key_cache = torch.randn(2, 2, 32, 16, 8, dtype=torch.float16)
                value_cache = torch.randn(2, 2, 256, 16, dtype=torch.float16)
                actual, actual_lse = MODULE.reference_forward(
                    query, key_new, value_new, key_cache, value_cache,
                    block_table, 32, 256 ** -0.5)
                context_key = (key_cache[block_table.long()]
                               .permute(0, 3, 1, 2, 4).contiguous()
                               .view(32, 2, 256))
                context_value = (value_cache[block_table.long()]
                                 .permute(0, 3, 1, 2).contiguous()
                                 .view(32, 2, 256))
                expected = torch.empty_like(actual)
                expected_lse = torch.empty_like(actual_lse)
                for token in range(query_len):
                    for head in range(8):
                        kv_head = head // 4
                        keys = torch.cat((
                            context_key[:, kv_head],
                            key_new[:token + 1, kv_head]), dim=0).float()
                        values = torch.cat((
                            context_value[:, kv_head],
                            value_new[:token + 1, kv_head]), dim=0).float()
                        scores = torch.mv(
                            keys, query[token, head].float()) * 256 ** -0.5
                        weights = torch.softmax(scores, dim=0)
                        expected[token, head] = torch.mv(
                            values.transpose(0, 1), weights)
                        expected_lse[token, head] = torch.logsumexp(scores, 0)
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
                torch.testing.assert_close(
                    actual_lse, expected_lse, rtol=2e-5, atol=2e-5)

    def test_tp1_reference_temporary_bound_is_context_independent(self):
        bound = MODULE.reference_peak_temporary_bytes(8176, 16, 2)
        self.assertLess(bound, 650 * 1024 * 1024)
        self.assertGreater(bound, 500 * 1024 * 1024)
        self.assertGreater(bound, 0)
        with self.assertRaises(ValueError):
            MODULE.reference_peak_temporary_bytes(32, 7, 2)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "torch is unavailable in the local unit-test environment",
    )
    def test_four_rank_outputs_reassemble_in_global_query_head_order(self):
        import torch

        torch.manual_seed(176)
        query = torch.randn(3, 16, 256, dtype=torch.float16)
        key = torch.randn(3, 2, 256, dtype=torch.float16)
        value = torch.randn(3, 2, 256, dtype=torch.float16)
        key_cache = torch.randn(2, 2, 32, 16, 8, dtype=torch.float16)
        value_cache = torch.randn(2, 2, 256, 16, dtype=torch.float16)
        for block_table in (
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([1, 0], dtype=torch.int32),
        ):
            global_output, global_lse = MODULE.reference_forward(
                query, key, value, key_cache, value_cache,
                block_table, 32, 256 ** -0.5)
            rank_outputs = []
            rank_lses = []
            for rank in range(4):
                q_start = 4 * rank
                kv_head = q_start // 8
                rank_output, rank_lse = MODULE.reference_forward(
                    query[:, q_start:q_start + 4],
                    key[:, kv_head:kv_head + 1],
                    value[:, kv_head:kv_head + 1],
                    key_cache[:, kv_head:kv_head + 1],
                    value_cache[:, kv_head:kv_head + 1],
                    block_table, 32, 256 ** -0.5)
                rank_outputs.append(rank_output)
                rank_lses.append(rank_lse)
            torch.testing.assert_close(
                torch.cat(rank_outputs, dim=1), global_output,
                rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(
                torch.cat(rank_lses, dim=1), global_lse,
                rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
