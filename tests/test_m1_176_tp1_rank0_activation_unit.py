from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "derive_m1_176_tp1_rank0_activation_bank.py"
SPEC = importlib.util.spec_from_file_location("m1_176_derive", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class M1176Tp1Rank0ActivationTest(unittest.TestCase):

    def _source_bank(self, root: Path) -> Path:
        import torch

        def half_values(size: int):
            return torch.arange(size, dtype=torch.int32).remainder(257).to(
                torch.float16)

        root.mkdir(mode=0o700)
        records = []
        for bucket in MODULE.FROZEN_SELECTION["context_buckets"]:
            context = bucket
            query_len = 32
            tensors = {
                "query": half_values(
                    query_len * 16 * 256
                ).reshape(query_len, 16, 256).contiguous(),
                "key": half_values(
                    query_len * 2 * 256
                ).reshape(query_len, 2, 256).contiguous(),
                "value": half_values(
                    query_len * 2 * 256
                ).reshape(query_len, 2, 256).add(3).contiguous(),
                "key_cache": half_values(
                    1 * 2 * 32 * 16 * 8
                ).reshape(1, 2, 32, 16, 8).contiguous(),
                "value_cache": half_values(
                    1 * 2 * 256 * 16
                ).reshape(1, 2, 256, 16).contiguous(),
                "block_table": torch.zeros(
                    context // 16, dtype=torch.int32),
            }
            filename = f"rank-0.bucket-{bucket}.ordinal-0.pt"
            path = root / filename
            torch.save({
                "schema": MODULE.SOURCE_CASE_SCHEMA,
                "version": 1,
                "context_tokens": context,
                "scale": 256 ** -0.5,
                "rank": 0,
                "bucket": bucket,
                "call_ordinal": 0,
                "tensors": tensors,
            }, path)
            path.chmod(0o600)
            records.append({
                "bucket_min_context_tokens": bucket,
                "call_ordinal": 0,
                "context_tokens": context,
                "query_length": query_len,
                "file": filename,
                "sha256": MODULE.sha256_file(path),
                "size_bytes": path.stat().st_size,
                "compact_physical_blocks": 1,
                "logical_blocks": context // 16,
                "tensors": {
                    name: MODULE._tensor_metadata(tensors[name])
                    for name in sorted(tensors)
                },
            })
        manifest = {
            "schema": MODULE.SOURCE_BANK_SCHEMA,
            "version": 1,
            "run_id": "capture",
            "rank": 0,
            "source_revision": "a" * 40,
            "runtime_identity": "overlay",
            "producer": "baseline-pytorch-fallback",
            "synthetic_prompt_attestation": "synthetic-exact-prompt-v1",
            "selection": MODULE.FROZEN_SELECTION,
            "record_count": len(records),
            "records": records,
            "privacy": {
                "raw_activation_files_private": True,
                "raw_activation_files_may_be_committed": False,
                "contains_prompts": False,
                "contains_model_outputs": False,
                "contains_token_ids": False,
                "contains_credentials": False,
            },
        }
        path = root / "rank-0.manifest.json"
        path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="ascii")
        path.chmod(0o600)
        return path

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "torch is unavailable in the local unit-test environment",
    )
    def test_derives_exact_contiguous_tp4_rank0_projection_slice(self):
        import torch

        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            root.chmod(0o700)
            source = self._source_bank(root / "source")
            output = root / "derived"
            result = MODULE.derive(
                source,
                output,
                expected_source_revision="a" * 40,
                expected_runtime_identity="overlay",
            )
            self.assertTrue(result["qualified"])
            self.assertEqual(result["record_count"], 3)
            manifest = json.loads(
                Path(result["manifest"]).read_text(encoding="ascii"))
            self.assertEqual(manifest["derivation"], MODULE.DERIVATION)
            self.assertFalse(
                manifest["authorization"]["tp4_activation_capture_claim"])

            derived_case = torch.load(
                output / manifest["records"][0]["file"],
                map_location="cpu")
            original_record = json.loads(source.read_text())["records"][0]
            original_case = torch.load(
                source.parent / original_record["file"], map_location="cpu")
            self.assertTrue(torch.equal(
                derived_case["tensors"]["query"],
                original_case["tensors"]["query"][:, :4]))
            self.assertTrue(torch.equal(
                derived_case["tensors"]["key_cache"],
                original_case["tensors"]["key_cache"][:, :1]))
            self.assertEqual(
                derived_case["source_case_sha256"],
                original_record["sha256"])

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "torch is unavailable in the local unit-test environment",
    )
    def test_source_rank_and_case_hash_fail_closed(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            root.chmod(0o700)
            source = self._source_bank(root / "source")
            value = json.loads(source.read_text(encoding="ascii"))
            value["rank"] = 1
            source.write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n",
                encoding="ascii")
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                MODULE.derive(
                    source,
                    root / "derived-rank",
                    expected_source_revision="a" * 40,
                    expected_runtime_identity="overlay",
                )

            value["rank"] = 0
            value["records"][0]["sha256"] = hashlib.sha256(b"wrong").hexdigest()
            source.write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n",
                encoding="ascii")
            with self.assertRaisesRegex(ValueError, "case identity differs"):
                MODULE.derive(
                    source,
                    root / "derived-hash",
                    expected_source_revision="a" * 40,
                    expected_runtime_identity="overlay",
                )

    def test_capture_mode_is_opt_in_and_tp1_only(self):
        runner = (ROOT / "scripts" / "run_qwen36_diagnostic_gate.sh").read_text(
            encoding="ascii")
        replay = (
            ROOT / "tests" / "replay_m1_176_tp1_rank0_activation.py"
        ).read_text(encoding="ascii")
        self.assertIn(
            "BI100_DIAGNOSTIC_ACTIVATION_CAPTURE:-0", runner)
        self.assertIn(
            "diagnostic activation capture is restricted to TP1", runner)
        self.assertIn(
            "BI100_ATTN_COREX_FUSED_PREFILL=0", runner)
        self.assertIn(
            '"tp4_activation_capture_claim": False', replay)
        self.assertIn(
            '"visible_physical_gpu": args.visible_physical_gpu', replay)
        self.assertIn('"logical_tp_rank": 0', replay)


if __name__ == "__main__":
    unittest.main()
