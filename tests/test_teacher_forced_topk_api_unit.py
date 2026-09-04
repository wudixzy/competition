from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import teacher_forced_topk_api as api


class TeacherForcedTopkApiTests(unittest.TestCase):

    def test_short_l3_target_parser_is_fixed_and_bounded(self) -> None:
        self.assertEqual(
            api.parse_targets("4096,16384,32768,65536"),
            (4096, 16384, 32768, 65536),
        )
        for raw in ("", "4096,4096", "65536,4096", "x", "262144"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                api.parse_targets(raw)

    def test_lightweight_runtime_manifest_v2(self) -> None:
        expected = {
            "source_revision": "a" * 40,
            "runtime_identity": "overlay:install-byte-equal",
            "instance": "private-instance",
            "model_path": "/model",
            "tokenizer_path": "/model",
            "gpu_count": 4,
            "tensor_parallel_size": 4,
            "max_model_len": 262144,
            "served_model_name": "llm",
        }
        value = {
            "schema": "bi100-quality-runtime-manifest-v2",
            "version": 2,
            **expected,
            "command": ["launch_service"],
            "environment": {
                "BI100_ATTN_COREX_FUSED_PREFILL": "0",
                "BI100_CACHE_TRACE": "1",
                "BI100_GDN_CACHE_POLICY": "admission64",
                "BI100_GDN_RESTORE_MODE": "hybrid64",
                "BI100_KV_EVICTION_POLICY": "lru",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            self.assertEqual(
                api.load_runtime_manifest_v2(path, expected), value)
            incomplete = json.loads(json.dumps(value))
            incomplete["environment"].pop("BI100_KV_EVICTION_POLICY")
            path.write_text(json.dumps(incomplete), encoding="utf-8")
            with self.assertRaises(ValueError):
                api.load_runtime_manifest_v2(path, expected)
            value["environment"]["API_TOKEN"] = "redacted"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ValueError):
                api.load_runtime_manifest_v2(path, expected)

    def test_attention_runtime_manifest_is_focused_and_bound(self) -> None:
        expected = {
            "source_revision": "a" * 40,
            "runtime_identity": "runtime-1",
            "instance": "instance-1",
            "model_path": "/model",
            "tokenizer_path": "/model",
            "gpu_count": 4,
            "tensor_parallel_size": 4,
            "max_model_len": 262144,
            "served_model_name": "llm",
        }
        value = {
            "schema": "bi100-attention-operator-runtime-v1",
            "version": 1,
            "change_scope": "attention_operator",
            "workload_mode": "teacher_forced",
            "dtype": "float16",
            "block_size": 16,
            **expected,
            "command": ["launch_service"],
            "environment": {
                "BI100_ATTN_COREX_FUSED_PREFILL": "1",
                "BI100_CACHE_TRACE": "0",
                "BI100_GDN_CACHE_POLICY": "admission64",
                "BI100_GDN_RESTORE_MODE": "hybrid64",
                "BI100_KV_EVICTION_POLICY": "lru",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            self.assertEqual(
                api.load_attention_runtime_manifest(path, expected), value)
            value["workload_mode"] = "performance"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ValueError):
                api.load_attention_runtime_manifest(path, expected)

    def test_fixed_sampler_is_stable_and_in_range(self) -> None:
        for token_count in api.TARGETS:
            first = api.sample_positions(token_count, 64)
            second = api.sample_positions(token_count, 64)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 64)
            self.assertEqual(len(set(first)), 64)
            self.assertTrue(all(
                1 <= position < token_count for position in first
            ))

    def test_sampler_covers_chunk_boundaries_for_long_prompts(self) -> None:
        positions = set(api.sample_positions(235000, 64))
        self.assertTrue(any(
            abs(position - 8192) <= 1 for position in positions
        ))
        self.assertTrue(any(
            abs(position - 229376) <= 1 for position in positions
        ))

    def test_m1_178_sampler_covers_every_available_chunk_boundary(self) -> None:
        for target in (4096, 16384, 32768, 65536):
            positions = set(api.sample_positions(target, 64))
            for boundary in range(8192, target, 8192):
                self.assertTrue(any(
                    abs(position - boundary) <= 1 for position in positions
                ), (target, boundary))

    def test_position_summary_keeps_topk_plus_teacher_only(self) -> None:
        raw = {
            str(token_id): {
                "logprob": -float(rank),
                "rank": rank,
                "decoded_token": "private",
            }
            for rank, token_id in enumerate(range(100, 107), start=1)
        }
        raw["999"] = {
            "logprob": -20.0,
            "rank": 100,
            "decoded_token": "private",
        }
        result = api.summarize_position(
            raw,
            position=17,
            actual_token_id=999,
            identity_key=b"a" * 32,
            top_k=5,
        )
        self.assertEqual(result["position"], 17)
        self.assertEqual(len(result["top_logprobs"]), 6)
        self.assertIn(
            result["actual_token_key"],
            {row["token_key"] for row in result["top_logprobs"]},
        )
        self.assertNotIn("decoded_token", result)

    def test_nonfinite_logprob_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not finite"):
            api.summarize_position(
                {
                    "1": {"logprob": 0.0},
                    "2": {"logprob": float("nan")},
                },
                position=1,
                actual_token_id=1,
                identity_key=b"a" * 32,
                top_k=5,
            )

    def test_server_tokenize_response_is_strictly_validated(self) -> None:
        result = api._response_token_ids(
            {
                "count": 3,
                "max_model_len": 262144,
                "tokens": [1, 2, 3],
            },
            3,
        )
        self.assertEqual(result, [1, 2, 3])
        with self.assertRaisesRegex(ValueError, "sequence is invalid"):
            api._response_token_ids(
                {
                    "count": 3,
                    "max_model_len": 262144,
                    "tokens": [1, True, 3],
                },
                3,
            )


if __name__ == "__main__":
    unittest.main()
