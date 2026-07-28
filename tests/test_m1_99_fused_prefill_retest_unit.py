from __future__ import annotations

from contextlib import redirect_stderr
import hashlib
from io import StringIO
from pathlib import Path
import types
import unittest

from tests.test_paged_attn_unit import _load_paged_attn


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "qwen3_6_scripts" / "corex_fused_paged_prefill_split4.cu")
PREBUILT = (
    ROOT / "qwen3_6_scripts" / "prebuilt"
    / "corex-3.2.3-ivcore10" / "corex_fused_paged_prefill.so")
PATCH = ROOT / "qwen3_6_scripts" / "patch_xformers_sdpa_seq.py"
RUN_CONFIG = ROOT / "computility-run.yaml"


class FakeTensor:

    def __init__(
        self,
        shape,
        dtype,
        *,
        device="cuda:0",
        is_cuda=True,
        contiguous=True,
    ):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.is_cuda = is_cuda
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous

    def contiguous(self):
        return self

    def __getitem__(self, index):
        row, column = index
        if row != 0 or not isinstance(column, slice):
            raise AssertionError("unexpected fake tensor index")
        return FakeTensor(
            (column.stop - (column.start or 0),),
            self.dtype,
            device=self.device,
            is_cuda=self.is_cuda,
            contiguous=True,
        )


def fused_inputs(module):
    half = module.torch.float16
    int32 = module.torch.int32
    return {
        "query": FakeTensor((8176, 4, 256), half),
        "key": FakeTensor((8176, 1, 256), half),
        "value": FakeTensor((8176, 1, 256), half),
        "prefix_key": FakeTensor((0, 1, 256), half),
        "prefix_value": FakeTensor((0, 1, 256), half),
        "key_cache": FakeTensor((16871, 1, 32, 16, 8), half),
        "value_cache": FakeTensor((16871, 1, 256, 16), half),
        "block_tables": FakeTensor((1, 16384), int32),
        "seq_index": 0,
        "block_context_len": 65520,
        "num_q_heads": 4,
        "num_kv_heads": 1,
        "head_dim": 256,
        "gqa_ratio": 4,
        "block_size": 16,
    }


class M199FusedPrefillRetestUnitTest(unittest.TestCase):

    def test_frozen_source_and_binary_identity(self) -> None:
        self.assertEqual(
            hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "11c387e6012834fe634ffa8d038f7a4bf4ec19fa13ec23779ee1f414037e564b",
        )
        self.assertEqual(
            hashlib.sha256(PREBUILT.read_bytes()).hexdigest(),
            "ad4ea7707bb2f2bfe04e07a7ad5fd58a647232be70a3056937a0d738c8254bff",
        )

    def test_softmax_normalization_is_one_native_pass(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("normalize_split_scores_kernel", source)
        self.assertIn(
            "scores.data_ptr<float>(), corrections.data_ptr<float>()",
            source,
        )
        for removed in (
            "scan_split_max_kernel",
            "merge_split_sums_kernel",
            "at::max(active_scores",
            "active_scores.sub_",
            "at::sum(active_scores",
        ):
            self.assertNotIn(removed, source)

    def test_candidate_defaults_off_and_is_absent_from_yaml(self) -> None:
        module = _load_paged_attn()
        self.assertFalse(module._ENABLE_COREX_FUSED_PAGED_PREFILL)
        self.assertFalse(module._USE_COREX_FUSED_PAGED_PREFILL)
        self.assertNotIn(
            "BI100_ATTN_COREX_FUSED_PREFILL",
            RUN_CONFIG.read_text(encoding="utf-8"),
        )
        with self.assertRaises(RuntimeError):
            _load_paged_attn(fused_prefill="sometimes")

    def test_xformers_patch_propagates_decoder_semantics(self) -> None:
        source = PATCH.read_text(encoding="utf-8")
        self.assertIn(
            "is_causal_decoder=(attn_type == AttentionType.DECODER)",
            source,
        )

    def test_segment_guard_accepts_only_frozen_shape(self) -> None:
        module = _load_paged_attn()
        module._USE_COREX_FUSED_PAGED_PREFILL = True
        inputs = fused_inputs(module)
        guard = module._can_use_corex_fused_paged_prefill
        self.assertTrue(guard(**inputs))

        rejected = dict(inputs)
        rejected["prefix_key"] = FakeTensor(
            (1, 1, 256),
            module.torch.float16,
        )
        self.assertFalse(guard(**rejected))

        rejected = dict(inputs)
        rejected["query"] = FakeTensor(
            (16, 4, 256),
            module.torch.float16,
        )
        rejected["key"] = FakeTensor(
            (16, 1, 256),
            module.torch.float16,
        )
        rejected["value"] = FakeTensor(
            (16, 1, 256),
            module.torch.float16,
        )
        self.assertFalse(guard(**rejected))

        rejected = dict(inputs)
        rejected["block_context_len"] = 65521
        self.assertFalse(guard(**rejected))

        rejected = dict(inputs)
        rejected["num_q_heads"] = 6
        rejected["gqa_ratio"] = 6
        self.assertFalse(guard(**rejected))

    def test_request_and_metadata_guards_fail_closed(self) -> None:
        module = _load_paged_attn()
        module._USE_COREX_FUSED_PAGED_PREFILL = True
        request = {
            "kv_cache_dtype": "auto",
            "max_query_len": 8176,
            "total_query_len": 8176,
            "alibi_slopes": None,
            "sliding_window": None,
            "k_scale": 1.0,
            "v_scale": 1.0,
            "is_causal_decoder": True,
        }
        request_guard = (
            module._can_enable_corex_fused_paged_prefill_request)
        self.assertTrue(request_guard(**request))
        for name, value in (
            ("kv_cache_dtype", "fp8"),
            ("max_query_len", 8192),
            ("alibi_slopes", object()),
            ("sliding_window", 4096),
            ("k_scale", 0.5),
            ("v_scale", 0.5),
            ("is_causal_decoder", False),
        ):
            rejected = dict(request)
            rejected[name] = value
            with self.subTest(request=name):
                self.assertFalse(request_guard(**rejected))

        metadata = {
            "batch_size": 1,
            "block_table_rows": 1,
            "query_start_count": 2,
            "query_start_first": 0,
            "query_start_last": 8176,
            "seq_lens_count": 1,
            "seq_len": 73696,
            "context_lens_count": 1,
            "context_len": 65520,
            "total_query_len": 8176,
        }
        metadata_guard = (
            module._is_single_sequence_fused_prefill_metadata)
        self.assertTrue(metadata_guard(**metadata))
        for name, value in (
            ("batch_size", 2),
            ("block_table_rows", 2),
            ("query_start_last", 8175),
            ("seq_len", 73695),
            ("context_len", -1),
        ):
            rejected = dict(metadata)
            rejected[name] = value
            with self.subTest(metadata=name):
                self.assertFalse(metadata_guard(**rejected))

    def test_native_dispatch_uses_active_blocks_and_fails_loudly(
        self,
    ) -> None:
        module = _load_paged_attn()
        inputs = fused_inputs(module)
        calls = []

        def forward(*args):
            calls.append(args)
            return [inputs["query"], object()]

        module._USE_COREX_FUSED_PAGED_PREFILL = True
        module._corex_fused_paged_prefill = types.SimpleNamespace(
            forward=forward,
        )
        stderr = StringIO()
        with redirect_stderr(stderr):
            output = module.PagedAttention._forward_prefix_segment_pytorch(
                **inputs,
                tile_sz=512,
                scale=0.0625,
                orig_dtype=module.torch.float16,
                fused_request_eligible=True,
            )
        self.assertIs(output, inputs["query"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][5].shape, (4095,))
        self.assertEqual(calls[0][6:], (65520, 0.0625))
        self.assertIn("path=corex_split4", stderr.getvalue())

        def fail(*args):
            raise RuntimeError("native split4 failure")

        module._corex_fused_paged_prefill = types.SimpleNamespace(
            forward=fail,
        )
        with self.assertRaisesRegex(RuntimeError, "native split4 failure"):
            module.PagedAttention._forward_prefix_segment_pytorch(
                **inputs,
                tile_sz=512,
                scale=0.0625,
                orig_dtype=module.torch.float16,
                fused_request_eligible=True,
            )


if __name__ == "__main__":
    unittest.main()
