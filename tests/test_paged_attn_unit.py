import importlib
import importlib.util
import inspect
import os
import pathlib
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr
from io import StringIO

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGED_ATTN = ROOT / "qwen3_6_scripts" / "paged_attn.py"


class _EnvPatch:

    def __init__(self, **updates):
        self.updates = updates
        self.previous = {}

    def __enter__(self):
        for key, value in self.updates.items():
            self.previous[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def __exit__(self, exc_type, exc, tb):
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _install_stubs():
    torch_mod = types.ModuleType("torch")

    class _Tensor:
        pass

    torch_mod.Tensor = _Tensor
    vllm_mod = types.ModuleType("vllm")
    vllm_mod._custom_ops = types.SimpleNamespace()
    env_mod = types.ModuleType("vllm.bi100_env")

    def env_bool(name, default=False):
        raw = os.environ.get(name)
        if raw is None:
            return default
        if raw in ("1", "true", "True", "yes", "YES", "on", "ON"):
            return True
        if raw in ("0", "false", "False", "no", "NO", "off", "OFF"):
            return False
        raise RuntimeError(f"{name} must be boolean, got {raw!r}")

    def env_int(name, default, min_value, max_value):
        raw = os.environ.get(name)
        if raw is None:
            return default
        value = int(raw)
        if not (min_value <= value <= max_value):
            raise RuntimeError(
                f"{name}={value} outside [{min_value}, {max_value}]")
        return value

    env_mod.env_bool = env_bool
    env_mod.env_int = env_int
    profile_mod = types.ModuleType("vllm.bi100_profile")

    class _NoopTimer:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    profile_mod.bi100_timer = lambda name: _NoopTimer()
    sys.modules["torch"] = torch_mod
    sys.modules["vllm"] = vllm_mod
    sys.modules["vllm.bi100_env"] = env_mod
    sys.modules["vllm.bi100_profile"] = profile_mod
    return torch_mod, vllm_mod


def _clear_stubs():
    for name in ["torch", "vllm", "vllm.bi100_env", "vllm.bi100_profile"]:
        sys.modules.pop(name, None)


def _load_paged_attn(**env):
    with _EnvPatch(
            BI100_PYTORCH_DECODE_THRESHOLD=env.get("threshold"),
            BI100_PREFIX_BLOCKS_PER_TILE=env.get("tile"),
            BI100_FORCE_PAGED_ATTN_V2=env.get("force_v2"),
            BI100_PAGED_ATTN_DIAGNOSTICS=env.get("diagnostics"),
    ):
        old_modules = {
            name: sys.modules.get(name)
            for name in ["torch", "vllm", "vllm.bi100_env", "vllm.bi100_profile"]
        }
        _clear_stubs()
        _install_stubs()
        try:
            module_name = f"paged_attn_unit_{id(env)}"
            spec = importlib.util.spec_from_file_location(module_name, PAGED_ATTN)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module
        finally:
            _clear_stubs()
            for name, module in old_modules.items():
                if module is not None:
                    sys.modules[name] = module
            importlib.invalidate_caches()


class PagedAttentionUnitTest(unittest.TestCase):

    def test_legacy_decode_interface_uses_head_mapping_tensor(self):
        module = _load_paged_attn()
        self.assertEqual(
            module.PagedAttention.get_kv_cache_shape(10, 16, 2, 256),
            (2, 10, 8192),
        )
        parameters = inspect.signature(
            module.PagedAttention.forward_decode).parameters
        self.assertIn("head_mapping", parameters)
        self.assertNotIn("num_kv_heads", parameters)

    def test_strict_prefix_segments_match_cache_boundaries(self):
        module = _load_paged_attn()
        segment = module._strict_prefix_query_segments
        self.assertEqual(segment(0, 8192, 16), [
            (0, 8176, 0),
            (8176, 8192, 8176),
        ])
        self.assertEqual(segment(8192, 520, 16), [
            (0, 512, 8192),
            (512, 520, 8704),
        ])
        self.assertEqual(segment(8176, 16, 16), [
            (0, 16, 8176),
        ])

    def test_strict_prefix_segments_handle_empty_and_short_queries(self):
        module = _load_paged_attn()
        segment = module._strict_prefix_query_segments
        self.assertEqual(segment(0, 0, 16), [])
        self.assertEqual(segment(17, 1, 16), [(0, 1, 17)])
        self.assertEqual(segment(31, 2, 16), [(0, 1, 31), (1, 2, 32)])

    def test_context_tiles_join_block_cache_and_preceding_query(self):
        module = _load_paged_attn()
        spans = module._prefix_context_tile_spans
        cold = spans(11296, 320, 512)
        warm = spans(11616, 0, 512)
        self.assertEqual(cold[-1], (11264, 11296, 0, 320))
        self.assertEqual(warm[-1], (11264, 11616, 0, 0))
        self.assertEqual(
            sum((b1 - b0) + (p1 - p0) for b0, b1, p0, p1 in cold),
            11616)

    def test_context_tile_spans_validate_inputs(self):
        module = _load_paged_attn()
        spans = module._prefix_context_tile_spans
        self.assertEqual(spans(0, 0, 512), [])
        with self.assertRaises(ValueError):
            spans(-1, 0, 512)
        with self.assertRaises(ValueError):
            spans(0, 1, 0)

    def test_attention_env_defaults_are_stable(self):
        module = _load_paged_attn()
        self.assertEqual(module.PagedAttention._PYTORCH_DECODE_THRESHOLD, 32768)
        self.assertEqual(module._PREFIX_BLOCKS_PER_TILE, 32)
        self.assertFalse(module.PagedAttention._FORCE_PAGED_ATTN_V2)
        self.assertFalse(module._PAGED_ATTN_DIAGNOSTICS)
        self.assertEqual(module._DECODE_LOG_INTERVAL, 0)
        self.assertTrue(module.PagedAttention._should_use_paged_attention_v1(
            max_seq_len=100000,
            max_num_partitions=196,
            num_seqs=1,
            num_heads=64,
        ))

    def test_attention_env_overrides_are_loaded_at_import(self):
        module = _load_paged_attn(
            threshold="4096",
            tile="64",
            force_v2="1",
            diagnostics="1",
        )
        self.assertEqual(module.PagedAttention._PYTORCH_DECODE_THRESHOLD, 4096)
        self.assertEqual(module._PREFIX_BLOCKS_PER_TILE, 64)
        self.assertTrue(module.PagedAttention._FORCE_PAGED_ATTN_V2)
        self.assertTrue(module._PAGED_ATTN_DIAGNOSTICS)
        self.assertEqual(module._DECODE_LOG_INTERVAL, 8192)
        self.assertFalse(module.PagedAttention._should_use_paged_attention_v1(
            max_seq_len=100000,
            max_num_partitions=196,
            num_seqs=1,
            num_heads=64,
        ))

    def test_attention_env_rejects_invalid_values(self):
        with self.assertRaises(RuntimeError):
            _load_paged_attn(threshold="0")

    def test_decode_layout_accepts_exact_block_table_boundary(self):
        module = _load_paged_attn()
        required = module._validate_decode_layout(
            num_seqs=1,
            seq_lens_count=1,
            block_table_rows=1,
            block_table_width=2048,
            actual_max=32768,
            block_size=16,
            physical_key_blocks=16871,
            physical_value_blocks=16871,
            num_heads=4,
            num_kv_heads=1,
        )
        self.assertEqual(required, 2048)

    def test_decode_layout_accepts_256k_capacity_boundaries(self):
        module = _load_paged_attn()
        common = dict(
            num_seqs=1,
            seq_lens_count=1,
            block_table_rows=1,
            physical_key_blocks=16871,
            physical_value_blocks=16871,
            num_heads=4,
            num_kv_heads=1,
            block_size=16,
        )
        self.assertEqual(module._validate_decode_layout(
            block_table_width=16000,
            actual_max=256000,
            **common,
        ), 16000)
        self.assertEqual(module._validate_decode_layout(
            block_table_width=16384,
            actual_max=262144,
            **common,
        ), 16384)

    def test_decode_layout_rejects_undersized_block_table(self):
        module = _load_paged_attn()
        with self.assertRaisesRegex(RuntimeError, "needs 2049 blocks"):
            module._validate_decode_layout(
                num_seqs=1,
                seq_lens_count=1,
                block_table_rows=1,
                block_table_width=2048,
                actual_max=32769,
                block_size=16,
                physical_key_blocks=16871,
                physical_value_blocks=16871,
                num_heads=4,
                num_kv_heads=1,
            )

    def test_decode_layout_rejects_inconsistent_cache_and_gqa(self):
        module = _load_paged_attn()
        kwargs = dict(
            num_seqs=1,
            seq_lens_count=1,
            block_table_rows=1,
            block_table_width=2,
            actual_max=17,
            block_size=16,
            physical_key_blocks=10,
            physical_value_blocks=9,
            num_heads=4,
            num_kv_heads=1,
        )
        with self.assertRaisesRegex(RuntimeError, "cache block counts differ"):
            module._validate_decode_layout(**kwargs)
        kwargs["physical_value_blocks"] = 10
        kwargs["num_kv_heads"] = 3
        with self.assertRaisesRegex(RuntimeError, "invalid GQA layout"):
            module._validate_decode_layout(**kwargs)

    def test_prefix_block_table_guard_raises_by_default(self):
        module = _load_paged_attn()
        with self.assertRaises(RuntimeError) as ctx:
            module.PagedAttention._validate_prefix_block_table(
                seq_index=0,
                num_ctx_blocks=3,
                block_table_width=2,
                ctx_len=33,
            )
        self.assertIn("refusing to truncate context", str(ctx.exception))

    def test_prefix_block_table_guard_debug_cap_is_explicit(self):
        module = _load_paged_attn()
        with _EnvPatch(BI100_ALLOW_PREFIX_GUARD_CAP="1"):
            stderr = StringIO()
            with redirect_stderr(stderr):
                capped = module.PagedAttention._validate_prefix_block_table(
                    seq_index=0,
                    num_ctx_blocks=3,
                    block_table_width=2,
                    ctx_len=33,
                )
        self.assertEqual(capped, 2)
        self.assertIn("[paged_attn RISK]", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
