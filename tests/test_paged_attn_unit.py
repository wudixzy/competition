import importlib
import importlib.util
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

    def test_attention_env_defaults_are_stable(self):
        module = _load_paged_attn()
        self.assertEqual(module.PagedAttention._PYTORCH_DECODE_THRESHOLD, 32768)
        self.assertEqual(module._PREFIX_BLOCKS_PER_TILE, 32)
        self.assertFalse(module.PagedAttention._FORCE_PAGED_ATTN_V2)
        self.assertTrue(module.PagedAttention._should_use_paged_attention_v1(
            max_seq_len=100000,
            max_num_partitions=196,
            num_seqs=1,
            num_heads=64,
        ))

    def test_attention_env_overrides_are_loaded_at_import(self):
        module = _load_paged_attn(threshold="4096", tile="64", force_v2="1")
        self.assertEqual(module.PagedAttention._PYTORCH_DECODE_THRESHOLD, 4096)
        self.assertEqual(module._PREFIX_BLOCKS_PER_TILE, 64)
        self.assertTrue(module.PagedAttention._FORCE_PAGED_ATTN_V2)
        self.assertFalse(module.PagedAttention._should_use_paged_attention_v1(
            max_seq_len=100000,
            max_num_partitions=196,
            num_seqs=1,
            num_heads=64,
        ))

    def test_attention_env_rejects_invalid_values(self):
        with self.assertRaises(RuntimeError):
            _load_paged_attn(threshold="0")

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
