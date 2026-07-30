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
    torch_mod.float16 = object()
    torch_mod.int32 = object()
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
    extension_mod = types.ModuleType("vllm.bi100_external_extension")
    extension_mod.load_hashed_private_extension = lambda *args, **kwargs: None
    profile_mod = types.ModuleType("vllm.bi100_profile")

    class _NoopTimer:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    profile_mod.bi100_timer = lambda name: _NoopTimer()
    profile_mod.bi100_profile_count = lambda name, **metadata: None
    sys.modules["torch"] = torch_mod
    sys.modules["vllm"] = vllm_mod
    sys.modules["vllm.bi100_env"] = env_mod
    sys.modules["vllm.bi100_external_extension"] = extension_mod
    sys.modules["vllm.bi100_profile"] = profile_mod
    return torch_mod, vllm_mod


def _clear_stubs():
    for name in [
        "torch", "vllm", "vllm.bi100_env",
        "vllm.bi100_external_extension", "vllm.bi100_profile",
    ]:
        sys.modules.pop(name, None)


def _load_paged_attn(**env):
    with _EnvPatch(
            BI100_PYTORCH_DECODE_THRESHOLD=env.get("threshold"),
            BI100_PREFIX_BLOCKS_PER_TILE=env.get("tile"),
            BI100_FORCE_PAGED_ATTN_V2=env.get("force_v2"),
            BI100_PAGED_ATTN_DIAGNOSTICS=env.get("diagnostics"),
            BI100_ATTN_COREX_FUSED_PREFILL=env.get("fused_prefill"),
            BI100_ATTN_COREX_FUSED_PREFILL_EXTENSION=None,
            BI100_ATTN_COREX_FUSED_PREFILL_EXTENSION_SHA256=None,
            BI100_ATTN_COREX_FUSED_PREFILL_SHADOW=env.get("shadow"),
            BI100_ATTN_COREX_FUSED_PREFILL_SHADOW_REPORT_DIR=(
                env.get("shadow_report_dir")),
            BI100_ATTN_COREX_FUSED_PREFILL_SHADOW_RUN_ID=(
                env.get("shadow_run_id")),
            BI100_ATTN_COREX_FUSED_PREFILL_SHADOW_CONTEXTS=(
                env.get("shadow_contexts")),
            BI100_ATTN_COREX_FUSED_PREFILL_SHADOW_MAX_CALLS_PER_CONTEXT=(
                env.get("shadow_max_calls")),
            BI100_ATTN_COREX_FUSED_PREFILL_SHADOW_NUMERIC_MODE=(
                env.get("shadow_numeric_mode")),
            BI100_ATTN_COREX_FUSED_PREFILL_SHADOW_FAILURE_ACTION=(
                env.get("shadow_failure_action")),
            BI100_ATTN_CAPTURE_REPLAY=env.get("capture"),
            BI100_ATTN_CAPTURE_REPLAY_DIR=env.get("capture_dir"),
            BI100_ATTN_CAPTURE_REPLAY_RUN_ID=env.get("capture_run_id"),
            BI100_ATTN_CAPTURE_REPLAY_CONTEXTS=(
                env.get("capture_contexts")),
            BI100_ATTN_CAPTURE_REPLAY_CALL_ORDINALS=(
                env.get("capture_ordinals")),
            BI100_ATTN_CAPTURE_REPLAY_SOURCE_REVISION=(
                env.get("capture_source_revision")),
            BI100_ATTN_CAPTURE_REPLAY_RUNTIME_IDENTITY=(
                env.get("capture_runtime_identity")),
            BI100_ATTN_CAPTURE_REPLAY_SYNTHETIC_ATTESTATION=(
                env.get("capture_attestation")),
    ):
        old_modules = {
            name: sys.modules.get(name)
            for name in [
                "torch", "vllm", "vllm.bi100_env",
                "vllm.bi100_external_extension", "vllm.bi100_profile",
            ]
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
        self.assertFalse(module._FUSED_PREFILL_SHADOW)
        self.assertEqual(
            module._FUSED_PREFILL_SHADOW_CONTEXTS, (49152, 114688))
        self.assertEqual(
            module._FUSED_PREFILL_SHADOW_MAX_CALLS_PER_CONTEXT, 2)
        self.assertEqual(
            module._FUSED_PREFILL_SHADOW_NUMERIC_MODE, "legacy")
        self.assertEqual(
            module._FUSED_PREFILL_SHADOW_FAILURE_ACTION, "raise")
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
        with self.assertRaisesRegex(RuntimeError, "NUMERIC_MODE"):
            _load_paged_attn(shadow_numeric_mode="unknown")
        with self.assertRaisesRegex(RuntimeError, "FAILURE_ACTION"):
            _load_paged_attn(shadow_failure_action="unknown")

    def test_shadow_context_parser_rejects_ambiguous_buckets(self):
        module = _load_paged_attn()
        self.assertEqual(
            module._parse_fused_prefill_shadow_contexts("0,49152,114688"),
            (0, 49152, 114688),
        )
        for value in ("", "49152,", "49152,49152", "114688,49152", "x"):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                module._parse_fused_prefill_shadow_contexts(value)

    def test_shadow_configuration_requires_private_bound_identity(self):
        module = _load_paged_attn()
        with tempfile.TemporaryDirectory(prefix="bi100-shadow-") as directory:
            path = module._validate_fused_prefill_shadow_configuration(
                True, True, directory, "m1-136-unit")
            self.assertEqual(path, pathlib.Path(directory))
        with self.assertRaisesRegex(RuntimeError, "requires the fused"):
            module._validate_fused_prefill_shadow_configuration(
                True, False, "/tmp/m1-136", "m1-136-unit")
        with self.assertRaisesRegex(RuntimeError, "under /tmp"):
            module._validate_fused_prefill_shadow_configuration(
                True, True, "/var/tmp/m1-136", "m1-136-unit")
        with self.assertRaisesRegex(RuntimeError, "under /tmp"):
            module._validate_fused_prefill_shadow_configuration(
                True, True, "/tmp/../var/tmp/m1-136", "m1-136-unit")
        with tempfile.TemporaryDirectory(prefix="bi100-shadow-link-") as root:
            escaped = pathlib.Path(root) / "escaped"
            escaped.symlink_to("/var/tmp", target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "under /tmp"):
                module._validate_fused_prefill_shadow_configuration(
                    True, True, str(escaped / "m1-136"), "m1-136-unit")
        with self.assertRaisesRegex(RuntimeError, "RUN_ID is invalid"):
            module._validate_fused_prefill_shadow_configuration(
                True, True, "/tmp/m1-136", "bad run id")
        module._FUSED_PREFILL_SHADOW_FAILURE_ACTION = "record"
        with self.assertRaisesRegex(RuntimeError, "calibrated numeric mode"):
            module._validate_fused_prefill_shadow_configuration(
                True, True, "/tmp/m1-136", "m1-136-unit")
        module._FUSED_PREFILL_SHADOW_NUMERIC_MODE = "calibrated"
        with tempfile.TemporaryDirectory(prefix="bi100-shadow-") as directory:
            self.assertEqual(
                module._validate_fused_prefill_shadow_configuration(
                    True, True, directory, "m1-138-unit"),
                pathlib.Path(directory),
            )

    def test_shadow_report_contains_only_aggregate_shape_metrics(self):
        module = _load_paged_attn()
        record = {
            "index": 0,
            "status": "pass",
            "bucket_min_context_tokens": 49152,
            "context_tokens": 57344,
            "query_shape": [8192, 4, 256],
            "query_heads": 4,
            "kv_heads": 1,
            "head_dim": 256,
            "block_size": 16,
            "candidate_finite": True,
            "reference_finite": True,
            "relative_l2": 2.5e-6,
            "max_abs": 0.00048828125,
            "error_stage": None,
            "error_type": None,
        }
        report = module._build_fused_prefill_shadow_report([record])
        self.assertEqual(report["status"], "collecting")
        self.assertEqual(report["observations"]["passed"], 1)
        self.assertEqual(report["observations"]["maximum_relative_l2"], 2.5e-6)
        serialized = repr(report).lower()
        for forbidden in ("prompt_text", "actual_token_key", "tensor_payload"):
            self.assertNotIn(forbidden, serialized)
        self.assertTrue(all(
            value is False for value in report["privacy"].values()))

    def test_calibrated_shadow_report_uses_rounding_relative_metrics(self):
        module = _load_paged_attn(shadow_numeric_mode="calibrated")
        module._FUSED_PREFILL_SHADOW_FAILURE_ACTION = "record"
        record = {
            "index": 0,
            "status": "pass",
            "bucket_min_context_tokens": 49152,
            "context_tokens": 57344,
            "query_shape": [8192, 4, 256],
            "query_heads": 4,
            "kv_heads": 1,
            "head_dim": 256,
            "block_size": 16,
            "candidate_finite": True,
            "reference_finite": True,
            "relative_l2": 7.0e-6,
            "max_abs": 0.001953125,
            "candidate_to_fp32_relative_l2": 5.0e-4,
            "candidate_to_fp32_max_abs": 0.0014,
            "rounded_to_fp32_relative_l2": 3.0e-4,
            "rounded_to_fp32_max_abs": 0.0009,
            "relative_l2_baseline_ratio": 5.0 / 3.0,
            "max_abs_baseline_ratio": 1.4 / 0.9,
            "error_stage": None,
            "error_type": None,
        }
        report = module._build_fused_prefill_shadow_report([record])
        self.assertEqual(
            report["schema"],
            "bi100-fused-prefill-real-activation-calibrated-shadow-v1",
        )
        self.assertEqual(
            report["thresholds"][
                "maximum_error_multiple_over_fp16_rounding"],
            2.0,
        )
        self.assertEqual(
            report["thresholds"]["fixed_max_abs_role"],
            "diagnostic_only",
        )
        self.assertEqual(
            report["observations"][
                "maximum_candidate_to_fp32_max_abs"],
            0.0014,
        )

    def test_calibrated_scalar_gate_is_hard_but_scale_aware(self):
        module = _load_paged_attn(shadow_numeric_mode="calibrated")
        metrics = {
            "relative_l2": 7.1e-6,
            "candidate_to_fp32_relative_l2": 5.0e-4,
            "rounded_to_fp32_relative_l2": 3.0e-4,
            "candidate_to_fp32_max_abs": 0.0015,
            "rounded_to_fp32_max_abs": 0.0008,
        }
        self.assertTrue(
            module._calibrated_shadow_metrics_qualified(metrics))
        changed = dict(metrics)
        changed["candidate_to_fp32_max_abs"] = 0.0017
        self.assertFalse(
            module._calibrated_shadow_metrics_qualified(changed))
        changed = dict(metrics)
        changed["relative_l2"] = 1.1e-5
        self.assertFalse(
            module._calibrated_shadow_metrics_qualified(changed))

    def test_shadow_rank_prefers_initialized_distributed_rank(self):
        module = _load_paged_attn()
        module.torch.distributed = types.SimpleNamespace(
            is_available=lambda: True,
            is_initialized=lambda: True,
            get_rank=lambda: 3,
        )
        with _EnvPatch(RANK=None, LOCAL_RANK=None):
            self.assertEqual(module._fused_prefill_shadow_rank(), 3)

    def test_enabling_shadow_without_native_extension_fails_startup(self):
        with self.assertRaisesRegex(RuntimeError, "requires the fused"):
            _load_paged_attn(
                fused_prefill="1",
                shadow="1",
                shadow_report_dir="/tmp/m1-136-unit",
                shadow_run_id="m1-136-unit",
            )

    def test_capture_requires_private_baseline_and_attestation(self):
        module = _load_paged_attn()
        revision = "a" * 40
        with tempfile.TemporaryDirectory(
                prefix="bi100-activation-bank-") as directory:
            path = module._validate_activation_capture_configuration(
                True,
                False,
                directory,
                "m1-140-unit",
                revision,
                "bare-host-overlay-v1:abc",
                "synthetic-exact-prompt-v1",
            )
            self.assertEqual(path, pathlib.Path(directory))
        with self.assertRaisesRegex(RuntimeError, "baseline PyTorch"):
            module._validate_activation_capture_configuration(
                True,
                True,
                "/tmp/m1-140",
                "m1-140-unit",
                revision,
                "bare-host-overlay-v1:abc",
                "synthetic-exact-prompt-v1",
            )
        with self.assertRaisesRegex(RuntimeError, "attestation"):
            module._validate_activation_capture_configuration(
                True,
                False,
                "/tmp/m1-140",
                "m1-140-unit",
                revision,
                "bare-host-overlay-v1:abc",
                "unknown",
            )

    def test_capture_selects_fixed_full_attention_ordinals_once(self):
        module = _load_paged_attn()
        module._ACTIVATION_CAPTURE_ENABLED = True
        module._ACTIVATION_CAPTURE_CONTEXTS = (24576, 57344)
        module._ACTIVATION_CAPTURE_CALL_ORDINALS = (0, 4, 9)
        module._ACTIVATION_CAPTURE_STATE = {
            "pid": None,
            "seen_by_bucket": {},
            "records": [],
        }
        selected = [
            module._reserve_activation_capture(32768)
            for _ in range(11)
        ]
        self.assertEqual(
            [value for value in selected if value is not None],
            [(24576, 0), (24576, 4), (24576, 9)],
        )
        self.assertEqual(
            module._reserve_activation_capture(65536),
            (57344, 0),
        )

    def test_shadow_context_buckets_are_disjoint(self):
        module = _load_paged_attn()
        with tempfile.TemporaryDirectory(prefix="bi100-shadow-buckets-") as root:
            module._FUSED_PREFILL_SHADOW = True
            module._FUSED_PREFILL_SHADOW_REPORT_DIR = pathlib.Path(root)
            module._FUSED_PREFILL_SHADOW_RUN_ID = "m1-136-unit"
            module._FUSED_PREFILL_SHADOW_CONTEXTS = (49152, 114688)
            module._FUSED_PREFILL_SHADOW_MAX_CALLS_PER_CONTEXT = 2
            module._FUSED_PREFILL_SHADOW_STATE = {"pid": None, "records": []}
            query = types.SimpleNamespace(shape=(8176, 4, 256))
            with _EnvPatch(RANK="0"):
                self.assertEqual(module._reserve_fused_prefill_shadow(
                    query, 122880, 4, 1, 256, 16), 0)
                self.assertEqual(module._reserve_fused_prefill_shadow(
                    query, 122880, 4, 1, 256, 16), 1)
                self.assertIsNone(module._reserve_fused_prefill_shadow(
                    query, 122880, 4, 1, 256, 16))
            buckets = [
                record["bucket_min_context_tokens"]
                for record in module._FUSED_PREFILL_SHADOW_STATE["records"]
            ]
            self.assertEqual(buckets, [114688, 114688])

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
