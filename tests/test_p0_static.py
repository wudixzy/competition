import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwen3_6_scripts.patch_utils import replace_once


def read(relpath: str) -> str:
    return (ROOT / relpath).read_text()


class PatchUtilsTest(unittest.TestCase):

    def test_replace_once_required_anchor_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "target.py"
            path.write_text("alpha\n")
            with self.assertRaises(RuntimeError):
                with redirect_stdout(StringIO()):
                    replace_once(path, "missing", "new", required=True)

    def test_replace_once_optional_anchor_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "target.py"
            path.write_text("alpha\n")
            with redirect_stdout(StringIO()):
                patched = replace_once(path, "missing", "new", required=False)
            self.assertFalse(patched)
            self.assertEqual(path.read_text(), "alpha\n")


class P0StaticCoverageTest(unittest.TestCase):

    def test_registry_aliases_are_registered(self):
        src = read("qwen3_6_scripts/patch_vllm_qwen3_5.py")
        unit_src = read("tests/test_patch_registry_unit.py")
        for name in [
                "Qwen3_5ForCausalLM",
                "Qwen3_5MoeForCausalLM",
                "Qwen3_6ForCausalLM",
                "Qwen3_6MoeForCausalLM",
                "Qwen3ForCausalLM",
                "Qwen3MoeForCausalLM",
        ]:
            self.assertIn(name, src)
        self.assertNotIn("Set config.json", src)
        self.assertIn("ast.parse", src)
        self.assertNotIn("exec_module", src)
        self.assertNotIn("import torch", src)
        self.assertIn("test_registry_alias_patch_installs_qwen36_aliases",
                      unit_src)
        self.assertIn("test_registry_alias_patch_fails_fast_when_anchor_missing",
                      unit_src)
        self.assertIn(
            "test_registry_verification_does_not_execute_model_module",
            unit_src)

    def test_protocol_thinking_and_tool_choice_none(self):
        src = read("qwen3_6_scripts/protocol.py")
        parser_src = read("qwen3_6_scripts/reasoning/qwen3_reasoning_parser.py")
        self.assertIn("thinking:", src)
        self.assertIn("def normalize_thinking", src)
        self.assertIn('chat_template_kwargs["enable_thinking"]', src)
        self.assertIn('if data["tool_choice"] == "none":', src)
        self.assertNotIn('`tool_choice="none" is not supported.', src)
        self.assertIn("if not self.thinking_enabled:", parser_src)
        self.assertIn("return None, content or \"\"", parser_src)

    def test_protocol_rejects_forced_tool_with_guided_decoding(self):
        src = read("qwen3_6_scripts/protocol.py")
        unit_src = read("tests/test_protocol_unit.py")
        self.assertIn("guide_count > 0", src)
        self.assertIn('not in ("none", "auto")', src)
        self.assertIn("You can only either use guided decoding or tools", src)
        self.assertNotIn("guide_count > 1 and data.get(\"tool_choice\"", src)
        self.assertIn(
            "test_thinking_disabled_variants_normalize_to_chat_template_kwargs",
            unit_src)
        self.assertIn("test_tools_default_to_auto_but_explicit_none_is_preserved",
                      unit_src)
        self.assertIn("test_named_tool_requires_matching_tool_and_conflicts",
                      unit_src)

    def test_smoke_covers_json_schema_response_format(self):
        src = read("tests/smoke_api.py")
        self.assertIn("def test_response_format_json_schema", src)
        self.assertIn('"type": "json_schema"', src)
        self.assertIn('"additionalProperties": False', src)
        self.assertIn("test_response_format_json_schema", src)

    def test_smoke_covers_streaming_tool_call(self):
        src = read("tests/smoke_api.py")
        self.assertIn("def test_tool_call_forced_streaming", src)
        self.assertIn('"stream": True', src)
        self.assertIn("delta.get(\"tool_calls\")", src)
        self.assertIn("json.loads(arguments or \"{}\")", src)
        self.assertIn("test_tool_call_forced_streaming", src)

    def test_smoke_covers_multilingual_multiturn_and_sampling_bounds(self):
        src = read("tests/smoke_api.py")
        self.assertIn("def test_multilingual_multiturn", src)
        self.assertIn("こんにちは", src)
        self.assertIn("\\ufffd", src)
        self.assertIn("def test_sampling_boundaries", src)
        self.assertIn('"temperature": -0.1', src)
        self.assertIn("expect=400", src)
        self.assertIn("test_multilingual_multiturn", src)
        self.assertIn("test_sampling_boundaries", src)

    def test_smoke_covers_seed_determinism(self):
        src = read("tests/smoke_api.py")
        self.assertIn("def test_seed_determinism", src)
        self.assertIn('"seed": 42', src)
        self.assertIn("first == second", src)
        self.assertIn("test_seed_determinism", src)

    def test_smoke_client_http_error_paths_are_exercised(self):
        src = read("tests/smoke_api.py")
        unit_src = read("tests/test_clients_unit.py")
        self.assertNotIn("from PIL", src)
        self.assertIn("def _solid_png_data_url", src)
        self.assertIn("zlib.compress", src)
        self.assertIn("except urllib.error.HTTPError", src)
        self.assertIn("except json.JSONDecodeError", src)
        self.assertIn("test_smoke_request_json_preserves_json_http_error_body",
                      unit_src)
        self.assertIn("test_smoke_request_json_preserves_raw_http_error_body",
                      unit_src)

    def test_bi100_preflight_checks_each_gpu_with_timeout(self):
        src = read("tests/bi100_preflight.py")
        self.assertIn("COREX_LIBRARY_PATHS", src)
        self.assertIn("torch.cuda.mem_get_info()", src)
        self.assertIn("torch.cuda.synchronize()", src)
        self.assertIn("TimeoutExpired", src)
        self.assertIn("returncode\": 124", src)
        self.assertIn("def _clean_stream", src)
        self.assertIn("decode(\"utf-8\", \"replace\")", src)
        self.assertIn("Probe BI100 GPUs before launching TP=4 vLLM", src)

    def test_bi100_nccl_preflight_checks_timed_all_reduce(self):
        src = read("tests/bi100_nccl_preflight.py")
        self.assertIn("backend=\"nccl\"", src)
        self.assertIn("dist.all_reduce(tensor)", src)
        self.assertIn("mp.get_context(\"spawn\")", src)
        self.assertIn("process.terminate()", src)
        self.assertIn("unexpected all_reduce value", src)
        self.assertIn("Run a timed BI100 NCCL all_reduce preflight", src)

    def test_streaming_tool_arguments_not_double_json_encoded(self):
        src = read("qwen3_6_scripts/serving_chat.py")
        unit_src = read("tests/test_serving_chat_unit.py")
        self.assertIn("def _serialize_tool_arguments", src)
        self.assertIn("isinstance(arguments, str)", src)
        self.assertIn("return arguments", src)
        self.assertIn("expected_call = _serialize_tool_arguments", src)
        self.assertIn(
            "test_tool_arguments_string_is_not_double_json_encoded",
            unit_src)
        self.assertIn("ast.parse(SERVING_CHAT.read_text()", unit_src)

    def test_tool_parser_conversion_fallbacks_are_logged(self):
        src = read("qwen3_6_scripts/qwen3coder_tool_parser.py")
        self.assertIn("Could not JSON-decode parameter", src)
        self.assertIn("falling back to literal evaluation", src)
        self.assertIn("Could not literal-eval parameter", src)
        self.assertIn("returning string value", src)
        self.assertNotIn(
            "except (json.JSONDecodeError, TypeError, ValueError):\n"
            "                    pass",
            src,
        )
        self.assertNotIn(
            "except (ValueError, SyntaxError, TypeError):\n"
            "                pass",
            src,
        )

    def test_streaming_tool_argument_names_are_json_escaped(self):
        src = read("qwen3_6_scripts/qwen3coder_tool_parser.py")
        unit_src = read("tests/test_tool_parser_unit.py")
        self.assertIn("json.dumps(current_param_name, ensure_ascii=False)", src)
        self.assertIn('json_fragments.append(f"{sep}{key}: {serialized}")',
                      src)
        self.assertNotIn('f\'{sep}"{current_param_name}": {serialized}\'',
                         src)
        self.assertIn("test_streaming_argument_name_is_json_escaped",
                      unit_src)
        self.assertIn('json.loads("{" + arguments + "}")', unit_src)

    def test_transformers_config_stubs_do_not_swallow_broad_exceptions(self):
        for relpath in [
                "qwen3_6_scripts/qwen3_5/configuration_qwen3_5.py",
                "qwen3_6_scripts/qwen3_5_moe/configuration_qwen3_5_moe.py",
        ]:
            src = read(relpath)
            self.assertIn("except ImportError:", src, relpath)
            self.assertNotIn("except Exception:", src, relpath)
            self.assertIn("RopeParameters = dict", src, relpath)

    def test_attention_guard_defaults_to_raise(self):
        src = read("qwen3_6_scripts/paged_attn.py")
        unit_src = read("tests/test_paged_attn_unit.py")
        self.assertIn("BI100_ALLOW_PREFIX_GUARD_CAP", src)
        self.assertIn("BI100_FORCE_PAGED_ATTN_V2", src)
        self.assertIn("def _should_use_paged_attention_v1", src)
        self.assertIn("def _validate_prefix_block_table", src)
        self.assertIn("raise RuntimeError(msg)", src)
        self.assertIn("[paged_attn RISK]", src)
        self.assertIn("test_attention_env_defaults_are_stable", unit_src)
        self.assertIn("test_attention_env_overrides_are_loaded_at_import",
                      unit_src)
        self.assertIn("test_prefix_block_table_guard_raises_by_default",
                      unit_src)
        self.assertIn("test_prefix_block_table_guard_debug_cap_is_explicit",
                      unit_src)

    def test_gated_deltanet_nonfinite_defaults_to_raise(self):
        src = read("qwen3_6_scripts/qwen3_5.py")
        self.assertIn("def _check_gdn_finite", src)
        self.assertIn('env_bool("BI100_GDN_FINITE_CHECK", False)', src)
        self.assertIn("if not _GDN_FINITE_CHECK:", src)
        self.assertIn("or _ALLOW_GDN_NAN_ZERO", src)
        self.assertIn("BI100_GDN_ALLOW_NAN_ZERO", src)
        self.assertIn("raise RuntimeError(msg)", src)
        self.assertNotIn("NaN in prefill GatedDeltaNet", src)
        self.assertNotIn("NaN in decode GatedDeltaNet", src)

    def test_gdn_input_projections_are_single_merged_gemm(self):
        src = read("qwen3_6_scripts/qwen3_5.py")
        self.assertIn("def _load_gdn_projection_weight", src)
        self.assertIn("self.in_proj_qkvzba = MergedColumnParallelLinear", src)
        self.assertIn("projected, _ = self.in_proj_qkvzba(hidden_states)", src)
        self.assertNotIn("self.in_proj_z = ColumnParallelLinear", src)
        self.assertNotIn("self.in_proj_b = ColumnParallelLinear", src)
        self.assertNotIn("self.in_proj_a = ColumnParallelLinear", src)

    def test_moe_prefill_groups_routes_once(self):
        src = read("qwen3_6_scripts/qwen3_5.py")
        self.assertIn("torch.argsort(flat_eids, stable=True)", src)
        self.assertIn("torch.bincount(", src)
        self.assertNotIn("mask = (topk_ids == eid)", src)

    def test_qwen36_image_mapper_pins_rgb_channels_last(self):
        src = read("qwen3_6_scripts/qwen3_5.py")
        self.assertIn("ChannelDimension", src)
        self.assertIn('image.convert("RGB")', src)
        self.assertIn("do_convert_rgb=True", src)
        self.assertIn("input_data_format=ChannelDimension.LAST", src)

    def test_model_runner_patch_aligns_mrope_with_physical_query(self):
        src = read("qwen3_6_scripts/patch_model_runner.py")
        unit_src = read("tests/test_mrope_chunk_alignment_unit.py")
        self.assertIn("def _slice_mrope_positions", src)
        self.assertIn("context_len=0", src)
        self.assertIn("inter_data.seq_lens[seq_idx]", src)
        self.assertIn("len(inter_data.input_tokens[seq_idx])", src)
        self.assertIn("positions, uncomputed_start, None", src)
        self.assertIn("test_chunked_multimodal_positions_match_current_query",
                      unit_src)
        self.assertIn("test_alignment_mismatch_fails_before_gpu_execution",
                      unit_src)
        probe_src = read("tests/mrope_chunk_api.py")
        self.assertIn('for label in ("cold", "warm")', probe_src)
        self.assertIn('cold["prompt_tokens"] <= 8192', probe_src)
        self.assertIn('warm["cached_tokens"] <= 0', probe_src)
        self.assertIn('cold["output_sha256"] != warm["output_sha256"]',
                      probe_src)

    def test_scheduler_gates_kv_hits_on_exact_gdn_state(self):
        scheduler_src = read("qwen3_6_scripts/scheduler.py")
        model_src = read("qwen3_6_scripts/qwen3_5.py")
        self.assertIn("_gdn_request_restore_keys", scheduler_src)
        self.assertIn("_gdn_request_capture_targets", scheduler_src)
        self.assertIn("gdn_restore_key=gdn_restore_key", scheduler_src)
        self.assertIn("gdn_capture_points=gdn_capture_points", scheduler_src)
        self.assertIn("gdn_evict_keys=gdn_evict_keys", scheduler_src)
        self.assertIn("Tuple[int, bytes]", model_src)
        smoke_src = read("tests/smoke_api.py")
        self.assertIn("_message(cached) == _message(uncached)", smoke_src)
        self.assertIn('"seed": 123', smoke_src)
        self.assertIn("_validate_gdn_prefix_key", model_src)
        self.assertIn("len(key[1]) != 32", model_src)
        self.assertIn("captured_temporal_states", model_src)
        self.assertNotIn("_gdn_prefix_checkpoints", scheduler_src)

    def test_executor_startup_debug_is_opt_in(self):
        patch_ops = read("qwen3_6_scripts/patch_ops.sh")
        debug_src = read("qwen3_6_scripts/patch_executor_startup_debug.py")
        self.assertIn("patch_executor_startup_debug.py", patch_ops)
        self.assertIn("BI100_EXECUTOR_STARTUP_DEBUG", debug_src)
        self.assertIn("os.getenv", debug_src)
        self.assertIn('\\"1\\"', debug_src)
        self.assertIn("[BI100 startup]", debug_src)
        self.assertIn("[BI100 worker]", debug_src)
        self.assertIn("required=True", debug_src)
        self.assertIn("already_contains=", debug_src)

    def test_benchmark_startup_trace_has_critical_stages(self):
        api_src = read("qwen3_6_scripts/api_server.py")
        model_src = read("qwen3_6_scripts/qwen3_5.py")
        for marker in ["api_server stdlib imports complete",
                       "engine client construction completed",
                       "starting HTTP server"]:
            self.assertIn(marker, api_src)
        for marker in ["qwen3_5 stdlib imports complete",
                       "Qwen3_5ForCausalLM initialization begin",
                       "first model forward entered",
                       "MoE load_weights complete"]:
            self.assertIn(marker, model_src)
        self.assertIn("flush=True", api_src)
        self.assertIn("flush=True", model_src)

    def test_env_knobs_are_registered_and_strongly_validated(self):
        env_src = read("qwen3_6_scripts/bi100_env.py")
        paged_src = read("qwen3_6_scripts/paged_attn.py")
        qwen_src = read("qwen3_6_scripts/qwen3_5.py")
        docs = read("docs/ENV_KNOBS.md")
        self.assertIn("def env_int", env_src)
        self.assertIn("def env_bool", env_src)
        self.assertIn("outside [", env_src)
        for name in [
                "BI100_PYTORCH_DECODE_THRESHOLD",
                "BI100_PREFIX_BLOCKS_PER_TILE",
                "BI100_FORCE_PAGED_ATTN_V2",
                "BI100_ALLOW_PREFIX_GUARD_CAP",
                "BI100_PAGED_ATTN_DIAGNOSTICS",
                "BI100_ATTN_COREX_PAGED_GATHER",
                "BI100_GDN_ALLOW_NAN_ZERO",
                "BI100_GDN_FINITE_CHECK",
                "BI100_GDN_COREX_BETA_DECAY",
                "BI100_GDN_COREX_QK_MAP",
                "BI100_GDN_COREX_PACKED_DECODE",
                "BI100_DNN_CHUNK",
                "BI100_PROFILE",
                "BI100_PROFILE_INCLUDE_STARTUP",
        ]:
            self.assertIn(name, docs)
        self.assertIn("env_int(", paged_src)
        self.assertIn("env_bool(", paged_src)
        self.assertIn("env_int(\"BI100_DNN_CHUNK\"", qwen_src)
        self.assertNotIn("int(os.getenv", paged_src)
        self.assertNotIn("_DNN_CHUNK = 4096", qwen_src)

    def test_bi100_profile_is_default_off_and_installed_by_patch_ops(self):
        patch_ops = read("qwen3_6_scripts/patch_ops.sh")
        profile_src = read("qwen3_6_scripts/bi100_profile.py")
        qwen_src = read("qwen3_6_scripts/qwen3_5.py")
        paged_src = read("qwen3_6_scripts/paged_attn.py")
        self.assertIn("cp ./bi100_profile.py", patch_ops)
        self.assertIn('os.getenv("BI100_PROFILE", "0") == "1"', profile_src)
        self.assertIn("BI100_IN_STARTUP_PROFILE", profile_src)
        self.assertIn("BI100_PROFILE_INCLUDE_STARTUP", profile_src)
        self.assertIn("[BI100_PROFILE]", profile_src)
        self.assertIn("bi100_timer", qwen_src)
        self.assertIn("bi100_timer", paged_src)

    def test_corex_gdn_causal_conv_is_built_with_explicit_fallback(self):
        patch_ops = read("qwen3_6_scripts/patch_ops.sh")
        qwen_src = read("qwen3_6_scripts/qwen3_5.py")
        build_src = read("qwen3_6_scripts/build_corex_gdn_causal_conv.sh")
        self.assertNotIn("build_corex_gdn_causal_conv.sh", patch_ops)
        self.assertIn("corex_gdn_causal_conv.so", build_src)
        self.assertIn("_USE_COREX_GDN_CAUSAL_CONV", qwen_src)
        self.assertIn("BI100_GDN_COREX_CAUSAL_CONV", qwen_src)
        self.assertIn("_torch_causal_conv1d_update", qwen_src)

    def test_corex_gdn_gated_norm_is_built_with_explicit_fallback(self):
        patch_ops = read("qwen3_6_scripts/patch_ops.sh")
        qwen_src = read("qwen3_6_scripts/qwen3_5.py")
        build_src = read("qwen3_6_scripts/build_corex_gdn_gated_norm.sh")
        self.assertNotIn("build_corex_gdn_gated_norm.sh", patch_ops)
        self.assertIn("corex_gdn_gated_norm.so", build_src)
        self.assertIn("BI100_GDN_COREX_GATED_NORM", qwen_src)
        self.assertIn("forward_decode", qwen_src)
        self.assertIn("return self.forward(hidden_states, gate)", qwen_src)

    def test_corex_gdn_beta_decay_is_built_with_explicit_fallback(self):
        patch_ops = read("qwen3_6_scripts/patch_ops.sh")
        qwen_src = read("qwen3_6_scripts/qwen3_5.py")
        build_src = read("qwen3_6_scripts/build_corex_gdn_beta_decay.sh")
        kernel_src = read("qwen3_6_scripts/corex_gdn_beta_decay.cu")
        self.assertNotIn("build_corex_gdn_beta_decay.sh", patch_ops)
        self.assertIn("corex_gdn_beta_decay.so", build_src)
        self.assertIn("BI100_GDN_COREX_BETA_DECAY", qwen_src)
        self.assertIn("_corex_gdn_beta_decay.beta_decay", qwen_src)
        self.assertIn("b_all.sigmoid()", qwen_src)
        self.assertIn("F.softplus(a_all.float() + self.dt_bias)", qwen_src)
        self.assertIn("__float2half(beta_fp32)", kernel_src)

    def test_corex_gdn_qk_map_is_built_with_explicit_fallback(self):
        patch_ops = read("qwen3_6_scripts/patch_ops.sh")
        qwen_src = read("qwen3_6_scripts/qwen3_5.py")
        build_src = read("qwen3_6_scripts/build_corex_gdn_qk_map.sh")
        kernel_src = read("qwen3_6_scripts/corex_gdn_qk_map.cu")
        self.assertNotIn("build_corex_gdn_qk_map.sh", patch_ops)
        self.assertIn("corex_gdn_qk_map.so", build_src)
        self.assertIn("BI100_GDN_COREX_QK_MAP", qwen_src)
        self.assertIn("_corex_gdn_qk_map.qk_map", qwen_src)
        self.assertIn("q_raw.repeat_interleave", qwen_src)
        self.assertIn("kQueryScale", kernel_src)

    def test_corex_gdn_packed_decode_is_default_off_and_shape_guarded(self):
        patch_ops = read("qwen3_6_scripts/patch_ops.sh")
        qwen_src = read("qwen3_6_scripts/qwen3_5.py")
        build_src = read("qwen3_6_scripts/build_corex_gdn_packed_decode.sh")
        kernel_src = read("qwen3_6_scripts/corex_gdn_packed_decode.cu")
        run_config = read("computility-run.yaml")
        self.assertNotIn("build_corex_gdn_packed_decode.sh", patch_ops)
        self.assertIn("corex_gdn_packed_decode.so", build_src)
        self.assertIn('env_bool("BI100_GDN_COREX_PACKED_DECODE", False)',
                      qwen_src)
        self.assertIn("temporal_state.shape == (1, 8, 128, 128)", qwen_src)
        self.assertIn("_corex_gdn_packed_decode.packed_decode", qwen_src)
        self.assertIn("state.size(0) == 1", kernel_src)
        self.assertIn("packed decode only supports one sequence", kernel_src)
        self.assertIn("BI100_GDN_COREX_PACKED_DECODE", run_config)
        self.assertRegex(
            run_config,
            r"name: BI100_GDN_COREX_PACKED_DECODE\s+value: 1")

    def test_corex_attention_head_rms_norm_is_decode_only_and_fallback_safe(self):
        patch_ops = read("qwen3_6_scripts/patch_ops.sh")
        qwen_src = read("qwen3_6_scripts/qwen3_5.py")
        build_src = read("qwen3_6_scripts/build_corex_attn_head_rms_norm.sh")
        kernel_src = read("qwen3_6_scripts/corex_attn_head_rms_norm.cu")
        self.assertNotIn("build_corex_attn_head_rms_norm.sh", patch_ops)
        self.assertIn("corex_attn_head_rms_norm.so", build_src)
        self.assertIn("BI100_ATTN_COREX_HEAD_RMS_NORM", qwen_src)
        self.assertIn("class Qwen3_5AttentionHeadRMSNorm", qwen_src)
        self.assertIn("x.shape[0] == 1", qwen_src)
        self.assertIn("return super().forward_cuda(x, residual)", qwen_src)
        self.assertIn("squares.mean(dim=-1, keepdim=True)", qwen_src)
        self.assertIn("__float2half_rn", kernel_src)

    def test_corex_moe_exact_reduce_is_built_with_explicit_fallback(self):
        patch_ops = read("qwen3_6_scripts/patch_ops.sh")
        qwen_src = read("qwen3_6_scripts/qwen3_5.py")
        build_src = read("qwen3_6_scripts/build_corex_moe_exact_reduce.sh")
        self.assertNotIn("build_corex_moe_exact_reduce.sh", patch_ops)
        self.assertIn("corex_moe_exact_reduce.so", build_src)
        self.assertIn("BI100_MOE_COREX_EXACT_REDUCE", qwen_src)
        self.assertIn("_corex_moe_exact_reduce.serial_float", qwen_src)
        self.assertIn("expert_out * ws.unsqueeze", qwen_src)
        self.assertIn("BI100_MOE_FUSED_ACTIVATION", qwen_src)
        self.assertIn("act = self.act_fn(gate_up)", qwen_src)

    def test_corex_moe_weight_gather_is_built_with_explicit_fallback(self):
        patch_ops = read("qwen3_6_scripts/patch_ops.sh")
        qwen_src = read("qwen3_6_scripts/qwen3_5.py")
        build_src = read("qwen3_6_scripts/build_corex_moe_weight_gather.sh")
        kernel_src = read("qwen3_6_scripts/corex_moe_weight_gather.cu")
        self.assertNotIn("build_corex_moe_weight_gather.sh", patch_ops)
        self.assertIn("corex_moe_weight_gather.so", build_src)
        self.assertIn("BI100_MOE_COREX_WEIGHT_GATHER", qwen_src)
        self.assertIn("_corex_moe_weight_gather.gather", qwen_src)
        self.assertIn("w13_sel = w13[eids]", qwen_src)
        self.assertIn("constexpr int kGridX = 8", kernel_src)
        self.assertIn("uint4", kernel_src)

    def test_corex_moe_direct_routed_is_guarded_and_submission_enabled(self):
        patch_ops = read("qwen3_6_scripts/patch_ops.sh")
        qwen_src = read("qwen3_6_scripts/qwen3_5.py")
        build_src = read("qwen3_6_scripts/build_corex_moe_direct_routed.sh")
        kernel_src = read("qwen3_6_scripts/corex_moe_direct_routed.cu")
        knobs = read("docs/ENV_KNOBS.md")
        submission = read("computility-run.yaml")
        self.assertNotIn("build_corex_moe_direct_routed.sh", patch_ops)
        self.assertIn("corex_moe_direct_routed.so", build_src)
        self.assertIn(
            'env_bool("BI100_MOE_COREX_DIRECT_ROUTED", False)', qwen_src)
        self.assertIn("hidden_states.shape == (1, 2048)", qwen_src)
        self.assertIn("w13.shape == (256, 256, 2048)", qwen_src)
        self.assertIn("w2.shape == (256, 2048, 128)", qwen_src)
        self.assertIn("_corex_moe_direct_routed.w13", qwen_src)
        self.assertIn("_corex_moe_direct_routed.w2_reduce", qwen_src)
        self.assertIn("constexpr int kExperts = 256", kernel_src)
        self.assertNotIn("w13_silu", kernel_src)
        self.assertIn("BI100_MOE_COREX_DIRECT_ROUTED", knobs)
        self.assertIn("name: BI100_MOE_COREX_DIRECT_ROUTED", submission)
        self.assertRegex(
            submission,
            r"name: BI100_MOE_COREX_DIRECT_ROUTED\s+value: 1",
        )

    def test_corex_paged_kv_gather_is_built_with_explicit_fallback(self):
        patch_ops = read("qwen3_6_scripts/patch_ops.sh")
        paged_src = read("qwen3_6_scripts/paged_attn.py")
        build_src = read("qwen3_6_scripts/build_corex_paged_kv_gather.sh")
        self.assertNotIn("build_corex_paged_kv_gather.sh", patch_ops)
        self.assertIn("corex_paged_kv_gather.so", build_src)
        self.assertIn("BI100_ATTN_COREX_PAGED_GATHER", paged_src)
        self.assertIn("_corex_paged_kv_gather.gather", paged_src)
        self.assertIn("key_cache[blk_ids]", paged_src)
        kernel_src = read("qwen3_6_scripts/corex_paged_kv_gather.cu")
        self.assertIn("kSmallGridMaxSeqLen = 96 * 1024", kernel_src)
        self.assertIn("kSmallGridBlocks = 256", kernel_src)

    def test_docker_sets_corex_environment_and_invokes_explicit_bash(self):
        dockerfile = read("Dockerfile")
        patch_ops = read("qwen3_6_scripts/patch_ops.sh")
        for fragment in [
                "ENV PATH=/usr/local/corex/bin:/usr/local/corex-3.2.3/bin:"
                "/usr/local/openmpi/bin:${PATH}",
                "ENV PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:"
                "/usr/local/corex/lib/python3/dist-packages",
                "ENV LD_LIBRARY_PATH=/usr/local/corex/lib:"
                "/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:"
                "/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib",
        ]:
            self.assertIn(fragment, dockerfile)
        self.assertIn(
            "RUN cd ./qwen3_6_scripts && bash ./patch_ops.sh", dockerfile)
        for name in ["VLLM_ENGINE_ITERATION_TIMEOUT_S=3600",
                     "PYTHONUNBUFFERED=1", "PYTHONFAULTHANDLER=1",
                     "BI100_EXECUTOR_STARTUP_DEBUG=1"]:
            self.assertIn(name, dockerfile)
        self.assertIn("[BI100 BUILD]", patch_ops)
        self.assertEqual(patch_ops.splitlines()[0], "#!/usr/bin/env bash")

    def test_patch_ops_installs_hash_pinned_corex_bundle_without_compiler(self):
        patch_ops = read("qwen3_6_scripts/patch_ops.sh")
        installer = read("qwen3_6_scripts/install_prebuilt_corex.sh")
        manifest = read(
            "qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10/SHA256SUMS")
        artifacts = [
            "corex_attn_head_rms_norm.so",
            "corex_gdn_beta_decay.so",
            "corex_gdn_causal_conv.so",
            "corex_gdn_gated_norm.so",
            "corex_gdn_packed_decode.so",
            "corex_gdn_qk_map.so",
            "corex_moe_direct_routed.so",
            "corex_moe_exact_reduce.so",
            "corex_moe_weight_gather.so",
            "corex_paged_kv_gather.so",
        ]
        self.assertIn("bash ./install_prebuilt_corex.sh", patch_ops)
        self.assertNotIn("build_corex_", patch_ops)
        self.assertIn("sha256sum --strict --check SHA256SUMS", installer)
        self.assertNotIn("import torch", installer)
        self.assertNotIn("torch.ops.load_library", installer)
        self.assertIn('header[:4] != b"\\x7fELF"', installer)
        self.assertIn("machine != 62", installer)
        self.assertEqual(len(manifest.splitlines()), len(artifacts))
        for artifact in artifacts:
            self.assertIn(artifact, manifest)

    def test_patch_ops_uses_offline_transformers_wheel_and_metadata_gate(self):
        src = read("qwen3_6_scripts/patch_ops.sh")
        self.assertIn("--no-index", src)
        self.assertIn("--no-deps", src)
        self.assertIn("TRANSFORMERS_REQUIRED_VERSION=\"4.55.3\"", src)
        self.assertGreaterEqual(
            src.count('importlib.metadata.version("transformers")'), 2)
        self.assertNotIn("import transformers", src)
        self.assertNotIn("pypi.tuna", src)
        self.assertIn("python3 -m py_compile", src)
        self.assertNotIn("patched vllm imports", src)
        self.assertTrue(
            src.rstrip().endswith('build_stage "patch script completed"'))

    def test_worker_profile_override_patch_is_diagnostic_only(self):
        patch_ops = read("qwen3_6_scripts/patch_ops.sh")
        patch_src = read("qwen3_6_scripts/patch_worker_profile_override.py")
        launch_src = read("launch_service")
        docs = read("docs/ENV_KNOBS.md")
        self.assertNotIn("patch_worker_profile_override.py", patch_ops)
        self.assertIn("num_gpu_blocks_override is not None", patch_src)
        self.assertIn("self.model_runner.profile_run()", patch_src)
        self.assertIn("BI100_IN_STARTUP_PROFILE", patch_src)
        self.assertIn("[BI100] skipping worker.profile_run", patch_src)
        self.assertIn("required=True", patch_src)
        self.assertIn("already_contains=", patch_src)
        self.assertNotIn("DEFAULT_NUM_GPU_BLOCKS_OVERRIDE", launch_src)
        self.assertNotIn("--num-gpu-blocks-override", launch_src)
        self.assertIn("NUM_GPU_BLOCKS_OVERRIDE", docs)
        self.assertIn("invalid for official comparison", docs)

    def test_tool_parser_patch_is_exercised(self):
        src = read("qwen3_6_scripts/patch_vllm_tool_parser.py")
        unit_src = read("tests/test_patch_tool_parser_unit.py")
        self.assertIn("Qwen3CoderToolParser", src)
        self.assertIn("required=True", src)
        self.assertIn("already_contains=", src)
        self.assertIn("test_tool_parser_patch_registers_qwen3_coder",
                      unit_src)
        self.assertIn("test_tool_parser_patch_fails_fast_when_anchor_missing",
                      unit_src)

    def test_transformers_patch_is_exercised(self):
        src = read("qwen3_6_scripts/patch_transformers_qwen3_5.py")
        unit_src = read("tests/test_patch_transformers_unit.py")
        self.assertIn("replace_one_of(AUTO_CONFIG", src)
        self.assertIn("replace_once(", src)
        self.assertIn("required=True", src)
        self.assertIn("already_contains=", src)
        self.assertNotIn("def __init__(self, **kwargs): pass", src)
        self.assertIn("test_transformers_patch_registers_qwen35_configs",
                      unit_src)
        self.assertIn("test_transformers_patch_fails_fast_when_anchor_missing",
                      unit_src)

    def test_patch_utils_are_exercised(self):
        src = read("qwen3_6_scripts/patch_utils.py")
        unit_src = read("tests/test_patch_utils_unit.py")
        for helper in [
                "package_root",
                "ensure_file",
                "ensure_dir",
                "replace_once",
                "replace_one_of",
                "shell_env_line",
        ]:
            self.assertIn(f"def {helper}", src)
            self.assertIn(helper, unit_src)
        self.assertIn("test_replace_one_of_required_missing_anchor_raises",
                      unit_src)
        self.assertIn("test_shell_env_line_is_shell_safe", unit_src)

    def test_patch_ops_prints_dynamic_roots(self):
        src = read("qwen3_6_scripts/patch_ops.sh")
        self.assertIn('echo "VLLM_ROOT=${VLLM_ROOT}"', src)
        self.assertIn('echo "TRANSFORMERS_ROOT=${TRANSFORMERS_ROOT}"', src)
        self.assertNotIn("/usr/local/corex/lib/python3/dist-packages/vllm",
                         src)
        self.assertNotIn("/usr/local/lib/python3.10/site-packages/transformers",
                         src)

    def test_launch_service_matches_submission_defaults(self):
        src = read("launch_service")
        yaml = read("computility-run.yaml")
        expected = {
            "DEFAULT_MODEL_PATH": "/model",
            "DEFAULT_SERVED_MODEL_NAME": "llm",
            "DEFAULT_MAX_MODEL_LEN": "262144",
            "DEFAULT_TENSOR_PARALLEL_SIZE": "4",
            "DEFAULT_GPU_MEMORY_UTILIZATION": "0.9",
            "DEFAULT_MAX_NUM_SEQS": "1",
            "DEFAULT_MAX_NUM_BATCHED_TOKENS": "8192",
            "DEFAULT_MAX_SEQ_LEN_TO_CAPTURE": "32768",
            "DEFAULT_TOOL_CALL_PARSER": "qwen3_coder",
            "DEFAULT_REASONING_PARSER": "qwen3",
        }
        for name, value in expected.items():
            self.assertIn(f'{name}="{value}"', src)
        for token in [
                "--model",
                "--served-model-name",
                "--max-model-len",
                "--gpu-memory-utilization",
                "--max-num-seqs",
                "--max-num-batched-tokens",
                "--enable-chunked-prefill",
                "--max-seq-len-to-capture",
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "--reasoning-parser",
                "--enable-prefix-caching",
        ]:
            self.assertIn(token, src)
            self.assertIn(token, yaml)

        for forbidden_override in [
                "MAX_MODEL_LEN:-",
                "TENSOR_PARALLEL_SIZE:-",
                "GPU_MEMORY_UTILIZATION:-",
                "MAX_NUM_SEQS:-",
                "MAX_NUM_BATCHED_TOKENS:-",
                "MAX_SEQ_LEN_TO_CAPTURE:-",
        ]:
            self.assertNotIn(forbidden_override, src)
        self.assertIn('DEFAULT_PORT="8000"', src)
        self.assertIn("export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600", src)
        self.assertIn(
            "export BI100_MOE_COREX_DIRECT_ROUTED="
            "${BI100_MOE_COREX_DIRECT_ROUTED:-1}", src)
        self.assertIn(
            "export BI100_GDN_COREX_PACKED_DECODE="
            "${BI100_GDN_COREX_PACKED_DECODE:-1}", src)

    def test_benchmark_defaults_to_evaluator_concurrency(self):
        src = read("tests/bench_perf.py")
        self.assertIn('parser.add_argument("--workers", type=int, default=1)',
                      src)

    def test_submission_contract_is_fixed(self):
        yaml = read("computility-run.yaml")
        expected_fragments = [
            "concurrency: 1",
            "    - '262144'",
            "    - '0.9'",
            "    - -tp",
            "    - '4'",
            "    - --max-num-seqs",
            "    - '1'",
            "    - --max-num-batched-tokens",
            "    - '8192'",
            "    - --max-seq-len-to-capture",
            "    - '32768'",
            "      value: 3600",
        ]
        for fragment in expected_fragments:
            self.assertIn(fragment, yaml)

    def test_launch_service_uses_full_corex_environment(self):
        src = read("launch_service")
        for path in [
                "/usr/local/corex/lib/python3/dist-packages",
                "/usr/local/corex/lib64/python3/dist-packages",
                "/usr/local/corex/lib",
                "/usr/local/corex/lib64",
                "/usr/local/corex-3.2.3/lib",
                "/usr/local/corex-3.2.3/lib64",
                "/usr/local/corex/bin",
                "/usr/local/corex-3.2.3/bin",
        ]:
            self.assertIn(path, src)
        self.assertIn("${PYTHONPATH:-}", src)
        self.assertIn("${LD_LIBRARY_PATH:-}", src)
        self.assertIn("${PATH:-}", src)

    def test_m1_32_runner_waits_for_port_release_between_services(self):
        src = read("scripts/run_m1_32_remaining_gates.sh")
        self.assertIn("wait_for_port_free", src)
        self.assertIn('sock.bind(("127.0.0.1", 8000))', src)
        self.assertIn('98 > "$output_dir/startup.rc"', src)
        self.assertIn("aligned-long", src)
        self.assertIn("--min-completion-tokens 1000", src)

    def test_m1_33_chunk64_gates_are_fail_closed(self):
        src = read("scripts/run_m1_33_chunk64_gates.sh")
        self.assertIn("runtime_preflight", src)
        self.assertIn("stale GDN runtime override", src)
        self.assertIn('gdn_restore_alignment("chunk64", 16, 8192)', src)
        self.assertIn("if [[ $runtime_rc -ne 0 ]]", src)
        self.assertIn("gdn_split_exactness.py", src)
        self.assertIn("if [[ $operator_rc -ne 0 ]]", src)
        self.assertIn("M1_32_FALLBACK_MODE=chunk64", src)
        self.assertIn("M1_32_FALLBACK_MIN_CACHED=234944", src)

    def test_m1_34_direct_suffix_probe_runs_both_fresh_services(self):
        src = read("scripts/run_m1_34_direct_suffix_probe.sh")
        self.assertIn('DEFAULT_TARGET_MODS="1 2"', src)
        self.assertIn("M1_34_TARGET_MODS", src)
        self.assertIn("M1_34_RUN_ID", src)
        self.assertIn("BI100_GDN_CACHE_POLICY=admission64", src)
        self.assertIn("BI100_GDN_RESTORE_MODE=direct", src)
        self.assertIn('--eviction-target-mod "$target_mod"', src)
        self.assertIn("stop_service", src)
        self.assertIn("classification.json", src)

    def test_direct_restore_avoids_single_token_prefill_path(self):
        policy_src = read("qwen3_6_scripts/gdn_prefix.py")
        scheduler_src = read("qwen3_6_scripts/scheduler.py")
        self.assertIn("GDN_DIRECT_MIN_REPLAY_TOKENS = 2", policy_src)
        self.assertIn("restore_key_is_eligible", scheduler_src)
        self.assertIn("step_key = final_capture_key(", scheduler_src)

    def test_m1_34_matrix_runner_requires_guard_and_fixed_contract(self):
        src = read("scripts/run_m1_34_fixed_matrix.sh")
        for required_rc in (
                "probe.rc",
                "suffix_mod1/startup.rc",
                "suffix_mod1/runtime_contract.rc",
                "suffix_mod1/pressure.rc",
                "suffix_mod1/fatal_scan.rc",
                "matrix.rc",
        ):
            self.assertIn(required_rc, src)
        self.assertIn('cold["prompt_tokens"] != 10593', src)
        self.assertIn('replay["cached_tokens"] != 10576', src)
        self.assertIn("BI100_GDN_CACHE_POLICY=admission64", src)
        self.assertIn("BI100_GDN_RESTORE_MODE=direct", src)
        self.assertIn("moe_direct=1 gdn_packed=1", src)
        self.assertIn("check_startup_capacity.py", src)
        self.assertIn("compare_dataset_shaped_policies.py", src)
        self.assertIn("qualification.rc", src)

    def test_m1_34_post_matrix_runner_enforces_direct_long_context(self):
        src = read("scripts/run_m1_34_post_matrix_gates.sh")
        self.assertIn("stage_qualified", src)
        self.assertIn("BI100_GDN_RESTORE_MODE=direct", src)
        self.assertIn("--target-prompt-tokens 131000", src)
        self.assertIn("--min-completion-tokens 256", src)
        self.assertIn("--min-cached-tokens 130992", src)
        self.assertIn("--target-prompt-tokens 235000", src)
        self.assertIn("--min-completion-tokens 1000", src)
        self.assertIn("--min-cached-tokens 234992", src)
        self.assertIn("--target-prompt-tokens 262000", src)
        self.assertIn("--min-cached-tokens 261984", src)
        self.assertIn("check_startup_capacity.py", src)
        self.assertIn("fatal_scan.rc", src)
        self.assertIn("post_matrix.rc", src)

    def test_dataset_shaped_matrix_uses_tracked_profile_driver(self):
        src = read("scripts/run_dataset_shaped_matrix.sh")
        self.assertIn(
            'PROFILE_SCRIPT="$ROOT/scripts/profile_dataset_shaped_prompt.py"',
            src,
        )
        self.assertNotIn("bench_runs/m1_32/profile_dataset_shaped_prompt.py",
                         src)
        self.assertTrue(
            (ROOT / "scripts/profile_dataset_shaped_prompt.py").is_file())

    def test_m1_33_matrix_runner_is_fail_closed(self):
        src = read("scripts/run_m1_33_fixed_matrix.sh")
        for required_rc in (
                "runtime_preflight/runtime.rc",
                "operator_exactness/operator.rc",
                "pressure.rc",
                "long_235k_exact.rc",
                "remaining_gates.rc",
                "matrix.rc",
        ):
            self.assertIn(required_rc, src)
        self.assertIn("BI100_GDN_CACHE_POLICY=admission64", src)
        self.assertIn("BI100_GDN_RESTORE_MODE=chunk64", src)
        self.assertIn("moe_direct=1 gdn_packed=1", src)
        self.assertIn("check_startup_capacity.py", src)
        self.assertIn("compare_dataset_shaped_policies.py", src)
        self.assertIn("qualification.rc", src)

    def test_m1_33_post_matrix_runner_enforces_long_capacity(self):
        src = read("scripts/run_m1_33_post_matrix_gates.sh")
        self.assertIn("stage_qualified", src)
        self.assertIn("--target-prompt-tokens 131000", src)
        self.assertIn("--min-completion-tokens 256", src)
        self.assertIn("--min-cached-tokens 130944", src)
        self.assertIn("--target-prompt-tokens 262000", src)
        self.assertIn("--min-completion-tokens 16", src)
        self.assertIn("--min-cached-tokens 261952", src)
        self.assertIn("check_startup_capacity.py", src)
        self.assertIn("fatal_scan.rc", src)
        self.assertIn("post_matrix.rc", src)

    def test_patch_scripts_do_not_keep_hardcoded_vllm_paths(self):
        for path in (ROOT / "qwen3_6_scripts").glob("patch_*.py"):
            src = path.read_text()
            self.assertNotIn(
                "/usr/local/corex/lib64/python3/dist-packages/vllm",
                src,
                path.name,
            )
            self.assertNotIn(
                "/usr/local/corex/lib/python3/dist-packages/vllm",
                src,
                path.name,
            )

    def test_xformers_patch_variants_fail_fast(self):
        for relpath in [
                "qwen3_6_scripts/patch_xformers_sdpa_batch.py",
                "qwen3_6_scripts/patch_xformers_sdpa_batch_kernel.py",
                "qwen3_6_scripts/patch_xformers_sdpa_seq.py",
                "qwen3_6_scripts/patch_xformers_sdpa_seq_kernel.py",
        ]:
            src = read(relpath)
            self.assertIn("replace_once", src, relpath)
            self.assertNotIn("anchor not found\")", src, relpath)

    def test_shell_entrypoints_are_lf_only(self):
        for relpath in [
                "qwen3_6_scripts/patch_ops.sh",
                "launch_service",
        ]:
            data = (ROOT / relpath).read_bytes()
            self.assertNotIn(b"\r\n", data, relpath)


if __name__ == "__main__":
    unittest.main()
