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
        self.assertIn("test_registry_alias_patch_installs_qwen36_aliases",
                      unit_src)
        self.assertIn("test_registry_alias_patch_fails_fast_when_anchor_missing",
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
        self.assertIn("BI100_GDN_ALLOW_NAN_ZERO", src)
        self.assertIn("raise RuntimeError(msg)", src)
        self.assertNotIn("NaN in prefill GatedDeltaNet", src)
        self.assertNotIn("NaN in decode GatedDeltaNet", src)

    def test_moe_prefill_groups_routes_once(self):
        src = read("qwen3_6_scripts/qwen3_5.py")
        self.assertIn("torch.argsort(flat_eids, stable=True)", src)
        self.assertIn("torch.bincount(", src)
        self.assertNotIn("mask = (topk_ids == eid)", src)

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
                "BI100_GDN_ALLOW_NAN_ZERO",
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

    def test_patch_ops_uses_offline_transformers_wheel_and_import_gate(self):
        src = read("qwen3_6_scripts/patch_ops.sh")
        self.assertIn("--no-index", src)
        self.assertIn("--no-deps", src)
        self.assertIn("TRANSFORMERS_REQUIRED_VERSION=\"4.55.3\"", src)
        self.assertNotIn("pypi.tuna", src)
        self.assertIn("python3 -m py_compile", src)
        self.assertIn("patched vllm imports", src)

    def test_worker_profile_override_patch_is_diagnostic_only(self):
        patch_ops = read("qwen3_6_scripts/patch_ops.sh")
        patch_src = read("qwen3_6_scripts/patch_worker_profile_override.py")
        launch_src = read("launch_service")
        docs = read("docs/ENV_KNOBS.md")
        self.assertIn("patch_worker_profile_override.py", patch_ops)
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
            "DEFAULT_MAX_MODEL_LEN": "100000",
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

    def test_benchmark_defaults_to_evaluator_concurrency(self):
        src = read("tests/bench_perf.py")
        self.assertIn('parser.add_argument("--workers", type=int, default=1)',
                      src)

    def test_submission_contract_is_fixed(self):
        yaml = read("computility-run.yaml")
        expected_fragments = [
            "concurrency: 1",
            "    - '100000'",
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
