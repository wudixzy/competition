import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))
MODULE_PATH = ROOT / "tests" / "agent_workload_matrix.py"
SPEC = importlib.util.spec_from_file_location("agent_workload_matrix", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class AgentWorkloadMatrixUnitTest(unittest.TestCase):

    def test_matrix_covers_agent_contracts(self):
        cases = MODULE.build_cases()
        self.assertEqual(len(cases), 11)
        self.assertEqual(
            cases["auto_terminal"]["expected_tool"], "terminal")
        self.assertEqual(
            len(cases["large_tool_schema"]["payload"]["tools"]), 92)
        self.assertGreaterEqual(
            len(cases["long_history"]["payload"]["messages"]), 42)
        self.assertIn(
            "tool", {
                message["role"]
                for message in cases["tool_result_roundtrip"]["payload"]["messages"]
            })
        for name in ("stream_forced_terminal", "stream_auto_terminal"):
            self.assertTrue(cases[name]["stream"])
            self.assertTrue(cases[name]["payload"]["stream"])
            self.assertTrue(
                cases[name]["payload"]["stream_options"]["include_usage"])

    def test_argument_parser_accepts_string_and_object(self):
        self.assertEqual(MODULE.parse_arguments('{"value": 7}'), {"value": 7})
        self.assertEqual(MODULE.parse_arguments({"value": 7}), {"value": 7})

    def test_safe_observation_retains_only_hashes_and_rules(self):
        secret = "raw-agent-output-must-not-be-retained"
        result = {
            "elapsed_s": 1.0,
            "finish_reason": "tool_calls",
            "content": secret,
            "reasoning_content": secret,
            "tool_calls": [{
                "name": "terminal",
                "arguments": {"command": secret},
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "total_tokens": 13,
                "prompt_tokens_details": {"cached_tokens": 4},
            },
        }
        observation = MODULE.safe_observation(result, {"rule": True})
        serialized = json.dumps(observation)
        self.assertNotIn(secret, serialized)
        self.assertEqual(observation["cached_tokens"], 4)
        self.assertEqual(len(observation["semantic_output_sha256"]), 64)

    def test_manifest_and_runtime_contract_are_bound(self):
        manifest, digest = MODULE.load_manifest(
            ROOT / "quality/agent_workload_matrix.v1.json")
        self.assertEqual(len(manifest["cases"]), 11)
        self.assertEqual(digest, MODULE.EXPECTED_MANIFEST_SHA256)

        runtime = MODULE.runtime_contract
        contract = {
            "schema": "bi100-quality-runtime-contract-v1",
            "version": 1,
            "source_revision": "a" * 40,
            "runtime_identity": "runtime-test",
            "runtime_overlay_sha256": "b" * 64,
            "instance": "private-instance",
            "gpu_count": 4,
            "tensor_parallel_size": 4,
            "max_model_len": 262144,
            "model_path": "/model",
            "tokenizer_path": "/model",
            "served_model_name": "llm",
            "base_image": runtime.BASE_IMAGE,
            "command": runtime.service_command("/model"),
            "environment": runtime.service_environment(
                "/runtime/site-packages",
                gdn_cache_policy="fine32",
                gdn_restore_mode="direct",
                fused_prefill="0",
                kv_eviction_policy="lru",
            ),
            "cache_trace_enabled": True,
            "optimization_label": "fine32",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "runtime.json"
            path.write_text(json.dumps(contract), encoding="utf-8")
            loaded, contract_sha = MODULE.load_runtime_contract(
                path, "a" * 40, "runtime-test", "private-instance")
        self.assertEqual(loaded, contract)
        self.assertEqual(len(contract_sha), 64)

    def test_runtime_contract_rejects_obsolete_base_image(self):
        runtime = MODULE.runtime_contract
        contract = {
            "schema": "bi100-quality-runtime-contract-v1",
            "version": 1,
            "source_revision": "a" * 40,
            "runtime_identity": "runtime-test",
            "runtime_overlay_sha256": "b" * 64,
            "instance": "private-instance",
            "gpu_count": 4,
            "tensor_parallel_size": 4,
            "max_model_len": 262144,
            "model_path": "/model",
            "tokenizer_path": "/model",
            "served_model_name": "llm",
            "base_image": "git.modelhub.org.cn:9443/obsolete:v1.2.3",
            "command": runtime.service_command("/model"),
            "environment": runtime.service_environment(
                "/runtime/site-packages",
                gdn_cache_policy="fine32",
                gdn_restore_mode="direct",
                fused_prefill="0",
                kv_eviction_policy="lru",
            ),
            "cache_trace_enabled": True,
            "optimization_label": "fine32",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "runtime.json"
            path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "base image"):
                MODULE.load_runtime_contract(
                    path, "a" * 40, "runtime-test", "private-instance")

    def test_validation_checks_tool_finish_and_multiple_system_markers(self):
        tool_case = MODULE.build_cases()["auto_terminal"]
        tool_result = {
            "finish_reason": "tool_calls",
            "content": "",
            "reasoning_content": "",
            "tool_calls": [{
                "name": "terminal", "arguments": {"command": "pwd"},
            }],
        }
        facts = MODULE.validate(tool_case, tool_result)
        self.assertTrue(facts["tool_call_valid"])

        system_case = MODULE.build_cases()["multiple_system"]
        system_result = {
            "finish_reason": "stop",
            "content": "SYSTEM_A SYSTEM_B",
            "reasoning_content": "",
            "tool_calls": [],
        }
        facts = MODULE.validate(system_case, system_result)
        self.assertTrue(facts["primary_content_rule_passed"])
        self.assertTrue(facts["secondary_content_rule_passed"])

    def test_stream_normalization_requires_complete_tool_sse(self):
        stream = {
            "chunks": 4,
            "done": 1,
            "usage_blocks": 1,
            "finish_reasons": ["tool_calls"],
            "content": "",
            "reasoning_content": "",
            "tool_calls": [{
                "name": "terminal",
                "arguments": {"command": "printf STREAM_NAMED_OK"},
            }],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "total_tokens": 25,
            },
        }
        result = MODULE.normalize_stream(200, stream, 1.0)
        self.assertEqual(result["finish_reason"], "tool_calls")
        self.assertEqual(result["tool_calls"][0]["name"], "terminal")
        bad = dict(stream, finish_reasons=["stop"])
        with self.assertRaisesRegex(AssertionError, "tool_calls"):
            MODULE.normalize_stream(200, bad, 1.0)


if __name__ == "__main__":
    unittest.main()
