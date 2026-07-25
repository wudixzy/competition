from __future__ import annotations

import ast
import base64
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))
SCRIPT = ROOT / "tests/long_context_quality_api.py"
SPEC = importlib.util.spec_from_file_location("long_quality", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    chat_template = "fake {{ enable_thinking }}"

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(self, token_ids, **kwargs):
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(self, messages, **kwargs):
        parts = ["<chat>"]
        for message in messages:
            parts.append(message["role"])
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(json.dumps(item, sort_keys=True) for item in content)
            parts.append(json.dumps(
                message.get("tool_calls") or [], sort_keys=True))
        parts.append(json.dumps(kwargs.get("tools") or [], sort_keys=True))
        thinking = kwargs.get("enable_thinking")
        if thinking is None:
            thinking = (kwargs.get("chat_template_kwargs") or {}).get(
                "enable_thinking")
        parts.append("<think>" if thinking else "<no-think>")
        return [ord(character) for character in "".join(parts)]


class FakeClient:
    def __init__(self, content: str):
        self.content = content

    def post(self, payload, timeout=1):
        prompt_tokens = payload["_test_prompt_tokens"]
        return 200, {
            "id": "chatcmpl-matrix-test",
            "object": "chat.completion",
            "created": 1,
            "model": "llm",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": self.content,
                    "tool_calls": [],
                },
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 2,
                "total_tokens": prompt_tokens + 2,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }


class InvalidToolClient:

    def post(self, payload, timeout=1):
        prompt_tokens = payload["_test_prompt_tokens"]
        return 200, {
            "id": "chatcmpl-matrix-invalid-tool",
            "object": "chat.completion",
            "created": 1,
            "model": "llm",
            "choices": [{
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "",
                    "tool_calls": [{
                        "id": "call-invalid",
                        "type": "function",
                        "function": {
                            "name": "private_tool_name",
                            "arguments": "private-not-json-value",
                        },
                    }],
                },
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 2,
                "total_tokens": prompt_tokens + 2,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }


class LongContextQualityApiTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.manifest_sha = MODULE._load_manifest(
            ROOT / "quality/long_context_matrix.v5.json")

    def test_handlers_and_tiers_match_frozen_matrix(self):
        self.assertEqual(
            set(MODULE.HANDLERS),
            {case["id"] for case in self.manifest["cases"]},
        )
        self.assertEqual(len(MODULE._selected_cases(
            self.manifest, "quick", [])), 2)
        self.assertEqual(len(MODULE._selected_cases(
            self.manifest, "full", [])), 7)
        self.assertEqual(len(MODULE._selected_cases(
            self.manifest, "extended", [])), 12)

    def test_main_does_not_shadow_runtime_contract_module(self):
        tree = ast.parse(
            SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
        main = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main")
        assigned_names = {
            node.id for node in ast.walk(main)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        self.assertNotIn("runtime_contract", assigned_names)

    def test_exact_recipe_fitting_is_deterministic(self):
        tokenizer = FakeTokenizer()
        recipe = MODULE._recall_recipe("unit", "EXPECTED")
        first, first_tools, first_evidence = MODULE._fit_recipe(
            tokenizer, 512, recipe, namespace="unit")
        second, second_tools, second_evidence = MODULE._fit_recipe(
            tokenizer, 512, recipe, namespace="unit")
        self.assertEqual(first, second)
        self.assertIsNone(first_tools)
        self.assertIsNone(second_tools)
        self.assertEqual(first_evidence, second_evidence)
        self.assertEqual(
            MODULE.exact_prompt.chat_template_token_count(tokenizer, first),
            512,
        )

    def test_large_tool_schema_has_92_unique_names(self):
        tools = MODULE._large_tools(
            "target_tool", target_arguments=("key",))
        names = [tool["function"]["name"] for tool in tools]
        self.assertEqual(len(tools), 92)
        self.assertEqual(len(set(names)), 92)
        self.assertEqual(names.count("target_tool"), 1)
        target = next(
            tool["function"] for tool in tools
            if tool["function"]["name"] == "target_tool")
        self.assertEqual(
            target["parameters"]["properties"],
            {"key": {"type": "string"}},
        )
        self.assertEqual(target["parameters"]["required"], ["key"])

        pair_tools = MODULE._large_tools("target_tool")
        pair_target = next(
            tool["function"] for tool in pair_tools
            if tool["function"]["name"] == "target_tool")
        self.assertEqual(
            set(pair_target["parameters"]["properties"]),
            {"key", "ordinal"},
        )
        self.assertEqual(
            pair_target["parameters"]["required"], ["key", "ordinal"])

    def test_large_tool_schema_rejects_invalid_target_arguments(self):
        with self.assertRaisesRegex(
                MODULE.MatrixFailure, "target tool arguments are invalid"):
            MODULE._large_tools(
                "target_tool", target_arguments=("key", "key"))

    @staticmethod
    def _tool_response(arguments, content=""):
        return {
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [{
                        "id": "call-unit",
                        "type": "function",
                        "function": {
                            "name": "lookup_quality_marker",
                            "arguments": json.dumps(arguments),
                        },
                    }],
                },
            }],
        }

    def test_tool_argument_diagnostic_reports_keys_without_values(self):
        response = self._tool_response({"key": "secret", "ordinal": 7})
        with self.assertRaisesRegex(
                MODULE.MatrixFailure,
                "expected=key;actual=key,ordinal") as raised:
            MODULE._require_single_tool_call(
                response, "lookup_quality_marker", {"key": "expected"})
        self.assertNotIn("secret", str(raised.exception))
        self.assertNotIn("expected\"", str(raised.exception))

    def test_tool_argument_diagnostic_reports_mismatched_field_only(self):
        response = self._tool_response({"key": "secret"})
        with self.assertRaisesRegex(
                MODULE.MatrixFailure,
                "values differ for fields: key") as raised:
            MODULE._require_single_tool_call(
                response, "lookup_quality_marker", {"key": "expected"})
        self.assertNotIn("secret", str(raised.exception))
        self.assertNotIn("expected", str(raised.exception))

    def test_tool_content_is_strict_by_default_and_optional_for_auto(self):
        response = self._tool_response(
            {"key": "expected"}, content="protocol-valid preamble")
        with self.assertRaisesRegex(
                MODULE.MatrixFailure, "unexpected content"):
            MODULE._require_single_tool_call(
                response, "lookup_quality_marker", {"key": "expected"})
        MODULE._require_single_tool_call(
            response,
            "lookup_quality_marker",
            {"key": "expected"},
            allow_content=True,
        )

    def test_tool_content_must_remain_string_or_null(self):
        response = self._tool_response({"key": "expected"}, content={})
        with self.assertRaisesRegex(
                MODULE.MatrixFailure, "string or null"):
            MODULE._require_single_tool_call(
                response,
                "lookup_quality_marker",
                {"key": "expected"},
                allow_content=True,
            )

    def test_post_summary_does_not_retain_model_output(self):
        secret = "raw-long-context-output-must-not-be-retained"
        context = MODULE.Context(
            FakeClient(secret), FakeTokenizer(), 1,
            served_model_name="llm", template_kwargs_mode="direct")
        payload = {
            "_test_prompt_tokens": 10,
            "model": "llm",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 4,
        }
        data, summary = MODULE._post(context, payload, 10)
        self.assertEqual(data["choices"][0]["message"]["content"], secret)
        self.assertNotIn(secret, json.dumps(summary))
        self.assertEqual(len(summary["semantic_output_sha256"]), 64)

    def test_failed_case_retains_privacy_safe_request_summaries(self):
        context = MODULE.Context(
            client=None,
            tokenizer=None,
            timeout_s=1,
            served_model_name="llm",
            template_kwargs_mode="direct",
        )
        context.begin_case()
        source = {
            "status": 200,
            "semantic_output_sha256": "a" * 64,
            "elapsed_s": 1.25,
        }
        context.record_request(source)
        source["status"] = 500

        observation = context.failure_observation()

        self.assertEqual(observation["requests"], [{
            "status": 200,
            "semantic_output_sha256": "a" * 64,
            "elapsed_s": 1.25,
        }])
        self.assertEqual(
            observation["facts"]
            ["privacy_safe_requests_captured_before_failure"], 1)
        self.assertEqual(observation["construction"], [])

    def test_reasoning_failure_facts_are_boolean_and_reset_per_case(self):
        context = MODULE.Context(
            client=None,
            tokenizer=None,
            timeout_s=1,
            served_model_name="llm",
            template_kwargs_mode="direct",
        )
        context.begin_case()
        context.record_failure_facts({
            "content_exact_expected": False,
            "content_contains_expected": True,
        })
        facts = context.failure_observation()["facts"]
        self.assertFalse(facts["content_exact_expected"])
        self.assertTrue(facts["content_contains_expected"])

        context.begin_case()
        self.assertEqual(
            context.failure_observation()["facts"],
            {"privacy_safe_requests_captured_before_failure": 0},
        )

    def test_failure_facts_reject_values_and_unknown_keys(self):
        context = MODULE.Context(
            client=None,
            tokenizer=None,
            timeout_s=1,
            served_model_name="llm",
            template_kwargs_mode="direct",
        )
        with self.assertRaisesRegex(
                MODULE.MatrixFailure, "diagnostic facts are invalid"):
            context.record_failure_facts({
                "content_exact_expected": "raw-output",
            })
        with self.assertRaisesRegex(
                MODULE.MatrixFailure, "diagnostic facts are invalid"):
            context.record_failure_facts({"raw_model_output": False})

    def test_reasoning_rule_diagnostics_do_not_retain_output(self):
        expected = (
            "BEGIN-MARKER-731|MIDDLE-MARKER-552|END-MARKER-947|323")
        private_prefix = "private-output-before "
        diagnostics = MODULE._reasoning_rule_diagnostics(
            private_prefix + expected,
            "BEGIN-MARKER-731 MIDDLE-MARKER-552 END-MARKER-947 323",
            expected,
        )
        self.assertFalse(diagnostics["content_exact_expected"])
        self.assertTrue(diagnostics["content_contains_expected"])
        self.assertTrue(diagnostics["content_expected_single_occurrence"])
        self.assertTrue(diagnostics["content_expected_suffix"])
        self.assertTrue(diagnostics["content_markers_in_order"])
        self.assertTrue(diagnostics["content_arithmetic_present"])
        self.assertTrue(diagnostics["reasoning_markers_in_order"])
        self.assertTrue(diagnostics["reasoning_arithmetic_present"])
        self.assertTrue(all(
            isinstance(value, bool) for value in diagnostics.values()))
        serialized = json.dumps(diagnostics, sort_keys=True)
        self.assertNotIn(expected, serialized)
        self.assertNotIn(private_prefix, serialized)

    def test_reasoning_semantic_rule_allows_only_one_final_suffix(self):
        expected = (
            "BEGIN-MARKER-731|MIDDLE-MARKER-552|END-MARKER-947|323")
        valid = MODULE._reasoning_rule_diagnostics(
            "brief conclusion\n" + expected, "reasoning", expected)
        duplicate = MODULE._reasoning_rule_diagnostics(
            expected + "\n" + expected, "reasoning", expected)
        trailing = MODULE._reasoning_rule_diagnostics(
            expected + "\ntrailing text", "reasoning", expected)

        self.assertTrue(MODULE._reasoning_semantic_rule_passed(valid))
        self.assertFalse(MODULE._reasoning_semantic_rule_passed(duplicate))
        self.assertFalse(MODULE._reasoning_semantic_rule_passed(trailing))

    def test_invalid_tool_arguments_retain_only_structural_diagnostics(self):
        context = MODULE.Context(
            InvalidToolClient(), FakeTokenizer(), 1,
            served_model_name="llm", template_kwargs_mode="direct")
        context.begin_case()
        payload = {
            "_test_prompt_tokens": 10,
            "model": "llm",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 4,
        }

        with self.assertRaisesRegex(
                MODULE.quality.CaseFailure,
                "tool arguments are not valid JSON"):
            MODULE._post(context, payload, 10)

        observation = context.failure_observation()
        self.assertEqual(len(observation["requests"]), 1)
        summary = observation["requests"][0]
        structure = summary["tool_call_structure"]["calls"][0]
        self.assertEqual(structure["json_type"], "invalid")
        self.assertEqual(structure["arguments_length"], 22)
        self.assertFalse(summary["protocol_validated"])
        serialized = json.dumps(observation, sort_keys=True)
        self.assertNotIn("private-not-json-value", serialized)
        self.assertNotIn("private_tool_name", serialized)

    def test_matrix_file_hash_cannot_be_overridden(self):
        value = json.loads((
            ROOT / "quality/long_context_matrix.v5.json"
        ).read_text(encoding="utf-8"))
        value["cases"][0]["max_tokens"] = 32
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(
                    MODULE.MatrixFailure, "file identity is invalid"):
                MODULE._load_manifest(path)

    def test_cache_trace_proof_is_parsed_without_raw_tokens(self):
        encoded = base64.b64encode(b"a" * 32 + b"b" * 32).decode("ascii")
        record = {
            "version": 4,
            "trace_session_sha256": "session",
            "hash_encoding": "sha256_base64",
            "block_size": 16,
            "prompt_tokens": 32,
            "observed_effective_cached_tokens": 0,
            "block_hashes": encoded,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "service.log"
            path.write_text("".join(
                "[BI100_CACHE_TRACE] " + json.dumps(record) + "\n"
                for _ in range(3)
            ), encoding="utf-8")
            records = MODULE._cache_trace_records(path, 0, 3)
        self.assertEqual(len(records), 3)
        self.assertEqual(
            MODULE._prompt_trace_hashes(records[0], 0),
            [b"a" * 32, b"b" * 32],
        )

    def test_cache_trace_accounting_mismatch_fails(self):
        record = {
            "version": 4,
            "hash_encoding": "sha256_base64",
            "block_size": 16,
            "prompt_tokens": 16,
            "observed_effective_cached_tokens": 16,
            "block_hashes": base64.b64encode(b"a" * 32).decode("ascii"),
        }
        with self.assertRaisesRegex(
                MODULE.MatrixFailure, "API cached_tokens differ"):
            MODULE._prompt_trace_hashes(record, 0)

    def _runtime_args(self):
        return SimpleNamespace(
            source_revision="a" * 40,
            runtime_identity="unit-overlay",
            instance="unit-tp4",
            gpu_count=4,
            tensor_parallel_size=4,
            max_model_len=262144,
            model_path="/model",
            served_model_name="llm",
        )

    def _runtime_contract(self):
        args = self._runtime_args()
        return {
            "schema": "bi100-quality-runtime-contract-v1",
            "version": 1,
            "source_revision": args.source_revision,
            "runtime_identity": args.runtime_identity,
            "runtime_overlay_sha256": "b" * 64,
            "instance": args.instance,
            "gpu_count": args.gpu_count,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "model_path": args.model_path,
            "tokenizer_path": args.model_path,
            "served_model_name": args.served_model_name,
            "base_image": MODULE.BASE_IMAGE,
            "command": MODULE.runtime_contract.service_command(
                args.model_path),
            "environment": MODULE.runtime_contract.service_environment(
                "/runtime/site-packages",
                gdn_cache_policy="fine32",
                gdn_restore_mode="direct",
                fused_prefill="0",
                kv_eviction_policy="lru",
            ),
            "cache_trace_enabled": True,
            "optimization_label": "baseline",
        }

    def test_runtime_contract_requires_new_base_image_and_no_secrets(self):
        contract = self._runtime_contract()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.json"
            path.write_text(json.dumps(contract), encoding="utf-8")
            loaded, digest = MODULE._load_runtime_contract(
                path, self._runtime_args())
            self.assertEqual(loaded, contract)
            self.assertEqual(len(digest), 64)

            contract["base_image"] = (
                "git.modelhub.org.cn:9443/enginex-iluvatar/obsolete:v1.2.3")
            path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(MODULE.MatrixFailure, "base image"):
                MODULE._load_runtime_contract(path, self._runtime_args())

            contract = self._runtime_contract()
            contract["environment"]["MODELHUB_ACCESS_TOKEN"] = "secret"
            path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(
                    MODULE.MatrixFailure, "secret-bearing"):
                MODULE._load_runtime_contract(path, self._runtime_args())


if __name__ == "__main__":
    unittest.main()
