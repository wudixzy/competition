from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))
SCRIPT = ROOT / "tests/build_quality_runtime_contract.py"
SPEC = importlib.util.spec_from_file_location("quality_contract_builder", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class QualityRuntimeContractBuilderTest(unittest.TestCase):

    def build(self) -> dict:
        return MODULE.build_contract(
            source_revision="a" * 40,
            runtime_overlay_sha256="b" * 64,
            runtime_site_packages="/runtime/site-packages",
            model_path="/model",
            instance="private-tp4",
            optimization_label="baseline-fine32",
            gdn_cache_policy="fine32",
            gdn_restore_mode="direct",
            fused_prefill="0",
            kv_eviction_policy="lru",
        )

    def test_contract_is_valid_and_binds_overlay(self):
        contract = self.build()
        expected = {
            "source_revision": "a" * 40,
            "runtime_identity": "bare-host-overlay-v1:" + "b" * 20,
            "instance": "private-tp4",
            "gpu_count": 4,
            "tensor_parallel_size": 4,
            "max_model_len": 262144,
            "model_path": "/model",
            "tokenizer_path": "/model",
            "served_model_name": "llm",
        }
        digest = MODULE.runtime_contract.validate_runtime_contract(
            contract, expected, require_cache_trace=True)
        self.assertTrue(MODULE.runtime_contract.is_sha256(digest))
        self.assertEqual(
            contract["environment"]["BI100_RUNTIME_SITE_PACKAGES"],
            "/runtime/site-packages",
        )

    def test_command_matches_fixed_capacity_and_request_semantics(self):
        command = self.build()["command"]
        self.assertIn("262144", command)
        self.assertIn("qwen3_coder", command)
        self.assertIn("qwen3", command)
        self.assertNotIn("--dtype", command)
        self.assertNotIn("--quantization", command)
        self.assertNotIn("--speculative-model", command)
        self.assertNotIn("--chat-template", command)

    def test_environment_keeps_quality_sensitive_paths_enabled(self):
        environment = self.build()["environment"]
        self.assertEqual(environment["BI100_CACHE_TRACE"], "1")
        self.assertEqual(environment["BI100_GDN_ALLOW_NAN_ZERO"], "0")
        self.assertEqual(environment["BI100_GDN_FINITE_CHECK"], "0")
        self.assertEqual(environment["BI100_HYBRID_KV_ACCOUNTING"],
                         "full_attention")
        self.assertEqual(environment["BI100_MOE_COREX_EXACT_REDUCE"], "1")
        self.assertEqual(environment["BI100_ATTN_COREX_PAGED_GATHER"], "1")
        self.assertEqual(environment["BI100_PREFIX_DTYPE"], "float16")

    def test_strict_reference_profile_is_attested(self):
        contract = MODULE.build_contract(
            source_revision="a" * 40,
            runtime_overlay_sha256="b" * 64,
            runtime_site_packages="/runtime/site-packages",
            model_path="/model",
            instance="private-tp4",
            optimization_label="strict-reference",
            gdn_cache_policy="fine32",
            gdn_restore_mode="direct",
            fused_prefill="0",
            kv_eviction_policy="lru",
            kernel_profile="strict-reference",
        )
        environment = contract["environment"]
        self.assertEqual(environment["BI100_MOE_COREX_DIRECT_ROUTED"], "0")
        self.assertEqual(environment["BI100_GDN_COREX_PACKED_DECODE"], "0")
        self.assertEqual(environment["BI100_GDN_COMBINED_QK_NORM"], "0")

    def test_combined_qk_profile_is_attested(self):
        contract = MODULE.build_contract(
            source_revision="a" * 40,
            runtime_overlay_sha256="b" * 64,
            runtime_site_packages="/runtime/site-packages",
            model_path="/model",
            instance="private-tp4",
            optimization_label="strict-reference-combined-qk",
            gdn_cache_policy="fine32",
            gdn_restore_mode="direct",
            fused_prefill="0",
            kv_eviction_policy="lru",
            kernel_profile="strict-reference-combined-qk",
        )
        environment = contract["environment"]
        self.assertEqual(environment["BI100_MOE_COREX_DIRECT_ROUTED"], "0")
        self.assertEqual(environment["BI100_GDN_COREX_PACKED_DECODE"], "0")
        self.assertEqual(environment["BI100_GDN_COMBINED_QK_NORM"], "1")

    def test_current_admission64_hybrid64_candidate_is_attested(self):
        contract = MODULE.build_contract(
            source_revision="a" * 40,
            runtime_overlay_sha256="b" * 64,
            runtime_site_packages="/runtime/site-packages",
            model_path="/model",
            instance="private-tp4",
            optimization_label="m1-112-fused-prefill",
            gdn_cache_policy="admission64",
            gdn_restore_mode="hybrid64",
            fused_prefill="1",
            kv_eviction_policy="lru",
        )
        environment = contract["environment"]
        self.assertEqual(environment["BI100_GDN_CACHE_POLICY"], "admission64")
        self.assertEqual(environment["BI100_GDN_RESTORE_MODE"], "hybrid64")
        self.assertEqual(environment["BI100_ATTN_COREX_FUSED_PREFILL"], "1")
        MODULE.runtime_contract.validate_runtime_contract(
            contract,
            {
                "source_revision": "a" * 40,
                "runtime_identity": "bare-host-overlay-v1:" + "b" * 20,
                "instance": "private-tp4",
                "gpu_count": 4,
                "tensor_parallel_size": 4,
                "max_model_len": 262144,
                "model_path": "/model",
                "tokenizer_path": "/model",
                "served_model_name": "llm",
            },
            require_cache_trace=True,
        )

    def test_documented_example_matches_canonical_command_and_environment(self):
        example = json.loads((
            ROOT / "quality/runtime_contract.example.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(
            example["command"],
            MODULE.runtime_contract.service_command(example["model_path"]),
        )
        self.assertEqual(
            example["environment"],
            MODULE.runtime_contract.service_environment(
                example["environment"]["BI100_RUNTIME_SITE_PACKAGES"],
                gdn_cache_policy="fine32",
                gdn_restore_mode="direct",
                fused_prefill="0",
                kv_eviction_policy="lru",
            ),
        )


if __name__ == "__main__":
    unittest.main()
