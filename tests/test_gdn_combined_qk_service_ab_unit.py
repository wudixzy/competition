from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

API_SPEC = importlib.util.spec_from_file_location(
    "gdn_combined_qk_decode_api",
    TESTS / "gdn_combined_qk_decode_api.py",
)
API = importlib.util.module_from_spec(API_SPEC)
assert API_SPEC.loader is not None
API_SPEC.loader.exec_module(API)

COMPARE_SPEC = importlib.util.spec_from_file_location(
    "compare_gdn_combined_qk_service_ab",
    TESTS / "compare_gdn_combined_qk_service_ab.py",
)
COMPARE = importlib.util.module_from_spec(COMPARE_SPEC)
assert COMPARE_SPEC.loader is not None
COMPARE_SPEC.loader.exec_module(COMPARE)


def contract(profile: str) -> dict:
    runtime = COMPARE.runtime_contract
    return {
        "schema": "bi100-quality-runtime-contract-v1",
        "version": 1,
        "source_revision": "a" * 40,
        "runtime_identity": "unit-runtime",
        "runtime_overlay_sha256": "b" * 64,
        "instance": "private-tp4",
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
            kernel_profile=profile,
        ),
        "cache_trace_enabled": True,
        "optimization_label": profile,
    }


def row(index: int, output_tps: float) -> dict:
    return {
        "index": index,
        "ok": True,
        "http_status": 200,
        "elapsed_s": 60.0,
        "ttft_s": 0.5,
        "decode_s": 59.5,
        "output_tps": output_tps,
        "prompt_tokens": 32,
        "cached_tokens": 16,
        "completion_tokens": 1000,
        "finish_reason": "length",
        "content_chars": 4000,
        "reasoning_chars": 0,
        "tool_call_fragments": 0,
        "first_output_sha256": "c" * 64,
        "semantic_output_sha256": "d" * 64,
        "error_type": "",
        "error_sha256": None,
    }


def report(profile: str, rates: tuple[float, ...]) -> dict:
    value = contract(profile)
    rows = [row(index, rate) for index, rate in enumerate(rates)]
    return {
        "schema": API.SCHEMA,
        "version": API.VERSION,
        "qualified": True,
        "production_promotion_authorized": False,
        "label": profile,
        "runtime": {
            "source_revision": value["source_revision"],
            "runtime_identity": value["runtime_identity"],
            "runtime_overlay_sha256": value["runtime_overlay_sha256"],
            "instance": value["instance"],
            "gpu_count": 4,
            "tensor_parallel_size": 4,
            "max_model_len": 262144,
        },
        "runtime_contract": {
            "sha256": COMPARE.runtime_contract.sha256_json(value),
            "contract": value,
        },
        "config": {
            **COMPARE.EXPECTED_CONFIG,
            "timeout_s": 1200.0,
            "prompt_sha256": hashlib.sha256(
                API.PROMPT.encode("ascii")).hexdigest(),
        },
        "privacy": {
            "contains_raw_requests": False,
            "contains_raw_model_outputs": False,
            "contains_credentials": False,
        },
        "warmup": [row(0, rates[0])],
        "requests": rows,
        "summary": API.summarize(rows, COMPARE.EXPECTED_CONFIG["tokens"]),
    }


class GdnCombinedQkServiceAbTest(unittest.TestCase):

    def test_decode_summary_requires_exact_complete_rows(self):
        rows = [row(0, 10.0), row(1, 11.0), row(2, 12.0)]
        summary = API.summarize(rows, 1000)
        self.assertEqual(summary["successful_requests"], 3)
        self.assertTrue(summary["repeated_output_exact"])
        self.assertGreater(summary["output_tps_p10"], 10.0)

        rows[1]["semantic_output_sha256"] = "e" * 64
        self.assertFalse(API.summarize(
            rows, 1000)["repeated_output_exact"])

    def test_exact_candidate_with_measurable_gain_qualifies(self):
        control = report("strict-reference", (10.0, 11.0, 12.0))
        candidate = report(
            "strict-reference-combined-qk", (10.2, 11.22, 12.24))
        result = COMPARE.compare(control, candidate)
        self.assertTrue(result["qualified"], result)
        self.assertTrue(result["model_output_non_regression_authorized"])
        self.assertFalse(result["production_promotion_authorized"])
        self.assertFalse(result["final_output_tps_gate_passed"])

    def test_output_change_rejects_candidate(self):
        control = report("strict-reference", (10.0, 11.0, 12.0))
        candidate = report(
            "strict-reference-combined-qk", (10.2, 11.22, 12.24))
        candidate["requests"][1]["semantic_output_sha256"] = "e" * 64
        result = COMPARE.compare(control, candidate)
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "semantic_output_sha256 differs" in reason
            for reason in result["reasons"]))

    def test_rejected_kernel_delta_cannot_hide_in_candidate(self):
        control = report("strict-reference", (10.0, 11.0, 12.0))
        candidate = report(
            "strict-reference-combined-qk", (10.2, 11.22, 12.24))
        candidate = copy.deepcopy(candidate)
        candidate["runtime_contract"]["contract"]["environment"][
            "BI100_GDN_COREX_PACKED_DECODE"] = "1"
        candidate["runtime_contract"]["sha256"] = (
            COMPARE.runtime_contract.sha256_json(
                candidate["runtime_contract"]["contract"]))
        result = COMPARE.compare(control, candidate)
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "environment delta" in reason or "kernel profile" in reason
            for reason in result["reasons"]))

    def test_speed_regression_rejects_candidate(self):
        control = report("strict-reference", (10.0, 11.0, 12.0))
        candidate = report(
            "strict-reference-combined-qk", (9.0, 9.9, 10.8))
        result = COMPARE.compare(control, candidate)
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "speedup" in reason or "P10 ratio" in reason
            for reason in result["reasons"]))


if __name__ == "__main__":
    unittest.main()
