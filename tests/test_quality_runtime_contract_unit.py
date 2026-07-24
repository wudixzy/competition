from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests/quality_runtime_contract.py"
SPEC = importlib.util.spec_from_file_location("runtime_contract", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def contract() -> dict:
    return {
        "schema": "bi100-quality-runtime-contract-v1",
        "version": 1,
        "source_revision": "a" * 40,
        "runtime_identity": "unit-runtime",
        "runtime_overlay_sha256": "b" * 64,
        "instance": "private-unit",
        "gpu_count": 4,
        "tensor_parallel_size": 4,
        "max_model_len": 262144,
        "model_path": "/model",
        "tokenizer_path": "/model",
        "served_model_name": "llm",
        "base_image": MODULE.BASE_IMAGE,
        "command": ["python3", "-m", "vllm.entrypoints.openai.api_server"],
        "environment": {"BI100_CACHE_TRACE": "1"},
        "cache_trace_enabled": True,
        "optimization_label": "baseline",
    }


def expected(value: dict) -> dict:
    fields = {
        "source_revision", "runtime_identity", "instance", "gpu_count",
        "tensor_parallel_size", "max_model_len", "model_path",
        "tokenizer_path", "served_model_name",
    }
    return {field: value[field] for field in fields}


class QualityRuntimeContractTest(unittest.TestCase):

    def test_valid_contract_is_canonical_and_loadable(self):
        value = contract()
        expected_digest = MODULE.sha256_json(value)
        self.assertEqual(MODULE.validate_runtime_contract(
            value, expected(value), require_cache_trace=True), expected_digest)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.json"
            path.write_text(json.dumps(value, indent=2), encoding="utf-8")
            loaded, digest = MODULE.load_runtime_contract(
                path, expected(value), require_cache_trace=True)
        self.assertEqual(loaded, value)
        self.assertEqual(digest, expected_digest)

    def test_retired_image_and_placeholders_fail(self):
        value = contract()
        value["base_image"] = (
            "git.modelhub.org.cn:9443/enginex-iluvatar/obsolete:v1.2.3")
        with self.assertRaisesRegex(MODULE.RuntimeContractError, "base image"):
            MODULE.validate_runtime_contract(
                value, expected(value), require_cache_trace=True)

        value = contract()
        value["source_revision"] = "0" * 40
        with self.assertRaisesRegex(MODULE.RuntimeContractError, "placeholder"):
            MODULE.validate_runtime_contract(
                value, expected(value), require_cache_trace=True)

    def test_trace_and_secret_contracts_fail_closed(self):
        value = contract()
        value["environment"]["BI100_CACHE_TRACE"] = "0"
        with self.assertRaisesRegex(MODULE.RuntimeContractError, "CACHE_TRACE"):
            MODULE.validate_runtime_contract(
                value, expected(value), require_cache_trace=True)

        value = contract()
        value["environment"]["MODELHUB_ACCESS_TOKEN"] = "private"
        with self.assertRaisesRegex(
                MODULE.RuntimeContractError, "secret-bearing"):
            MODULE.validate_runtime_contract(
                value, expected(value), require_cache_trace=True)

    def test_runtime_mismatch_fails(self):
        value = contract()
        mismatched = copy.deepcopy(expected(value))
        mismatched["tensor_parallel_size"] = 2
        with self.assertRaisesRegex(
                MODULE.RuntimeContractError, "tensor_parallel_size"):
            MODULE.validate_runtime_contract(
                value, mismatched, require_cache_trace=True)


if __name__ == "__main__":
    unittest.main()
