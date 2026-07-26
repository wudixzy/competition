from __future__ import annotations

import copy
import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests/compare_m1_58_block_major_ab.py"
SPEC = importlib.util.spec_from_file_location("compare_m158", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def startup(candidate: bool) -> dict:
    control_blocks = 67_512
    gpu_blocks = control_blocks - 1024 if candidate else control_blocks
    selector = "1" if candidate else "0"
    service = {
        "accounting": "full_attention",
        "cpu_kv_offload": "1",
        "block_major_cpu_kv": selector,
        "block_major_cpu_kv_trace": "0",
        "cache_trace": "0",
        "gdn_cache_policy": "admission64",
        "gdn_restore_mode": "direct",
        "fused_prefill": "0",
        "kv_eviction_policy": "lru",
        "max_model_len": "262144",
        "tensor_parallel_size": "4",
        "max_num_seqs": "1",
        "max_num_batched_tokens": "8192",
        "model_path": "/model",
    }
    capacity = []
    cache = []
    if candidate:
        capacity = [{
            "reserved_blocks": 1024,
            "reserved_bytes": 167_772_160,
            "profiled_blocks": control_blocks,
            "usable_blocks": gpu_blocks,
        } for _ in range(4)]
        cache = [{
            "device": f"cuda:{rank}",
            "gpu_blocks": gpu_blocks,
            "cpu_blocks": 26_212,
            "layers": 10,
            "block_bytes": 163_840,
            "staging_blocks": 512,
            "staging_buffers": 2,
        } for rank in range(4)]
    return {
        "schema": MODULE.STARTUP_SCHEMA,
        "version": 1,
        "qualified": True,
        "reasons": [],
        "mode": "full_attention",
        "config_mode": "full_attention",
        "expected_attention_layers": 10,
        "observed_attention_layers": 10,
        "observed_layer_count": 40,
        "dtype": "float16",
        "expected_kv_bytes_per_block": 163_840,
        "max_model_len_required": 262_144,
        "block_size": 16,
        "required_gpu_blocks": 16_384,
        "observed_max_seq_len": 262_144,
        "observed_gpu_blocks": gpu_blocks,
        "observed_cpu_blocks": 26_212,
        "block_major_cpu_kv": candidate,
        "block_major_capacity_reports": capacity,
        "block_major_cache_reports": cache,
        "runtime_contract": {
            "model_config_sha256": "a" * 64,
            "service": service,
            "engine": {
                "max_seq_len": 262_144,
                "block_size": 16,
                "swap_space_gib": 4.0,
                "dtype": "float16",
            },
        },
    }


def pressure(candidate: bool) -> dict:
    names = MODULE._request_sequence()
    requests = []
    for index, name in enumerate(names):
        elapsed = 10.0
        if name == "target_after_pressure":
            elapsed = 5.0 if candidate else 10.0
        cached = (
            0 if name == "target_cold" or name.startswith("pressure_cold_")
            else MODULE.TARGET_TOKENS)
        prompt_tokens = (
            MODULE.PRESSURE_TOKENS
            if name.startswith("pressure_cold_")
            else MODULE.TARGET_TOKENS)
        requests.append({
            "name": name,
            "status": "ok",
            "expected_prompt_tokens": prompt_tokens,
            "summary": {
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached,
                "completion_tokens": 8,
                "finish_reason": "length",
                "message_sha256": f"{index + 1:064x}",
                "elapsed_s": elapsed,
            },
        })
    return {
        "schema": MODULE.PRESSURE_SCHEMA,
        "version": 1,
        "qualified": True,
        "params": {
            "base": "http://127.0.0.1:8000",
            "model_path": "/model",
            "target_prompt_tokens": MODULE.TARGET_TOKENS,
            "pressure_prompt_tokens": MODULE.PRESSURE_TOKENS,
            "pressure_count": MODULE.PRESSURE_COUNT,
            "max_tokens": MODULE.MAX_TOKENS,
            "timeout_s": 900.0,
            "run_id": MODULE.EXPECTED_RUN_ID,
            "mode": "candidate",
            "block_size": MODULE.BLOCK_SIZE,
            "min_candidate_cached": MODULE.MIN_CACHED_TOKENS,
            "max_control_cached": 16,
            "json_out": "candidate.json" if candidate else "control.json",
        },
        "requests": requests,
        "validation": {
            "qualified": True,
            "reasons": [],
        },
    }


def runtime() -> dict:
    return {
        "schema": MODULE.RUNTIME_SCHEMA,
        "version": 1,
        "qualified": True,
        "reasons": [],
        "source_revision": "b" * 40,
        "runtime_tree_sha256": "c" * 64,
    }


def preflight() -> dict:
    return {
        "schema": MODULE.PREFLIGHT_SCHEMA,
        "version": 1,
        "qualified": True,
        "reasons": [],
        "stages": [
            {"label": "before_control"},
            {"label": "after_control"},
            {"label": "after_candidate"},
        ],
    }


def compare() -> dict:
    return MODULE.compare(
        startup(False),
        startup(True),
        pressure(False),
        pressure(True),
        runtime(),
        preflight(),
    )


class CompareM158BlockMajorAbTest(unittest.TestCase):

    def test_fixed_exact_ab_qualifies(self):
        report = compare()
        self.assertTrue(report["qualified"], report)
        self.assertEqual(report["capacity"]["gpu_block_delta"], 1024)
        self.assertEqual(report["performance"]["restore_speedup"], 2.0)
        self.assertEqual(report["performance"]["cold_elapsed_ratio"], 1.0)
        self.assertEqual(
            report["performance"]["gpu_warm_elapsed_ratio"], 1.0)

    def test_output_or_cache_drift_fails(self):
        candidate = pressure(True)
        candidate["requests"][-2]["summary"]["message_sha256"] = "f" * 64
        candidate["requests"][-2]["summary"]["cached_tokens"] -= 16
        report = MODULE.compare(
            startup(False), startup(True), pressure(False), candidate,
            runtime(), preflight())
        self.assertFalse(report["qualified"])
        self.assertTrue(any("message_sha256 differs" in reason
                            for reason in report["reasons"]))
        self.assertTrue(any("cached_tokens differs" in reason
                            for reason in report["reasons"]))

    def test_restore_speedup_and_nontransfer_regression_fail_closed(self):
        candidate = pressure(True)
        candidate["requests"][-2]["summary"]["elapsed_s"] = 9.0
        candidate["requests"][0]["summary"]["elapsed_s"] = 20.0
        report = MODULE.compare(
            startup(False), startup(True), pressure(False), candidate,
            runtime(), preflight())
        self.assertFalse(report["qualified"])
        self.assertTrue(any("speedup is below" in reason
                            for reason in report["reasons"]))
        self.assertTrue(any("cold path regressed" in reason
                            for reason in report["reasons"]))

    def test_gpu_warm_regression_fails_closed(self):
        candidate = pressure(True)
        candidate["requests"][1]["summary"]["elapsed_s"] = 20.0
        report = MODULE.compare(
            startup(False), startup(True), pressure(False), candidate,
            runtime(), preflight())
        self.assertFalse(report["qualified"])
        self.assertTrue(any("GPU-warm path regressed" in reason
                            for reason in report["reasons"]))

    def test_capacity_or_selector_drift_fails(self):
        candidate_startup = startup(True)
        candidate_startup["observed_gpu_blocks"] += 100
        candidate_startup["runtime_contract"]["service"][
            "block_major_cpu_kv"] = "0"
        report = MODULE.compare(
            startup(False), candidate_startup,
            pressure(False), pressure(True), runtime(), preflight())
        self.assertFalse(report["qualified"])
        self.assertTrue(any("block_major_cpu_kv" in reason
                            for reason in report["reasons"]))
        self.assertTrue(any("1024-block reserve" in reason
                            for reason in report["reasons"]))

    def test_workload_or_lifecycle_drift_fails(self):
        candidate_pressure = pressure(True)
        candidate_pressure["params"]["pressure_count"] = 8
        bad_preflight = copy.deepcopy(preflight())
        bad_preflight["stages"].pop()
        report = MODULE.compare(
            startup(False), startup(True), pressure(False),
            candidate_pressure, runtime(), bad_preflight)
        self.assertFalse(report["qualified"])
        self.assertTrue(any("pressure_count" in reason
                            for reason in report["reasons"]))
        self.assertTrue(any("preflight stages" in reason
                            for reason in report["reasons"]))


if __name__ == "__main__":
    unittest.main()
