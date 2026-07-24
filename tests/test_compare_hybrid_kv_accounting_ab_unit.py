from __future__ import annotations

import importlib.util
import hashlib
import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))
SCRIPT = TESTS / "compare_hybrid_kv_accounting_ab.py"
SPEC = importlib.util.spec_from_file_location("hybrid_ab", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _startup(mode: str, attention_layers: int, gpu: int, cpu: int) -> dict:
    contract = {
        "model_config_sha256": "c" * 64,
        "service": {
            "accounting": mode,
            "max_num_batched_tokens": "8192",
            "cpu_kv_offload": "0",
        },
        "engine": {
            "max_seq_len": 262_144,
            "block_size": 16,
            "swap_space_gib": 4.0,
            "dtype": "float16",
        },
    }
    encoded = json.dumps(
        contract, ensure_ascii=True, sort_keys=True,
        separators=(",", ":")).encode("ascii")
    invariant_contract = json.loads(encoded.decode("ascii"))
    invariant_contract["service"]["accounting"] = \
        MODULE.ACCOUNTING_MODE_PLACEHOLDER
    invariant_encoded = json.dumps(
        invariant_contract, ensure_ascii=True, sort_keys=True,
        separators=(",", ":")).encode("ascii")
    return {
        "schema": MODULE.STARTUP_SCHEMA,
        "version": MODULE.VERSION,
        "mode": mode,
        "config_mode": mode,
        "expected_attention_layers": attention_layers,
        "observed_attention_layers": attention_layers,
        "observed_layer_count": 40,
        "full_attention_ordinals": MODULE.EXPECTED_FULL_ATTENTION_ORDINALS,
        "max_model_len_required": 262_144,
        "block_size": 16,
        "required_gpu_blocks": 16_384,
        "observed_max_seq_len": 262_144,
        "observed_gpu_blocks": gpu,
        "observed_cpu_blocks": cpu,
        "observed_gpu_tokens": gpu * 16,
        "num_key_value_heads": 2,
        "rank_kv_heads": 1,
        "head_dim": 256,
        "tensor_parallel_size": 4,
        "dtype": "float16",
        "dtype_bytes": 2,
        "expected_kv_bytes_per_block": (
            attention_layers * 2 * 16 * 1 * 256 * 2),
        "runtime_accounting_reports": [{
            "tp_rank": rank,
            "env_mode": mode,
            "config_mode": mode,
            "configured_kv_layers": attention_layers,
            "full_attention_layers": 10,
            "full_attention_ordinals": (
                MODULE.EXPECTED_FULL_ATTENTION_ORDINALS),
        } for rank in range(4)],
        "runtime_contract": contract,
        "runtime_contract_sha256": hashlib.sha256(encoded).hexdigest(),
        "runtime_contract_invariant_sha256": hashlib.sha256(
            invariant_encoded).hexdigest(),
        "service_log_sha256": "a" * 64,
        "qualified": True,
        "reasons": [],
    }


def _request(name: str, expected: int, cached: int, elapsed: float) -> dict:
    return {
        "name": name,
        "status": "ok",
        "expected_prompt_tokens": expected,
        "summary": {
            "prompt_tokens": expected,
            "cached_tokens": cached,
            "completion_tokens": 8,
            "finish_reason": "length",
            "message_sha256": "b" * 64,
            "elapsed_s": elapsed,
        },
    }


def _pressure(mode: str, after: int, warm: float) -> dict:
    target = 65_536
    pressure = 135_040
    requests = [
        _request("target_cold", target, 0, 90.0),
        _request("target_immediate_warm", target, 65_520, warm),
        _request("pressure_cold_0000", pressure, 0, 240.0),
        _request("pressure_cold_0001", pressure, 0, 240.0),
        _request("target_after_pressure", target, after, 90.0),
        _request("target_refreshed", target, 65_520, 3.0),
    ]
    return {
        "schema": "bi100-cpu-kv-offload-pressure-api-v1",
        "version": 1,
        "params": {
            "run_id": "m149-fixed",
            "mode": mode,
            "json_out": f"/tmp/{mode}.json",
            "max_control_cached": 16,
            "min_candidate_cached": 65_504,
            "target_prompt_tokens": target,
            "pressure_prompt_tokens": pressure,
            "pressure_count": 2,
            "max_tokens": 8,
            "block_size": 16,
            "timeout_s": 900.0,
        },
        "requests": requests,
        "validation": {"qualified": True, "reasons": []},
        "qualified": True,
    }


def _compare(candidate_gpu: int = 67_512, candidate_warm: float = 3.03) -> dict:
    return MODULE.compare(
        _startup("legacy40", 40, 16_878, 6_553),
        _startup("full_attention", 10, candidate_gpu, 26_212),
        _pressure("control", 16, 3.0),
        _pressure("candidate", 65_520, candidate_warm),
    )


class CompareHybridKvAccountingAbTest(unittest.TestCase):

    def test_runner_isolates_layer_accounting(self):
        source = (ROOT / "scripts/run_m1_49_hybrid_kv_ab.sh").read_text(
            encoding="utf-8")
        self.assertIn("run_arm legacy40 control", source)
        self.assertIn("run_arm full_attention candidate", source)
        self.assertIn("BI100_CPU_KV_OFFLOAD=0", source)
        self.assertNotIn("BI100_CPU_KV_OFFLOAD=1", source)
        self.assertIn("BI100_ATTN_COREX_FUSED_PREFILL=0", source)
        self.assertIn("BI100_CACHE_TRACE=0", source)
        self.assertIn("export HOST=0.0.0.0", source)
        self.assertIn("export PORT=8000", source)
        self.assertIn("BI100_RUNTIME_SITE_PACKAGES", source)
        self.assertIn('setsid "$ROOT/launch_service"', source)
        self.assertIn(
            'source "$ROOT/scripts/lib/process_group.sh"', source)
        self.assertIn(
            'bi100_stop_process_group "$ACTIVE_PGID" "$ACTIVE_PID"',
            source,
        )
        self.assertIn('printf \'%s\\n\' "$cleanup_rc"', source)
        self.assertIn("run_preflight after_legacy", source)
        self.assertIn("run_preflight after_candidate", source)
        self.assertIn("compare_preflights after_legacy", source)
        self.assertIn("compare_preflights final", source)

    def test_fixed_ab_qualifies(self):
        report = _compare()
        self.assertTrue(report["qualified"], report)
        self.assertGreater(report["capacity"]["gpu_block_ratio"], 3.9)
        self.assertAlmostEqual(report["immediate_warm_ratio"], 1.01)

    def test_capacity_ratio_fails_closed(self):
        report = _compare(candidate_gpu=40_000)
        self.assertFalse(report["qualified"])
        self.assertTrue(any("observed_gpu_blocks ratio" in reason
                            for reason in report["reasons"]))

    def test_warm_regression_fails_closed(self):
        report = _compare(candidate_warm=3.07)
        self.assertFalse(report["qualified"])
        self.assertTrue(any("immediate-warm ratio" in reason
                            for reason in report["reasons"]))

    def test_layer_contract_fails_closed(self):
        candidate = _startup("full_attention", 10, 67_512, 26_212)
        candidate["observed_attention_layers"] = 40
        report = MODULE.compare(
            _startup("legacy40", 40, 16_878, 6_553),
            candidate,
            _pressure("control", 16, 3.0),
            _pressure("candidate", 65_520, 3.0),
        )
        self.assertFalse(report["qualified"])
        self.assertTrue(any("attention layers" in reason
                            for reason in report["reasons"]))

    def test_internally_inconsistent_startup_fails_closed(self):
        candidate = _startup("full_attention", 10, 67_512, 26_212)
        candidate["observed_gpu_tokens"] -= 16
        report = MODULE.compare(
            _startup("legacy40", 40, 16_878, 6_553),
            candidate,
            _pressure("control", 16, 3.0),
            _pressure("candidate", 65_520, 3.0),
        )
        self.assertFalse(report["qualified"])
        self.assertTrue(any("internally inconsistent" in reason
                            for reason in report["reasons"]))

    def test_runtime_contract_mismatch_fails_closed(self):
        candidate = _startup("full_attention", 10, 67_512, 26_212)
        candidate["runtime_contract"]["service"][
            "max_num_batched_tokens"] = "4096"
        encoded = json.dumps(
            candidate["runtime_contract"], ensure_ascii=True, sort_keys=True,
            separators=(",", ":")).encode("ascii")
        candidate["runtime_contract_sha256"] = hashlib.sha256(encoded).hexdigest()
        invariant = MODULE._canonical_runtime_contract(
            candidate["runtime_contract"])
        invariant_encoded = json.dumps(
            invariant, ensure_ascii=True, sort_keys=True,
            separators=(",", ":")).encode("ascii")
        candidate["runtime_contract_invariant_sha256"] = hashlib.sha256(
            invariant_encoded).hexdigest()
        report = MODULE.compare(
            _startup("legacy40", 40, 16_878, 6_553),
            candidate,
            _pressure("control", 16, 3.0),
            _pressure("candidate", 65_520, 3.0),
        )
        self.assertFalse(report["qualified"])
        self.assertIn(
            "runtime contract differs outside accounting mode",
            report["reasons"],
        )

    def test_startup_invariant_mismatch_fails_closed(self):
        candidate = _startup("full_attention", 10, 67_512, 26_212)
        candidate["observed_max_seq_len"] = 300_000
        report = MODULE.compare(
            _startup("legacy40", 40, 16_878, 6_553),
            candidate,
            _pressure("control", 16, 3.0),
            _pressure("candidate", 65_520, 3.0),
        )
        self.assertFalse(report["qualified"])
        self.assertIn(
            "startup invariants differ outside accounting capacity",
            report["reasons"],
        )


if __name__ == "__main__":
    unittest.main()
