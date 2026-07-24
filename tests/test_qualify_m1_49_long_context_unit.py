from __future__ import annotations

import copy
import pathlib
import sys
import unittest


TESTS = pathlib.Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import qualify_m1_49_long_context as qualifier


DIGEST = "a" * 64


def _startup() -> dict:
    report = {
        "schema": qualifier.STARTUP_SCHEMA,
        "version": 1,
        "qualified": True,
        "observed_gpu_blocks": 67_000,
        "observed_cpu_blocks": 26_000,
        "observed_gpu_tokens": 1_072_000,
        "service_log_sha256": DIGEST,
    }
    values = {
        "mode": "full_attention",
        "config_mode": "full_attention",
        "expected_attention_layers": 10,
        "observed_attention_layers": 10,
        "observed_layer_count": 40,
        "full_attention_ordinals": [3, 7, 11, 15, 19, 23, 27, 31, 35, 39],
        "num_key_value_heads": 2,
        "rank_kv_heads": 1,
        "head_dim": 256,
        "tensor_parallel_size": 4,
        "dtype": "float16",
        "dtype_bytes": 2,
        "expected_kv_bytes_per_block": 163_840,
        "max_model_len_required": 262_144,
        "block_size": 16,
        "required_gpu_blocks": 16_384,
        "observed_max_seq_len": 262_144,
        "runtime_contract": {
            "service": {"accounting": "full_attention"},
        },
        "runtime_contract_sha256": DIGEST,
        "runtime_contract_invariant_sha256": "b" * 64,
    }
    report.update(values)
    return report


def _ab(startup: dict) -> dict:
    return {
        "schema": qualifier.AB_SCHEMA,
        "version": 1,
        "qualified": True,
        "capacity": {
            "legacy_gpu_blocks": 16_878,
            "candidate_gpu_blocks": 67_000,
            "gpu_block_ratio": 3.970,
            "legacy_cpu_blocks": 6_553,
            "candidate_cpu_blocks": 26_000,
            "cpu_block_ratio": 3.967,
        },
        "startup": {"candidate": copy.deepcopy(startup)},
    }


def _preflight() -> dict:
    results = [{
        "gpu": gpu,
        "device_name": "Iluvatar BI-V100",
        "device_capability": [7, 0],
        "free": 32_000_000_000,
        "total": 34_057_748_480,
        "checksum": float(1024 ** 3),
        "ok": True,
    } for gpu in range(4)]
    return {
        "schema": qualifier.PREFLIGHT_SCHEMA,
        "version": 1,
        "qualified": True,
        "max_free_memory_drop_bytes": qualifier.MAX_FREE_MEMORY_DROP_BYTES,
        "stages": [
            {
                "label": "before_long",
                "qualified": True,
                "matmul_size": 1024,
                "timeout_s": 25.0,
                "results": copy.deepcopy(results),
            },
            {
                "label": "after_long",
                "qualified": True,
                "matmul_size": 1024,
                "timeout_s": 25.0,
                "results": copy.deepcopy(results),
            },
        ],
    }


def _smoke() -> dict:
    return {
        "mode": "quick",
        "ok": True,
        "tests": [
            {"name": name, "ok": True, "error": "", "elapsed_s": 1.0}
            for name in qualifier.EXPECTED_SMOKE_TESTS
        ],
    }


def _multimodal() -> dict:
    return {
        "schema": qualifier.MULTIMODAL_SCHEMA,
        "version": 2,
        "qualified": True,
        "checks": {
            "cold_has_no_hit": True,
            "red_cold_warm_exact": True,
            "same_image_hits": True,
            "different_image_isolated": True,
            "semantic_colors_observed": True,
        },
        "reasons": [],
        "source_sha256": DIGEST,
    }


def _long_report(contract: dict) -> dict:
    request = {
        "prompt_tokens": contract["target_prompt_tokens"],
        "cached_tokens": contract["min_cached_tokens"],
        "completion_tokens": contract["min_completion_tokens"],
        "finish_reason": "length",
        "message_sha256": DIGEST,
        "elapsed_s": 1.0,
        "raw_prompt": "must not survive",
    }
    requests = {
        "first": {
            **request,
            "cached_tokens": contract["max_first_cached_tokens"],
        },
        "second": dict(request),
    }
    if contract["equivalence_mode"] == "warm-repeat":
        requests["third"] = dict(request)
    return {
        "schema": qualifier.LONG_SCHEMA,
        "version": 1,
        "qualified": True,
        "contract": copy.deepcopy(contract),
        "reasons": [],
        "requests": requests,
        "source_sha256": DIGEST,
    }


def _inputs() -> dict:
    startup = _startup()
    return {
        "ab": _ab(startup),
        "startup": startup,
        "preflight": _preflight(),
        "smoke": _smoke(),
        "multimodal": _multimodal(),
        "long_reports": {
            name: _long_report(contract)
            for name, contract in qualifier.EXPECTED_LONG_CONTRACTS.items()
        },
        "source_sha256": {
            name: DIGEST for name in {
                "ab",
                "startup",
                "preflight",
                "smoke",
                "multimodal",
                *qualifier.EXPECTED_LONG_CONTRACTS,
            }
        },
    }


class M149LongContextQualificationTest(unittest.TestCase):

    def test_fixed_evidence_qualifies_and_drops_unknown_request_fields(self):
        report = qualifier.qualify(**_inputs())
        self.assertTrue(report["qualified"], report)
        requests = report["long_context"]["long_131k_exact"]["requests"]
        self.assertNotIn("raw_prompt", requests["first"])

    def test_startup_drift_from_ab_candidate_fails_closed(self):
        inputs = _inputs()
        inputs["startup"]["observed_attention_layers"] = 40
        report = qualifier.qualify(**inputs)
        self.assertFalse(report["qualified"])
        self.assertTrue(any(
            "observed_attention_layers" in reason
            for reason in report["reasons"]
        ))

    def test_weakened_235k_contract_fails_closed(self):
        inputs = _inputs()
        report_235k = inputs["long_reports"]["long_235k_warm_repeat"]
        report_235k["contract"]["min_completion_tokens"] = 16
        report = qualifier.qualify(**inputs)
        self.assertFalse(report["qualified"])
        self.assertIn(
            "long_235k_warm_repeat contract differs from the fixed gate",
            report["reasons"],
        )

    def test_262k_first_cache_over_fixed_bound_fails_closed(self):
        inputs = _inputs()
        requests = inputs["long_reports"]["long_262k_capacity"]["requests"]
        requests["first"]["cached_tokens"] = 33
        report = qualifier.qualify(**inputs)
        self.assertFalse(report["qualified"])
        self.assertIn(
            "long_262k_capacity/first cached tokens are invalid",
            report["reasons"],
        )

    def test_weakened_262k_first_cache_contract_fails_closed(self):
        inputs = _inputs()
        report_262k = inputs["long_reports"]["long_262k_capacity"]
        report_262k["contract"]["max_first_cached_tokens"] = 64
        report = qualifier.qualify(**inputs)
        self.assertFalse(report["qualified"])
        self.assertIn(
            "long_262k_capacity contract differs from the fixed gate",
            report["reasons"],
        )

    def test_missing_source_digest_fails_closed(self):
        inputs = _inputs()
        del inputs["source_sha256"]["smoke"]
        report = qualifier.qualify(**inputs)
        self.assertFalse(report["qualified"])
        self.assertTrue(any(
            "source digest" in reason for reason in report["reasons"]
        ))

    def test_shortened_smoke_suite_fails_closed(self):
        inputs = _inputs()
        inputs["smoke"]["tests"].pop()
        report = qualifier.qualify(**inputs)
        self.assertFalse(report["qualified"])
        self.assertIn(
            "smoke test order differs from the fixed quick suite",
            report["reasons"],
        )

    def test_missing_long_request_fails_closed(self):
        inputs = _inputs()
        del inputs["long_reports"][
            "long_235k_warm_repeat"]["requests"]["third"]
        report = qualifier.qualify(**inputs)
        self.assertFalse(report["qualified"])
        self.assertTrue(any(
            "request set differs" in reason for reason in report["reasons"]
        ))

    def test_unqualified_preflight_stage_fails_closed(self):
        inputs = _inputs()
        inputs["preflight"]["stages"][1]["qualified"] = False
        report = qualifier.qualify(**inputs)
        self.assertFalse(report["qualified"])
        self.assertTrue(any(
            "preflight stage is not qualified" in reason
            for reason in report["reasons"]
        ))

    def test_missing_multimodal_check_fails_closed(self):
        inputs = _inputs()
        del inputs["multimodal"]["checks"]["different_image_isolated"]
        report = qualifier.qualify(**inputs)
        self.assertFalse(report["qualified"])
        self.assertIn(
            "multimodal check set differs from the fixed gate",
            report["reasons"],
        )


if __name__ == "__main__":
    unittest.main()
