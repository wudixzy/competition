from __future__ import annotations

import copy
import unittest

import qualify_attention_operator_tp4_pair as qualifier
from test_attention_operator_tp4_service_unit import report


def status(selector: str) -> dict:
    return {
        "schema": "bi100-attention-operator-tp4-arm-v1", "version": 1,
        "change_scope": "attention_operator", "qualified": True,
        "result_status": "pass", "returncode": 0,
        "terminal_stage": "complete", "source_revision": "a" * 40,
        "source_dirty_summary": "clean", "runtime_identity": "runtime-1",
        "instance": "instance-1",
        "model_path": "/root/public-storage/models/Qwen/Qwen3.6-35B-A3B",
        "selector": selector, "workload_id": "pair-1",
        "session_preflight_id": "preflight-1",
        "targets": list(qualifier.service.TARGETS),
        "repetitions": qualifier.service.REPETITIONS,
        "service_startups": 1, "gpu_count": 4, "tensor_parallel_size": 4,
        "request_population": {"expected": 9, "attempted": 9,
                               "completed": 9, "failed": 0},
        "dispatch_count": 12 if selector == "candidate" else 0,
        "gates": {"all": 0},
    }


def manifest(selector: str) -> dict:
    return {
        "schema": "bi100-attention-operator-runtime-v1", "version": 1,
        "change_scope": "attention_operator", "source_revision": "a" * 40,
        "runtime_identity": "runtime-1", "instance": "instance-1",
        "model_path": "/root/public-storage/models/Qwen/Qwen3.6-35B-A3B",
        "tokenizer_path": "/root/public-storage/models/Qwen/Qwen3.6-35B-A3B",
        "tensor_parallel_size": 4, "max_model_len": 262144,
        "block_size": 16, "command": ["launch_service"],
        "environment": {
            "BI100_GDN_CACHE_POLICY": "admission64",
            "BI100_ATTN_COREX_FUSED_PREFILL":
                "1" if selector == "candidate" else "0",
        },
    }


def inputs(candidate_ttft: float) -> tuple[dict, dict, dict]:
    return (
        {name: status(name) for name in ("control", "candidate")},
        {name: manifest(name) for name in ("control", "candidate")},
        {"control": report("control", 1.0),
         "candidate": report("candidate", candidate_ttft)},
    )


class AttentionOperatorPairTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        qualifier.BOOTSTRAP_SAMPLES = 500

    def test_over_five_percent_stable_gain_passes(self) -> None:
        result = qualifier.qualify(*inputs(0.94))
        self.assertEqual(result["status"], "pass", result)
        self.assertTrue(result["long_context_authorized"])
        self.assertEqual(result["request_population_per_arm"], 9)

    def test_two_to_five_percent_is_inconclusive(self) -> None:
        result = qualifier.qualify(*inputs(1 / 1.03))
        self.assertEqual(result["status"], "inconclusive", result)

    def test_negative_gain_is_candidate_fail_not_invalid(self) -> None:
        result = qualifier.qualify(*inputs(1.10))
        self.assertEqual(result["status"], "fail", result)
        self.assertEqual(result["classification"], "gain_below_two_percent")

    def test_nonfinite_timing_is_invalid(self) -> None:
        statuses, manifests, measurements = inputs(0.94)
        measurements["candidate"]["cases"][0]["response"]["ttft_s"] = (
            float("nan"))
        result = qualifier.qualify(statuses, manifests, measurements)
        self.assertEqual(result["status"], "fail", result)

    def test_wrong_dispatch_is_candidate_fail(self) -> None:
        statuses, manifests, measurements = inputs(0.94)
        statuses["candidate"]["dispatch_count"] = 0
        result = qualifier.qualify(statuses, manifests, measurements)
        self.assertEqual(result["status"], "fail", result)

    def test_cross_arm_workload_or_runtime_drift_is_invalid(self) -> None:
        statuses, manifests, measurements = inputs(0.94)
        manifests["candidate"]["command"].append("--different")
        result = qualifier.qualify(statuses, manifests, measurements)
        self.assertEqual(result["status"], "invalid", result)

    def test_only_control_and_candidate_feed_estimator(self) -> None:
        statuses, manifests, measurements = inputs(0.94)
        result = qualifier.qualify(statuses, manifests, measurements)
        self.assertEqual(
            result["performance"]["estimator"],
            "mean_of_nine_paired_control_over_candidate_ttft_gains")
        self.assertNotIn("control_b", str(result))


if __name__ == "__main__":
    unittest.main()
