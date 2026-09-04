from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

import qualify_m1_176_short_tp4_v2 as qualification
import short_tp4_v2_service as service


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((
    ROOT / "quality/layered_quality_gate.v2.json").read_text(encoding="utf-8"))


def _response(prompt: int, cached: int, ttft: float,
              completion: int = 8) -> dict:
    return {
        "ok": True, "elapsed_s": ttft + 0.1, "ttft_s": ttft,
        "tpot_s": 0.01 if completion > 1 else 0.0,
        "itl_s": 0.01 if completion > 1 else 0.0,
        "prompt_tokens": prompt, "cached_tokens": cached,
        "completion_tokens": completion, "finish_reason": "length",
        "input_tps": prompt / ttft, "output_tps": completion / 0.1,
        "cache_tps": cached / ttft,
        "request_throughput_rps": 1.0 / (ttft + 0.1),
        "first_output_identity": "a" * 64,
        "output_identity": "b" * 64,
    }


def _measurement(selector: str, ttft: float) -> dict:
    cold = []
    partial = []
    for target in service.TARGETS:
        for repetition in range(service.REPETITIONS):
            cold.append({
                "target_prompt_tokens": target, "repetition": repetition,
                "cold": _response(target, 0, ttft),
                "warm": _response(target, target - 16, ttft),
                "output_exact": True,
            })
            context = target - service.PARTIAL_RESIDUAL_TOKENS
            partial.append({
                "target_prompt_tokens": target,
                "block_context_tokens": context,
                "partial_residual_tokens": service.PARTIAL_RESIDUAL_TOKENS,
                "shared_tokens_before_block_rounding": context,
                "repetition": repetition,
                "primer": _response(context, 0, ttft, 1),
                "first_sibling": _response(target, 0, ttft, 1),
                "partial": _response(target, context - 16, ttft),
                "warm": _response(target, target - 16, ttft),
                "output_exact": True,
            })
    return {
        "schema": service.SCHEMA, "version": 2,
        "run_id": f"run-{selector}", "prompt_set_id": "fixed-pair",
        "selector": selector, "targets": list(service.TARGETS),
        "partial_residual_tokens": service.PARTIAL_RESIDUAL_TOKENS,
        "block_size": service.BLOCK_SIZE, "max_tokens": service.MAX_TOKENS,
        "repetitions": service.REPETITIONS, "seed": service.SEED,
        "workload_order": "cold_then_full_warm_then_partial_sequence",
        "expected_requests": 72, "completed_requests": 72,
        "elapsed_s": 100.0, "cold_cases": cold, "partial_cases": partial,
        "metrics": {"ttft_p50_s": ttft, "success_rate": 1.0,
                    "error_rate": 0.0, "slo_goodput_requests": 36,
                    "slo_total_requests": 36},
        "ttft_slo_s": {str(k): v for k, v in service.TTFT_SLO_S.items()},
        "privacy": {}, "authorization": {}, "evaluation": {},
        "qualified": True, "reasons": [],
    }


def _status(selector: str) -> dict:
    return {
        "schema": "bi100-m1-176-short-tp4-arm-runner-v2", "version": 2,
        "qualified": True, "result_status": "pass", "returncode": 0,
        "terminal_stage": "complete", "selector": selector,
        "source_revision": "a" * 40, "runtime_identity": "overlay-byte-equal",
        "instance": "private-instance", "model_path": "/model",
        "pair_id": "fixed-pair", "targets": list(service.TARGETS),
        "cache_states": list(qualification.STATES), "repetitions": 3,
        "service_startups": 1, "gpu_count": 4, "tensor_parallel_size": 4,
        "request_population": {"service_expected": 72,
                               "teacher_forced_expected": 4,
                               "total_expected": 76,
                               "total_completed": 76},
        "candidate_artifact": {"sha256": "c" * 64, "size_bytes": 1,
                               "active": selector == "candidate"},
        "dispatch_count": 20 if selector == "candidate" else 0,
        "gates": {"all": 0}, "artifacts_present": {"all": True},
    }


def _manifest(selector: str) -> dict:
    return {
        "schema": "bi100-quality-runtime-manifest-v2", "version": 2,
        "source_revision": "a" * 40, "runtime_identity": "overlay-byte-equal",
        "instance": "private-instance", "model_path": "/model",
        "tokenizer_path": "/model", "gpu_count": 4,
        "tensor_parallel_size": 4, "max_model_len": 262144,
        "served_model_name": "llm", "command": ["launch_service"],
        "environment": {"BI100_CACHE_TRACE": "1",
                        "BI100_ATTN_COREX_FUSED_PREFILL":
                        "1" if selector == "candidate" else "0"},
    }


def _distribution(status: str = "pass") -> dict:
    aa = {
        "shared_logprob_delta_p99": 0.001,
        "paired_nll_upper_ci": 0.001,
        "sampled_positions": 256,
    }
    candidate = {
        "sampled_positions": 256,
        "top1_agreement": 1.0,
        "mutual_topk_coverage": 1.0,
        "teacher_token_logprob_delta": 0.001,
        "shared_token_logprob_delta": 0.001,
        "paired_nll_difference": 0.0,
        "paired_nll_one_sided_95_upper_ci": 0.001,
        "first_divergent_token": -1,
        "baseline_top1_margin": 0.2,
        "high_margin_flips": 0,
    }
    if status == "inconclusive":
        candidate["high_margin_flips"] = 1
    decision = qualification.metrics_contract.classify_distribution(
        candidate, aa, CONTRACT)
    return {
        "schema": "bi100-teacher-forced-distribution-v2", "version": 2,
        "status": decision["status"],
        "classification": decision["classification"],
        "source_revision": "a" * 40,
        "runtime_identity": "overlay-byte-equal",
        "instance": "private-instance",
        "model_path": "/model",
        "targets": list(service.TARGETS),
        "sampled_positions": 256,
        "workload_identity": {
            "case_ids": [f"length_{target}" for target in service.TARGETS],
            "prompt_tokens": list(service.TARGETS),
            "sampled_positions_per_case": [64] * len(service.TARGETS),
        },
        "arm_binding": {"control_a": "control", "control_b": "control",
                        "candidate": "candidate"},
        "aa": aa, "candidate": candidate, "decision": decision,
    }


class ShortTp4V2QualificationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        qualification.BOOTSTRAP_SAMPLES = 500

    def _qualify(self, candidate_ttft: float,
                 distribution_status: str = "pass") -> dict:
        statuses = {name: _status(name) for name in qualification.SELECTORS}
        measurements = {
            "control_a": _measurement("control_a", 1.0),
            "control_b": _measurement("control_b", 1.0),
            "candidate": _measurement("candidate", candidate_ttft),
        }
        manifests = {name: _manifest(name)
                     for name in qualification.SELECTORS}
        return qualification.qualify(
            statuses, measurements, manifests,
            _distribution(distribution_status), CONTRACT)

    def test_over_three_percent_with_positive_ci_passes(self) -> None:
        result = self._qualify(0.95)
        self.assertEqual(result["status"], "pass", result)
        self.assertGreater(
            result["performance"]["paired_gain_one_sided_95_lower_ci"], 0)
        self.assertFalse(result["privacy"]["contains_private_output_identities"])

    def test_below_two_percent_fails(self) -> None:
        result = self._qualify(0.99)
        self.assertEqual(result["status"], "fail", result)

    def test_two_to_three_percent_is_inconclusive(self) -> None:
        result = self._qualify(1.0 / 1.025)
        self.assertEqual(result["status"], "inconclusive", result)

    def test_distribution_drift_remains_inconclusive(self) -> None:
        result = self._qualify(0.95, "inconclusive")
        self.assertEqual(result["status"], "inconclusive", result)

    def test_nonselector_environment_change_is_invalid(self) -> None:
        statuses = {name: _status(name) for name in qualification.SELECTORS}
        measurements = {name: _measurement(name, 1.0)
                        for name in qualification.SELECTORS}
        manifests = {name: _manifest(name)
                     for name in qualification.SELECTORS}
        manifests["candidate"]["environment"]["extra"] = "1"
        result = qualification.qualify(
            statuses, measurements, manifests, _distribution(), CONTRACT)
        self.assertEqual(result["status"], "invalid", result)

    def test_empty_distribution_is_invalid(self) -> None:
        statuses = {name: _status(name) for name in qualification.SELECTORS}
        measurements = {name: _measurement(name, 0.95)
                        for name in qualification.SELECTORS}
        manifests = {name: _manifest(name)
                     for name in qualification.SELECTORS}
        evidence = _distribution()
        evidence["aa"] = {}
        result = qualification.qualify(
            statuses, measurements, manifests, evidence, CONTRACT)
        self.assertEqual(result["status"], "invalid", result)

    def test_unknown_distribution_status_is_invalid(self) -> None:
        evidence = _distribution()
        evidence["status"] = "mystery"
        statuses = {name: _status(name) for name in qualification.SELECTORS}
        result = qualification.qualify(
            statuses,
            {name: _measurement(name, 0.95) for name in qualification.SELECTORS},
            {name: _manifest(name) for name in qualification.SELECTORS},
            evidence, CONTRACT)
        self.assertEqual(result["status"], "invalid", result)

    def test_unbound_candidate_distribution_is_invalid(self) -> None:
        evidence = _distribution()
        evidence["arm_binding"]["candidate"] = "control"
        statuses = {name: _status(name) for name in qualification.SELECTORS}
        result = qualification.qualify(
            statuses,
            {name: _measurement(name, 0.95) for name in qualification.SELECTORS},
            {name: _manifest(name) for name in qualification.SELECTORS},
            evidence, CONTRACT)
        self.assertEqual(result["status"], "invalid", result)

    def test_distribution_decision_mismatch_is_invalid(self) -> None:
        evidence = _distribution()
        evidence["decision"]["top1_agreement"] = 0.0
        statuses = {name: _status(name) for name in qualification.SELECTORS}
        result = qualification.qualify(
            statuses,
            {name: _measurement(name, 0.95) for name in qualification.SELECTORS},
            {name: _manifest(name) for name in qualification.SELECTORS},
            evidence, CONTRACT)
        self.assertEqual(result["status"], "invalid", result)


if __name__ == "__main__":
    unittest.main()
