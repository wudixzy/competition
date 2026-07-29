from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

import compare_teacher_forced_logprobs as comparison


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads(
    (ROOT / "quality/layered_quality_gate.v1.json").read_text(
        encoding="utf-8")
)
LENGTHS = (4096, 32768, 65536, 131072, 235000)


def _digest(value: int) -> str:
    return f"{value:064x}"


def _position(ordinal: int, *, mismatch: str | None = None) -> dict:
    top = [
        {"token_key": _digest(1000 + ordinal), "logprob": -0.10},
        {"token_key": _digest(2000 + ordinal), "logprob": -0.20},
        {"token_key": _digest(3000 + ordinal), "logprob": -1.00},
        {"token_key": _digest(4000 + ordinal), "logprob": -2.00},
        {"token_key": _digest(5000 + ordinal), "logprob": -3.00},
    ]
    if mismatch in {"low_margin_control", "low_margin_candidate"}:
        top[0]["logprob"] = -0.100
        top[1]["logprob"] = -0.101
    if mismatch == "low_margin_candidate":
        top[0]["token_key"], top[1]["token_key"] = (
            top[1]["token_key"],
            top[0]["token_key"],
        )
    elif mismatch == "high_margin":
        top[0], top[1] = top[1], top[0]
        top[0]["logprob"] = -0.11
        top[1]["logprob"] = -0.40
    return {
        "position": ordinal,
        "actual_token_key": _digest(1000 + ordinal),
        "top_logprobs": top,
    }


def _report(mode: str) -> dict:
    cases = []
    offset = 0
    for prompt_tokens in LENGTHS:
        cases.append({
            "id": f"length_{prompt_tokens}",
            "prompt_tokens": prompt_tokens,
            "positions": [
                _position(offset + index)
                for index in range(64)
            ],
        })
        offset += 64
    return {
        "schema": comparison.OBSERVATION_SCHEMA,
        "version": 1,
        "mode": mode,
        "source_revision": "a" * 40,
        "runtime_identity": "b" * 64,
        "instance": "private-instance",
        "model_path": "/model",
        "gpu_count": 4,
        "tensor_parallel_size": 4,
        "max_model_len": 262144,
        "top_k": 5,
        "optimization": {
            "fused_prefill": "0" if mode == "control" else "1",
            "gdn_cache_policy": "admission64",
            "gdn_restore_mode": "hybrid64",
            "kv_eviction_policy": "lru",
        },
        "cases": cases,
        "privacy": {
            "contains_private_hmac_token_keys": True,
            "must_remain_outside_repository": True,
            "contains_raw_prompts": False,
            "contains_raw_model_outputs": False,
            "contains_raw_token_ids": False,
            "contains_credentials": False,
        },
    }


class TeacherForcedLogprobComparisonTests(unittest.TestCase):

    def test_identical_observations_pass(self) -> None:
        result = comparison.compare(
            _report("control"),
            _report("candidate"),
            CONTRACT,
        )
        self.assertTrue(result["qualified"])
        self.assertEqual(result["sampled_positions"], 320)
        self.assertEqual(result["metrics"]["top1_agreement"], 1.0)
        self.assertFalse(result["authorization"][
            "overall_promotion_authorized"])

    def test_fresh_control_repeat_passes_same_numerical_gate(self) -> None:
        result = comparison.compare(
            _report("control"),
            _report("control"),
            CONTRACT,
            comparison_mode="control-repeat",
        )
        self.assertTrue(result["qualified"], result)
        self.assertEqual(result["comparison_mode"], "control-repeat")
        self.assertEqual(result["metrics"]["top1_agreement"], 1.0)

    def test_low_margin_mutually_covered_flip_is_diagnostic(self) -> None:
        control = _report("control")
        candidate = _report("candidate")
        control["cases"][0]["positions"][0] = _position(
            0, mismatch="low_margin_control")
        candidate["cases"][0]["positions"][0] = _position(
            0, mismatch="low_margin_candidate")
        result = comparison.compare(control, candidate, CONTRACT)
        self.assertTrue(result["qualified"])
        self.assertEqual(result["metrics"]["top1_mismatch_count"], 1)
        self.assertEqual(
            result["metrics"]["high_margin_top1_mismatches"],
            0,
        )

    def test_high_margin_flip_fails(self) -> None:
        control = _report("control")
        candidate = _report("candidate")
        candidate["cases"][0]["positions"][0] = _position(
            0, mismatch="high_margin")
        result = comparison.compare(control, candidate, CONTRACT)
        self.assertFalse(result["qualified"])
        self.assertIn(
            "one or more high-margin top-1 choices changed",
            result["reasons"],
        )

    def test_hidden_logprob_drift_fails_even_when_top1_matches(self) -> None:
        control = _report("control")
        candidate = _report("candidate")
        for item in candidate["cases"][0]["positions"][0]["top_logprobs"]:
            item["logprob"] -= 0.2
        result = comparison.compare(control, candidate, CONTRACT)
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "logprob delta" in reason or "NLL" in reason
            for reason in result["reasons"]
        ))

    def test_teacher_token_alignment_is_hard(self) -> None:
        control = _report("control")
        candidate = _report("candidate")
        candidate["cases"][0]["positions"][0][
            "actual_token_key"] = _digest(999999)
        candidate["cases"][0]["positions"][0][
            "top_logprobs"][-1]["token_key"] = _digest(999999)
        result = comparison.compare(control, candidate, CONTRACT)
        self.assertFalse(result["qualified"])
        self.assertEqual(result["status"], "invalid")
        self.assertIn(
            "length_4096: teacher token differs at sampled position",
            result["validation_reasons"],
        )

    def test_incomplete_matrix_is_invalid_not_numerical_failure(self) -> None:
        control = _report("control")
        candidate = _report("candidate")
        control["cases"].pop()
        candidate["cases"].pop()
        result = comparison.compare(control, candidate, CONTRACT)
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["reasons"], [])
        self.assertIn(
            "required prompt-token matrix is incomplete",
            result["validation_reasons"],
        )
        self.assertEqual(comparison.exit_code(result["status"]), 2)

    def test_extra_runtime_change_invalidates_pair(self) -> None:
        control = _report("control")
        candidate = _report("candidate")
        candidate["optimization"]["gdn_cache_policy"] = "off"
        result = comparison.compare(control, candidate, CONTRACT)
        self.assertEqual(result["status"], "invalid")
        self.assertTrue(result["validation_reasons"])

    def test_output_contains_no_private_token_keys(self) -> None:
        result = comparison.compare(
            _report("control"),
            _report("candidate"),
            CONTRACT,
        )
        serialized = json.dumps(result, sort_keys=True)
        self.assertNotIn(_digest(1000), serialized)
        self.assertNotIn("token_key", serialized)
        self.assertTrue(all(
            value is False for value in result["privacy"].values()
        ))

    def test_input_reports_are_not_mutated(self) -> None:
        control = _report("control")
        candidate = _report("candidate")
        original_control = copy.deepcopy(control)
        original_candidate = copy.deepcopy(candidate)
        comparison.compare(control, candidate, CONTRACT)
        self.assertEqual(control, original_control)
        self.assertEqual(candidate, original_candidate)


if __name__ == "__main__":
    unittest.main()
