from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

import compare_teacher_forced_logprobs as legacy
import compare_teacher_forced_logprobs_v2 as comparison


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads(
    (ROOT / "quality/layered_quality_gate.v2.json").read_text(
        encoding="utf-8")
)


def _key(value: int) -> str:
    return f"{value:064x}"


def _position(ordinal: int) -> dict:
    return {
        "position": ordinal,
        "actual_token_key": _key(1000 + ordinal),
        "top_logprobs": [
            {"token_key": _key(1000 + ordinal), "logprob": -0.10},
            {"token_key": _key(2000 + ordinal), "logprob": -0.40},
            {"token_key": _key(3000 + ordinal), "logprob": -1.00},
            {"token_key": _key(4000 + ordinal), "logprob": -2.00},
            {"token_key": _key(5000 + ordinal), "logprob": -3.00},
        ],
    }


def _report(mode: str) -> dict:
    cases = []
    offset = 0
    for target in comparison.TARGETS:
        cases.append({
            "id": f"length_{target}",
            "prompt_tokens": target,
            "positions": [_position(offset + index) for index in range(64)],
        })
        offset += 64
    return {
        "schema": legacy.OBSERVATION_SCHEMA,
        "version": 1,
        "mode": mode,
        "source_revision": "a" * 40,
        "runtime_identity": "overlay-install-44a-byte-equal",
        "instance": "private-instance",
        "model_path": "/model",
        "gpu_count": 4,
        "tensor_parallel_size": 4,
        "max_model_len": 262144,
        "top_k": 5,
        "optimization": {
            "fused_prefill": "1" if mode == "candidate" else "0",
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


class TeacherForcedV2Tests(unittest.TestCase):

    def test_identical_aa_and_candidate_pass(self) -> None:
        result = comparison.compare(
            _report("control"), _report("control"),
            _report("candidate"), CONTRACT)
        self.assertEqual(result["status"], "pass", result)
        self.assertEqual(result["sampled_positions"], 256)
        self.assertEqual(result["candidate"]["top1_agreement"], 1.0)
        self.assertEqual(result["runtime_identity"],
                         "overlay-install-44a-byte-equal")
        self.assertEqual(result["aa"]["sampled_positions"], 256)
        self.assertEqual(result["arm_binding"]["candidate"], "candidate")

    def test_high_margin_flip_requires_adjudication(self) -> None:
        candidate = _report("candidate")
        top = candidate["cases"][0]["positions"][0]["top_logprobs"]
        top[0], top[1] = top[1], top[0]
        top[0]["logprob"] = -0.11
        top[1]["logprob"] = -0.40
        result = comparison.compare(
            _report("control"), _report("control"), candidate, CONTRACT)
        self.assertEqual(result["status"], "inconclusive", result)
        self.assertEqual(
            result["classification"],
            "distribution_drift_requires_adjudication",
        )

    def test_aa_noise_calibrates_thresholds(self) -> None:
        control_b = _report("control")
        candidate = _report("candidate")
        for report in (control_b, candidate):
            for item in report["cases"][0]["positions"][0]["top_logprobs"]:
                item["logprob"] -= 0.02
        result = comparison.compare(
            _report("control"), control_b, candidate, CONTRACT)
        self.assertEqual(result["status"], "pass", result)
        self.assertAlmostEqual(
            result["decision"]["high_margin_threshold_nats"], 0.1)

    def test_incomplete_population_is_invalid(self) -> None:
        candidate = _report("candidate")
        candidate["cases"].pop()
        result = comparison.compare(
            _report("control"), _report("control"), candidate, CONTRACT)
        self.assertEqual(result["status"], "invalid")

    def test_output_strips_private_token_keys(self) -> None:
        candidate = _report("candidate")
        before = copy.deepcopy(candidate)
        result = comparison.compare(
            _report("control"), _report("control"), candidate, CONTRACT)
        serialized = json.dumps(result, sort_keys=True)
        self.assertNotIn(_key(1000), serialized)
        self.assertNotIn('"token_key":', serialized)
        self.assertEqual(candidate, before)
        self.assertTrue(all(value is False for value in result["privacy"].values()))


if __name__ == "__main__":
    unittest.main()
