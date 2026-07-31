from __future__ import annotations

import copy
import hashlib
import unittest

from tests import bench_m1_104_admission64_policy_matrix as matrix
from tests import compare_m1_169_tail64_nofinal_tp1 as module


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _report(policy: str, scale: float, cached: int) -> dict:
    rows = []
    for target in matrix.SHAPES:
        for pair in matrix.PAIRS:
            for phase in matrix.PHASES:
                request_id = f"{target}_pair{pair}_{phase}"
                rows.append({
                    "request_id": request_id,
                    "target_prompt_tokens": target,
                    "pair": pair,
                    "phase": phase,
                    "salt_sha256": _digest(f"{target}:{pair}"),
                    "rendered_tokens_local": target,
                    "seed": matrix.SEED,
                    "ttft_s": scale * target / 1000,
                    "cached_tokens": cached if phase == "warm" else 0,
                    "first_token_sha256": _digest("first"),
                    "output_sha256": _digest("output"),
                    "content_sha256": _digest("content"),
                    "reasoning_sha256": _digest("reasoning"),
                    "tool_calls_sha256": matrix._sha256_json([]),
                    "finish_reason": "length",
                    "completion_tokens": matrix.MAX_TOKENS,
                })
    return {
        "schema": matrix.SCHEMA,
        "version": matrix.VERSION,
        "policy": policy,
        "qualified_measurement": True,
        "reasons": [],
        "request_count": matrix.REQUEST_COUNT,
        "request_manifest_sha256": _digest("manifest"),
        "requests": rows,
        "aggregate": {
            "ttft_p90_s": 10.0 * scale,
            "weighted": 1000.0 / scale,
            "cached_tokens": cached * 9,
        },
    }


class M1169Tail64NoFinalComparisonUnitTest(unittest.TestCase):

    def test_comparison_reports_shape_and_closed_scope(self) -> None:
        result = module.compare(
            _report("admission64", 1.0, 16000),
            _report("tail64_nofinal", 0.8, 8192),
        )
        self.assertTrue(result["qualified_analysis"])
        self.assertAlmostEqual(
            result["aggregate"]["candidate_relative_improvement"]["ttft_p90"],
            0.2,
        )
        self.assertEqual(
            result["cache_transparency"]["cross_policy_output_identity_rate"],
            1.0,
        )
        self.assertFalse(result["scope"]["production_promotion_authorized"])
        self.assertFalse(result["scope"]["tp4_evaluated"])

    def test_request_identity_difference_fails_closed(self) -> None:
        control = _report("admission64", 1.0, 16000)
        candidate = copy.deepcopy(
            _report("tail64_nofinal", 0.8, 8192))
        candidate["requests"][0]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "request identities differ"):
            module.compare(control, candidate)

    def test_cross_policy_generation_difference_is_measured_not_hidden(
            self) -> None:
        control = _report("admission64", 1.0, 16000)
        candidate = _report("tail64_nofinal", 0.8, 8192)
        candidate["requests"][0]["output_sha256"] = _digest("different")
        result = module.compare(control, candidate)
        self.assertEqual(
            result["cache_transparency"]["cross_policy_output_identity_matches"],
            matrix.REQUEST_COUNT - 1,
        )
        self.assertFalse(
            result["cache_transparency"]["cross_policy_exact_required_for_analysis"])


if __name__ == "__main__":
    unittest.main()
