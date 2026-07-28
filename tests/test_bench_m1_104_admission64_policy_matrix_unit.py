from __future__ import annotations

import hashlib
import unittest

from tests import bench_m1_104_admission64_policy_matrix as module


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def record(
    target: int,
    pair: int,
    phase: str,
    *,
    cached: int = 0,
    output: str = "same",
) -> dict:
    return {
        "request_id": f"{target}_pair{pair}_{phase}",
        "target_prompt_tokens": target,
        "pair": pair,
        "phase": phase,
        "salt_sha256": digest(f"{target}:{pair}"),
        "rendered_tokens_local": target,
        "seed": module.SEED,
        "ok": True,
        "http_status": 200,
        "done_seen": True,
        "terminal_choice_seen": True,
        "usage_seen": True,
        "data_event_count": 4,
        "malformed_sse_count": 0,
        "health_after": True,
        "prompt_tokens": target,
        "cached_tokens": cached,
        "completion_tokens": 64,
        "finish_reason": "length",
        "ttft_s": 2.0,
        "latency_s": 5.0,
        "decode_window_s": 3.0,
        "output_tps": 64.0 / 3.0,
        "first_token_sha256": digest("first"),
        "output_sha256": digest(output),
        "content_sha256": digest("content"),
        "reasoning_sha256": digest("reasoning"),
        "tool_calls_sha256": module._sha256_json([]),
        "tool_call_delta_count": 0,
    }


def complete_records() -> list[dict]:
    rows = []
    for target in module.SHAPES:
        for pair in module.PAIRS:
            cold_cached = 0 if not rows else target // 4
            rows.extend((
                record(target, pair, "cold", cached=cold_cached),
                record(
                    target,
                    pair,
                    "warm",
                    cached=target - module.TOKEN_ERROR_LIMIT,
                ),
            ))
    return rows


class M1104Admission64PolicyMatrixUnitTest(unittest.TestCase):

    def test_fixed_contract_matches_historical_matrix(self):
        self.assertEqual(module.SHAPES, (4096, 7800, 16000))
        self.assertEqual(module.PAIRS, (1, 2, 3))
        self.assertEqual(module.REQUEST_COUNT, 18)
        self.assertEqual(module.TOOL_COUNT, 29)
        self.assertEqual(module.MAX_TOKENS, 64)

    def test_tools_are_fixed_and_complete(self):
        tools = module.make_tools()
        self.assertEqual(len(tools), 29)
        self.assertEqual(tools[0]["function"]["name"], "read_file_0")
        self.assertFalse(
            tools[-1]["function"]["parameters"]["additionalProperties"])

    def test_complete_matrix_is_valid_and_aggregates(self):
        rows = complete_records()
        self.assertEqual(module.validate_requests(rows), [])
        aggregate = module.aggregate(rows)
        self.assertEqual(aggregate["success_rate"], 1.0)
        self.assertGreater(aggregate["cold_cached_tokens"], 0)
        self.assertEqual(aggregate["first_request_cached_tokens"], 0)
        self.assertGreater(aggregate["effective_hit_rate"], 0.0)
        self.assertGreater(aggregate["weighted"], 0.0)

    def test_missing_request_is_rejected(self):
        reasons = module.validate_requests(complete_records()[:-1])
        self.assertIn(
            "the fixed 18-request matrix is incomplete", reasons)

    def test_first_request_must_be_cold(self):
        rows = complete_records()
        rows[0]["cached_tokens"] = 16
        reasons = module.validate_requests(rows)
        self.assertTrue(any("first request" in reason
                            for reason in reasons))

    def test_warm_cache_cannot_fall_below_cold(self):
        rows = complete_records()
        rows[3]["cached_tokens"] = rows[2]["cached_tokens"] - 1
        reasons = module.validate_requests(rows)
        self.assertTrue(any("cache progression differs" in reason
                            for reason in reasons))

    def test_cold_warm_output_or_finish_change_is_rejected(self):
        rows = complete_records()
        rows[1]["output_sha256"] = digest("different")
        rows[1]["finish_reason"] = "stop"
        reasons = module.validate_requests(rows)
        self.assertTrue(any("output_sha256 differs" in reason
                            for reason in reasons))
        self.assertTrue(any("finish_reason differs" in reason
                            for reason in reasons))

    def test_bad_prompt_or_timing_is_rejected(self):
        rows = complete_records()
        rows[0]["prompt_tokens"] += module.TOKEN_ERROR_LIMIT
        rows[0]["output_tps"] = float("nan")
        reasons = module.validate_requests(rows)
        self.assertTrue(any("prompt token contract" in reason
                            for reason in reasons))
        self.assertTrue(any("output_tps" in reason for reason in reasons))

    def test_decode_timing_formula_is_bound_to_full_latency(self):
        rows = complete_records()
        rows[0]["latency_s"] += 1.0
        reasons = module.validate_requests(rows)
        self.assertTrue(any("decode timing formula differs" in reason
                            for reason in reasons))

    def test_terminal_sse_and_tool_choice_none_are_enforced(self):
        rows = complete_records()
        rows[0]["terminal_choice_seen"] = False
        rows[0]["malformed_sse_count"] = 1
        rows[1]["finish_reason"] = "tool_calls"
        rows[1]["tool_call_delta_count"] = 1
        reasons = module.validate_requests(rows)
        self.assertTrue(any("request or health failed" in reason
                            for reason in reasons))
        self.assertTrue(any("completion contract differs" in reason
                            for reason in reasons))

    def test_request_failure_and_bad_hash_are_rejected(self):
        rows = complete_records()
        rows[0]["ok"] = False
        rows[1]["first_token_sha256"] = "not-a-digest"
        reasons = module.validate_requests(rows)
        self.assertTrue(any("request or health failed" in reason
                            for reason in reasons))
        self.assertTrue(any("first_token_sha256" in reason
                            for reason in reasons))


if __name__ == "__main__":
    unittest.main()
