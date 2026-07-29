from __future__ import annotations

import unittest

import teacher_forced_topk_api as api


class TeacherForcedTopkApiTests(unittest.TestCase):

    def test_fixed_sampler_is_stable_and_in_range(self) -> None:
        for token_count in api.TARGETS:
            first = api.sample_positions(token_count, 64)
            second = api.sample_positions(token_count, 64)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 64)
            self.assertEqual(len(set(first)), 64)
            self.assertTrue(all(
                1 <= position < token_count for position in first
            ))

    def test_sampler_covers_chunk_boundaries_for_long_prompts(self) -> None:
        positions = set(api.sample_positions(235000, 64))
        self.assertTrue(any(
            abs(position - 8192) <= 1 for position in positions
        ))
        self.assertTrue(any(
            abs(position - 229376) <= 1 for position in positions
        ))

    def test_position_summary_keeps_topk_plus_teacher_only(self) -> None:
        raw = {
            str(token_id): {
                "logprob": -float(rank),
                "rank": rank,
                "decoded_token": "private",
            }
            for rank, token_id in enumerate(range(100, 107), start=1)
        }
        raw["999"] = {
            "logprob": -20.0,
            "rank": 100,
            "decoded_token": "private",
        }
        result = api.summarize_position(
            raw,
            position=17,
            actual_token_id=999,
            identity_key=b"a" * 32,
            top_k=5,
        )
        self.assertEqual(result["position"], 17)
        self.assertEqual(len(result["top_logprobs"]), 6)
        self.assertIn(
            result["actual_token_key"],
            {row["token_key"] for row in result["top_logprobs"]},
        )
        self.assertNotIn("decoded_token", result)

    def test_nonfinite_logprob_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not finite"):
            api.summarize_position(
                {
                    "1": {"logprob": 0.0},
                    "2": {"logprob": float("nan")},
                },
                position=1,
                actual_token_id=1,
                identity_key=b"a" * 32,
                top_k=5,
            )

    def test_server_tokenize_response_is_strictly_validated(self) -> None:
        result = api._response_token_ids(
            {
                "count": 3,
                "max_model_len": 262144,
                "tokens": [1, 2, 3],
            },
            3,
        )
        self.assertEqual(result, [1, 2, 3])
        with self.assertRaisesRegex(ValueError, "sequence is invalid"):
            api._response_token_ids(
                {
                    "count": 3,
                    "max_model_len": 262144,
                    "tokens": [1, True, 3],
                },
                3,
            )


if __name__ == "__main__":
    unittest.main()
