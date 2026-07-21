from __future__ import annotations

import unittest

from tests.measure_m1_48_prefill_service import validate_measurement


class M148PrefillServiceTest(unittest.TestCase):
    def test_valid_privacy_safe_measurement(self):
        request = {
            "elapsed_s": 20.5,
            "ttft_s": 20.0,
            "prompt_tokens": 235000,
            "cached_tokens": 0,
            "completion_tokens": 1,
            "finish_reason": "length",
            "output_sha256": "a" * 64,
        }
        self.assertEqual(validate_measurement(request, 235000), [])

    def test_cached_or_incomplete_measurement_fails(self):
        request = {
            "elapsed_s": 20.5,
            "ttft_s": float("nan"),
            "prompt_tokens": 235000,
            "cached_tokens": 16,
            "completion_tokens": 0,
            "output_sha256": "not-a-digest",
        }
        reasons = validate_measurement(request, 235000)
        self.assertGreaterEqual(len(reasons), 4)


if __name__ == "__main__":
    unittest.main()
