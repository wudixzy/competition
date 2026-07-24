from __future__ import annotations

import pathlib
import sys
import unittest


TESTS = pathlib.Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from qualify_long_context_summary import qualify


def _request(cached: int, digest: str) -> dict:
    return {
        "cached_tokens": cached,
        "completion_tokens": 16,
        "elapsed_s": 1.0,
        "finish_reason": "length",
        "message_sha256": digest,
        "prompt_tokens": 262_000,
        "ignored_raw_field": "must not be copied",
    }


def _source(mode: str = "exact") -> dict:
    digest = "a" * 64
    source = {
        "equivalence_mode": mode,
        "first": _request(0, digest),
        "max_tokens": 16,
        "min_cached_tokens": 261_984,
        "max_first_cached_tokens": 0,
        "min_completion_tokens": 16,
        "second": _request(261_984, digest),
        "target_prompt_tokens": 262_000,
    }
    if mode == "warm-repeat":
        source["first"]["message_sha256"] = "b" * 64
        source["third"] = _request(261_984, digest)
    return source


def _qualify(
    source: dict,
    mode: str = "exact",
    maximum_first_cached_tokens: int = 0,
) -> dict:
    return qualify(
        source,
        target_prompt_tokens=262_000,
        max_tokens=16,
        minimum_cached_tokens=261_984,
        maximum_first_cached_tokens=maximum_first_cached_tokens,
        minimum_completion_tokens=16,
        equivalence_mode=mode,
    )


class LongContextSafeGateTest(unittest.TestCase):

    def test_exact_gate_qualifies_and_drops_unknown_fields(self):
        report = _qualify(_source())
        self.assertTrue(report["qualified"], report)
        self.assertNotIn(
            "ignored_raw_field", report["requests"]["first"])

    def test_warm_repeat_compares_second_and_third(self):
        report = _qualify(_source("warm-repeat"), "warm-repeat")
        self.assertTrue(report["qualified"], report)

    def test_nonzero_cold_cache_is_rejected(self):
        source = _source()
        source["first"]["cached_tokens"] = 16
        report = _qualify(source)
        self.assertFalse(report["qualified"], report)
        self.assertIn(
            "first cached_tokens must not exceed 0", report["reasons"])

    def test_bounded_shared_cold_prefix_is_accepted(self):
        source = _source()
        source["max_first_cached_tokens"] = 32
        source["first"]["cached_tokens"] = 32
        report = _qualify(source, maximum_first_cached_tokens=32)
        self.assertTrue(report["qualified"], report)

    def test_shared_cold_prefix_over_bound_is_rejected(self):
        source = _source()
        source["max_first_cached_tokens"] = 32
        source["first"]["cached_tokens"] = 33
        report = _qualify(source, maximum_first_cached_tokens=32)
        self.assertFalse(report["qualified"], report)
        self.assertIn(
            "first cached_tokens must not exceed 32", report["reasons"])

    def test_missing_bound_is_compatible_only_with_zero(self):
        source = _source()
        del source["max_first_cached_tokens"]
        self.assertTrue(_qualify(source)["qualified"])
        report = _qualify(source, maximum_first_cached_tokens=32)
        self.assertFalse(report["qualified"], report)
        self.assertIn(
            "max_first_cached_tokens must equal 32", report["reasons"])

    def test_invalid_bound_type_fails_closed(self):
        source = _source()
        source["max_first_cached_tokens"] = "32"
        report = _qualify(source, maximum_first_cached_tokens="32")
        self.assertFalse(report["qualified"], report)
        self.assertIn(
            "maximum_first_cached_tokens must be nonnegative and less than "
            "minimum_cached_tokens",
            report["reasons"],
        )

    def test_digest_mismatch_is_rejected(self):
        source = _source()
        source["second"]["message_sha256"] = "c" * 64
        report = _qualify(source)
        self.assertFalse(report["qualified"], report)
        self.assertTrue(any("message_sha256" in reason
                            for reason in report["reasons"]))

    def test_contract_mismatch_is_rejected(self):
        source = _source()
        source["target_prompt_tokens"] = 261_999
        report = _qualify(source)
        self.assertFalse(report["qualified"], report)
        self.assertTrue(any("target_prompt_tokens" in reason
                            for reason in report["reasons"]))


if __name__ == "__main__":
    unittest.main()
