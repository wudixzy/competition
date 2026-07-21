from __future__ import annotations

import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import qualify_multimodal_prefix_isolation as qualifier


def _request(cached_tokens: int, prompt_tokens: int, digest: str) -> dict:
    return {
        "cached_tokens": cached_tokens,
        "completion_tokens": 2,
        "elapsed_s": 1.0,
        "expected_color_observed": True,
        "finish_reason": "stop",
        "prompt_tokens": prompt_tokens,
        "response_sha256": digest,
    }


def _source(different_prompt_tokens: int = 5072) -> dict:
    red_digest = "a" * 64
    return {
        "checks": {},
        "qualified": False,
        "reasons": ["legacy prompt token equality"],
        "requests": {
            "same_image_cold": _request(0, 5070, red_digest),
            "same_image_warm": _request(5056, 5070, red_digest),
            "different_image": _request(
                0, different_prompt_tokens, "b" * 64),
        },
        "run_id": "unit",
        "schema": qualifier.SOURCE_SCHEMA,
        "version": 1,
    }


class MultimodalIsolationQualifierTest(unittest.TestCase):

    def test_different_image_token_count_is_informational(self):
        report = qualifier.qualify(_source())
        self.assertTrue(report["qualified"], report)
        self.assertEqual(report["different_image_prompt_token_delta"], 2)

    def test_cross_image_hit_fails(self):
        source = _source()
        source["requests"]["different_image"]["cached_tokens"] = 16
        report = qualifier.qualify(source)
        self.assertFalse(report["qualified"], report)
        self.assertIn("different_image_isolated", report["reasons"])

    def test_same_image_prompt_mismatch_fails(self):
        source = _source()
        source["requests"]["same_image_warm"]["prompt_tokens"] = 5069
        report = qualifier.qualify(source)
        self.assertFalse(report["qualified"], report)
        self.assertIn("red_cold_warm_exact", report["reasons"])


if __name__ == "__main__":
    unittest.main()
