from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = pathlib.Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import multimodal_prefix_isolation_api as isolation_api


def _response(color: str, cached_tokens: int, prompt_tokens: int = 9000) -> dict:
    return {
        "choices": [{
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": color + "色"},
        }],
        "usage": {
            "completion_tokens": 2,
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
        },
    }


class MultimodalPrefixIsolationApiTest(unittest.TestCase):

    @patch.object(isolation_api.smoke_api, "post_chat")
    def test_same_image_hits_and_different_image_isolated(self, post_chat):
        post_chat.side_effect = [
            _response("红", 0),
            _response("红", 8992),
            _response("绿", 0, prompt_tokens=9002),
        ]

        report = isolation_api.run_gate("http://unit.test", "unit-gate")

        self.assertTrue(report["qualified"], report)
        self.assertEqual(report["reasons"], [])
        self.assertEqual(
            report["requests"]["same_image_warm"]["cached_tokens"], 8992)
        encoded = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("红色", encoded)
        self.assertNotIn("绿色", encoded)
        self.assertEqual(post_chat.call_count, 3)

    @patch.object(isolation_api.smoke_api, "post_chat")
    def test_cross_image_hit_fails_closed(self, post_chat):
        post_chat.side_effect = [
            _response("红", 0),
            _response("红", 8992),
            _response("绿", 16),
        ]

        report = isolation_api.run_gate("http://unit.test", "unit-fail")

        self.assertFalse(report["qualified"], report)
        self.assertEqual(report["reasons"], ["different_image_isolated"])

    def test_atomic_report_contains_no_payload(self):
        report = {
            "qualified": True,
            "response_sha256": "a" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "nested" / "result.json"
            isolation_api._write_report(path, report)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), report)
            self.assertFalse(path.with_name(path.name + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
