import json
import pathlib
import sys
import tempfile
import unittest
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import bench_perf
import smoke_api


class FakeStreamResponse:

    def __init__(self, lines: list[dict | str], status: int = 200):
        self.status = status
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        for item in self._lines:
            if item == "[DONE]":
                yield b"data: [DONE]\n\n"
            else:
                payload = json.dumps(item, ensure_ascii=False)
                yield f"data: {payload}\n\n".encode("utf-8")


def _stream_chunks() -> list[dict | str]:
    return [
        {"choices": [], "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 7},
        }},
        {"choices": [{"delta": {"content": "ok"}}]},
        {"choices": [], "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 7},
        }},
        "[DONE]",
    ]


class StreamingClientTest(unittest.TestCase):

    def test_smoke_streaming_accepts_usage_only_chunks(self):
        with patch("urllib.request.urlopen",
                   return_value=FakeStreamResponse(_stream_chunks())):
            smoke_api.test_streaming_sse("http://unit.test")

    def test_benchmark_streaming_accepts_usage_only_chunks(self):
        with patch("urllib.request.urlopen",
                   return_value=FakeStreamResponse(_stream_chunks())):
            result = bench_perf.post_stream(
                "http://unit.test",
                bench_perf.make_payload("prompt", 8),
                timeout=1,
            )

        self.assertTrue(result.ok, result.error)
        self.assertIsNotNone(result.ttft_s)
        self.assertEqual(result.prompt_tokens, 11)
        self.assertEqual(result.completion_tokens, 2)
        self.assertEqual(result.cached_tokens, 7)

    def test_benchmark_percentile_and_score_formula(self):
        self.assertEqual(bench_perf.percentile([], 90), 0.0)
        self.assertEqual(bench_perf.percentile([3.0], 90), 3.0)
        self.assertAlmostEqual(bench_perf.percentile([1.0, 2.0, 3.0], 50),
                               2.0)
        self.assertAlmostEqual(bench_perf.percentile([1.0, 2.0, 3.0], 10),
                               1.2)
        self.assertAlmostEqual(
            bench_perf.score(output_tps_p10=2.0, input_tps=3.0, cache_tps=4.0),
            2.0 * 16.796 + 3.0 * 2.799 + 4.0 * 0.56,
        )

    def test_benchmark_payload_adds_seed_only_when_requested(self):
        self.assertNotIn("seed", bench_perf.make_payload("prompt", 8))
        self.assertEqual(
            bench_perf.make_payload("prompt", 8, seed=123)["seed"], 123)

    def test_smoke_request_json_preserves_json_http_error_body(self):
        error = HTTPError(
            url="http://unit.test/v1/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs={},
            fp=BytesIO(b'{"error":{"message":"bad messages"}}'),
        )
        with patch("urllib.request.urlopen", side_effect=error):
            status, data = smoke_api._request_json(
                "POST",
                "http://unit.test/v1/chat/completions",
                {"model": "llm"},
                timeout=1,
            )

        self.assertEqual(status, 400)
        self.assertEqual(data["error"]["message"], "bad messages")

    def test_smoke_request_json_preserves_raw_http_error_body(self):
        error = HTTPError(
            url="http://unit.test/v1/chat/completions",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=BytesIO("非 JSON 错误".encode("utf-8")),
        )
        with patch("urllib.request.urlopen", side_effect=error):
            status, data = smoke_api._request_json(
                "POST",
                "http://unit.test/v1/chat/completions",
                {"model": "llm"},
                timeout=1,
            )

        self.assertEqual(status, 500)
        self.assertEqual(data, {"raw": "非 JSON 错误"})

    def test_smoke_runner_writes_machine_readable_success_report(self):
        def passing(base: str) -> None:
            self.assertEqual(base, "http://unit.test")

        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "smoke.json"
            report = smoke_api.run_smoke_tests(
                "http://unit.test", [passing], mode="quick", json_out=str(out))

            self.assertTrue(report["ok"])
            persisted = json.loads(out.read_text())
            self.assertTrue(persisted["ok"])
            self.assertEqual(persisted["tests"][0]["name"], "passing")
            self.assertTrue(persisted["tests"][0]["ok"])

    def test_smoke_runner_persists_failure_before_raising(self):
        def failing(base: str) -> None:
            raise RuntimeError("expected failure")

        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "smoke.json"
            with self.assertRaisesRegex(RuntimeError, "expected failure"):
                smoke_api.run_smoke_tests(
                    "http://unit.test", [failing], mode="full", json_out=str(out))

            persisted = json.loads(out.read_text())
            self.assertFalse(persisted["ok"])
            self.assertFalse(persisted["tests"][0]["ok"])
            self.assertIn("expected failure", persisted["tests"][0]["error"])


if __name__ == "__main__":
    unittest.main()
