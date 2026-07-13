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

    def test_benchmark_first_event_supports_all_output_delta_types(self):
        deltas = [
            {"content": "answer"},
            {"reasoning_content": "thinking"},
            {"tool_calls": [{"index": 0, "function": {"name": "search"}}]},
        ]
        for delta in deltas:
            with self.subTest(delta=delta), patch(
                    "urllib.request.urlopen",
                    return_value=FakeStreamResponse([
                        {"choices": [{"delta": {"role": "assistant"}}]},
                        {"choices": [{"delta": delta}]},
                        {"choices": [], "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 1,
                        }},
                        "[DONE]",
                    ])), patch(
                        "bench_perf.time.perf_counter",
                        side_effect=[10.0, 10.05, 10.25, 10.75, 10.9, 11.0]):
                result = bench_perf.post_stream(
                    "http://unit.test",
                    bench_perf.make_payload("prompt", 8),
                    timeout=1,
                )

            self.assertTrue(result.ok, result.error)
            self.assertEqual(result.ttft_s, 0.25)
            self.assertEqual(result.output_event_times_s, [0.25])
            for actual, expected in zip(
                    result.sse_event_times_s, [0.05, 0.25, 0.75, 0.9]):
                self.assertAlmostEqual(actual, expected)

    def test_benchmark_reports_decode_itl_and_both_scores(self):
        results = [
            bench_perf.RequestResult(
                True, 10.0, 2.0, 100, 16, 40,
                output_event_times_s=[2.0, 3.0, 5.0, 9.0],
                sse_event_times_s=[1.9, 2.0, 3.0, 5.0, 9.0, 9.9]),
            bench_perf.RequestResult(
                True, 8.0, 1.0, 100, 14, 60,
                output_event_times_s=[1.0, 2.0, 4.0, 7.0]),
        ]

        report = bench_perf.summarize_results(
            results, request_count=2, workers=1, wall_s=20.0, label="unit")

        self.assertAlmostEqual(report["first_event_p90_s"], 1.9)
        self.assertAlmostEqual(report["output_tps_decode_p10"], 2.0)
        self.assertAlmostEqual(report["output_rate_e2e_p10"], 1.615)
        self.assertAlmostEqual(report["prompt_tps_total"], 10.0)
        self.assertAlmostEqual(report["prompt_tps_uncached"], 5.0)
        self.assertAlmostEqual(report["cache_tps"], 5.0)
        self.assertAlmostEqual(report["inter_token_latency_p50_s"], 2.0)
        self.assertAlmostEqual(report["inter_token_latency_p90_s"], 3.5)
        self.assertAlmostEqual(
            report["score_overlap"], bench_perf.score(2.0, 10.0, 5.0))
        self.assertAlmostEqual(
            report["score_disjoint"], bench_perf.score(2.0, 5.0, 5.0))
        self.assertEqual(
            report["request_metrics"][0]["output_event_times_s"],
            [2.0, 3.0, 5.0, 9.0])
        self.assertEqual(
            report["request_metrics"][0]["sse_event_times_s"],
            [1.9, 2.0, 3.0, 5.0, 9.0, 9.9])

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
