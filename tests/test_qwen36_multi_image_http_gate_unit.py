from __future__ import annotations

import base64
import importlib.util
from pathlib import Path
import struct
import sys
import unittest
from unittest import mock
import zlib


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tests" / "qwen36_multi_image_http_gate.py"
sys.path.insert(0, str(ROOT / "tests"))
SPEC = importlib.util.spec_from_file_location(
    "qwen36_multi_image_http_gate", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def image_count(payload: dict) -> int:
    return sum(
        item.get("type") == "image_url"
        for item in payload["messages"][0]["content"]
    )


def image_order(payload: dict) -> tuple[str, ...]:
    return tuple(
        item["image_url"]["url"]
        for item in payload["messages"][0]["content"]
        if item.get("type") == "image_url"
    )


def stream_payload(
    semantic: str,
    *,
    prompt_tokens: int,
    cached_tokens: int,
) -> dict:
    return {
        "chunks": 3,
        "done": 1,
        "usage_blocks": 1,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 1,
            "total_tokens": prompt_tokens + 1,
            "prompt_tokens_details": {
                "cached_tokens": cached_tokens,
            },
        },
        "content": semantic,
        "reasoning_content": "",
        "finish_reasons": ["stop"],
        "tool_calls": [],
    }


def png_chunks(data_url: str) -> dict[bytes, bytes]:
    raw = base64.b64decode(data_url.split(",", 1)[1], validate=True)
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError("PNG signature differs")
    chunks: dict[bytes, bytes] = {}
    offset = 8
    while offset < len(raw):
        length = struct.unpack(">I", raw[offset:offset + 4])[0]
        kind = raw[offset + 4:offset + 8]
        payload = raw[offset + 8:offset + 8 + length]
        expected_crc = struct.unpack(
            ">I", raw[offset + 8 + length:offset + 12 + length])[0]
        if zlib.crc32(kind + payload) & 0xffffffff != expected_crc:
            raise AssertionError("PNG chunk CRC differs")
        chunks[kind] = payload
        offset += 12 + length
    if offset != len(raw):
        raise AssertionError("PNG chunk boundary differs")
    return chunks


class FakeClient:

    def __init__(
        self,
        image_limit: int,
        *,
        warm_drift: bool = False,
        reversed_hit: bool = False,
        indexed_warm_drift: bool = False,
        cross_indexed_hit: bool = False,
        max_model_len: int = 262144,
        top_level_error: bool = False,
    ) -> None:
        self.image_limit = image_limit
        self.warm_drift = warm_drift
        self.reversed_hit = reversed_hit
        self.indexed_warm_drift = indexed_warm_drift
        self.cross_indexed_hit = cross_indexed_hit
        self.max_model_len = max_model_len
        self.top_level_error = top_level_error
        self.seen: dict[tuple[str, ...], int] = {}
        self.indexed_cold_count = 0

    def models(self) -> dict:
        return {
            "data": [{
                "id": "llm",
                "max_model_len": self.max_model_len,
            }],
        }

    def stream(self, payload: dict, *, timeout: float) -> tuple[int, dict]:
        del timeout
        count = image_count(payload)
        if count > self.image_limit:
            return 400, {}
        order = image_order(payload)
        visits = self.seen.get(order, 0)
        self.seen[order] = visits + 1
        cached_tokens = 32 if visits else 0
        indexed = (
            count == 1
            and order[0].startswith("data:image/png;base64,iVBOR")
        )
        if self.reversed_hit and count == 2 and visits == 0:
            red_first = "255,0,0" in order[0]
            if not red_first:
                cached_tokens = 16
        if indexed and visits == 0:
            if self.cross_indexed_hit and self.indexed_cold_count:
                cached_tokens = 16
            self.indexed_cold_count += 1
        semantic = f"images={count};order={order}"
        if self.warm_drift and count == 2 and visits:
            semantic += ";drift"
        if self.indexed_warm_drift and indexed and visits:
            semantic += ";indexed-drift"
        return 200, stream_payload(
            semantic,
            prompt_tokens=128 * count + 64,
            cached_tokens=cached_tokens,
        )

    def post(self, payload: dict, *, timeout: float) -> tuple[int, dict]:
        del timeout
        count = image_count(payload)
        if count > self.image_limit:
            error = {
                "message": "At most one image is allowed.",
                "type": "BadRequestError",
                "code": 400,
            }
            return 400, error if self.top_level_error else {"error": error}
        raise AssertionError("post is only expected for the 4xx control path")


def health_request(
    method: str,
    url: str,
    payload: dict | None = None,
    *,
    timeout_s: float,
) -> tuple[int, dict]:
    del method, url, payload, timeout_s
    return 200, {}


class MultiImageHttpGateUnitTest(unittest.TestCase):

    def run_gate(self, client: FakeClient, expected: int) -> dict:
        with mock.patch.object(MODULE, "_solid_png_data_url") as image:
            image.side_effect = lambda rgb: (
                "data:image/png;base64," + ",".join(map(str, rgb))
            )
            return MODULE.run_gate(
                "http://127.0.0.1:8000",
                Path("/tmp/model"),
                30,
                expected,
                client=client,
                request_json=health_request,
            )

    def test_indexed_png_variants_preserve_pixels_and_change_metadata(self):
        palette_a = MODULE._indexed_png_data_url(
            ((10, 20, 30), (40, 50, 60)), (255, 255))
        palette_b = MODULE._indexed_png_data_url(
            ((200, 20, 30), (40, 50, 60)), (255, 255))
        transparency = MODULE._indexed_png_data_url(
            ((10, 20, 30), (40, 50, 60)), (0, 255))
        a = png_chunks(palette_a)
        b = png_chunks(palette_b)
        transparent = png_chunks(transparency)
        self.assertEqual(a[b"IHDR"], b[b"IHDR"])
        self.assertEqual(a[b"IHDR"], transparent[b"IHDR"])
        self.assertNotEqual(a[b"PLTE"], b[b"PLTE"])
        self.assertEqual(a[b"PLTE"], transparent[b"PLTE"])
        self.assertEqual(a[b"tRNS"], b[b"tRNS"])
        self.assertNotEqual(a[b"tRNS"], transparent[b"tRNS"])
        self.assertEqual(
            zlib.decompress(a[b"IDAT"]),
            zlib.decompress(b[b"IDAT"]),
        )
        self.assertEqual(
            zlib.decompress(a[b"IDAT"]),
            zlib.decompress(transparent[b"IDAT"]),
        )

    def test_control_rejects_only_second_image_and_stays_healthy(self):
        report = self.run_gate(FakeClient(1), 400)
        self.assertTrue(report["qualified"])
        self.assertEqual(report["case_count"], 13)
        cases = {case["name"]: case for case in report["cases"]}
        self.assertEqual(
            cases["stream_one_image_cold"]["evidence"]["http_status"], 200)
        self.assertEqual(
            cases["stream_two_images_cold"]["evidence"]["http_status"], 400)
        self.assertEqual(
            cases["stream_two_images_cold"]["evidence"]["error_fields"],
            ["code", "message", "type"],
        )
        self.assertEqual(
            cases["stream_two_images_cold"]["evidence"]["error_shape"],
            "nested",
        )
        self.assertTrue(
            cases["stream_two_images_warm"]["evidence"]["skipped"])
        for cold_name, warm_name in MODULE.PALETTE_PAIRS:
            cold = cases[cold_name]["evidence"]
            warm = cases[warm_name]["evidence"]
            self.assertEqual(cold["cached_tokens"], 0)
            self.assertTrue(cold["cross_variant_cached_tokens_zero"])
            self.assertGreater(warm["cached_tokens"], 0)
            self.assertTrue(warm["cold_generation_exact"])
        self.assertEqual(
            cases["post_request_health"]["evidence"]["http_status"], 200)

    def test_control_accepts_top_level_openai_error_response(self):
        report = self.run_gate(
            FakeClient(1, top_level_error=True),
            400,
        )
        self.assertTrue(report["qualified"])
        cases = {case["name"]: case for case in report["cases"]}
        evidence = cases["stream_two_images_cold"]["evidence"]
        self.assertEqual(evidence["http_status"], 400)
        self.assertEqual(evidence["error_shape"], "top_level")
        self.assertEqual(
            evidence["error_fields"],
            ["code", "message", "type"],
        )

    def test_candidate_accepts_two_images_with_exact_warm_and_isolation(self):
        report = self.run_gate(FakeClient(2), 200)
        self.assertTrue(report["qualified"])
        cases = {case["name"]: case for case in report["cases"]}
        warm = cases["stream_two_images_warm"]["evidence"]
        reversed_images = cases["stream_two_images_reversed"]["evidence"]
        reversed_warm = cases[
            "stream_two_images_reversed_warm"
        ]["evidence"]
        self.assertTrue(warm["cold_generation_exact"])
        self.assertGreater(warm["cached_tokens"], 0)
        self.assertTrue(
            reversed_images["cache_isolation_deferred_to_trace"])
        self.assertTrue(reversed_warm["cold_generation_exact"])
        self.assertGreater(reversed_warm["cached_tokens"], 0)
        for cold_name, warm_name in MODULE.PALETTE_PAIRS:
            cold = cases[cold_name]["evidence"]
            warm = cases[warm_name]["evidence"]
            self.assertEqual(cold["cached_tokens"], 0)
            self.assertTrue(cold["cross_variant_cached_tokens_zero"])
            self.assertGreater(warm["cached_tokens"], 0)
            self.assertTrue(warm["cold_generation_exact"])

    def test_candidate_fails_closed_on_warm_generation_drift(self):
        report = self.run_gate(FakeClient(2, warm_drift=True), 200)
        self.assertFalse(report["qualified"])
        failed = [
            case for case in report["cases"]
            if not case["ok"]
        ]
        self.assertEqual(
            [case["name"] for case in failed],
            [
                "stream_two_images_warm",
                "stream_two_images_reversed_warm",
            ],
        )

    def test_indexed_palette_warm_generation_drift_fails_closed(self):
        report = self.run_gate(
            FakeClient(2, indexed_warm_drift=True), 200)
        self.assertFalse(report["qualified"])
        self.assertEqual(
            [
                case["name"] for case in report["cases"]
                if not case["ok"]
            ],
            [warm for _, warm in MODULE.PALETTE_PAIRS],
        )

    def test_cross_palette_cache_hit_fails_closed(self):
        report = self.run_gate(
            FakeClient(2, cross_indexed_hit=True), 200)
        self.assertFalse(report["qualified"])
        failed = {
            case["name"] for case in report["cases"]
            if not case["ok"]
        }
        self.assertIn("stream_palette_b_cold", failed)
        self.assertIn("stream_transparency_cold", failed)

    def test_cross_image_cache_accounting_is_deferred_to_trace(self):
        report = self.run_gate(FakeClient(2, reversed_hit=True), 200)
        self.assertTrue(report["qualified"])
        cases = {case["name"]: case for case in report["cases"]}
        reversed_images = cases[
            "stream_two_images_reversed"
        ]["evidence"]
        self.assertGreater(reversed_images["cached_tokens"], 0)
        self.assertTrue(
            reversed_images["cache_isolation_deferred_to_trace"])

    def test_capacity_contract_is_mandatory(self):
        report = self.run_gate(
            FakeClient(2, max_model_len=100000), 200)
        self.assertFalse(report["qualified"])
        self.assertFalse(report["cases"][0]["ok"])

    def test_report_is_privacy_safe_and_does_not_authorize_promotion(self):
        report = self.run_gate(FakeClient(2), 200)
        privacy = report["privacy"]
        self.assertFalse(privacy["contains_raw_request"])
        self.assertFalse(privacy["contains_raw_response"])
        self.assertFalse(privacy["contains_image_url_or_bytes"])
        self.assertFalse(privacy["contains_prompt_or_generated_text"])
        self.assertFalse(privacy["contains_credentials"])
        self.assertTrue(privacy["synthetic_images_only"])
        self.assertFalse(report["semantic_quality_evaluated"])
        self.assertFalse(report["full_model_evaluated"])
        self.assertFalse(report["production_promotion_authorized"])
        rendered = str(report)
        self.assertNotIn("data:image", rendered)
        self.assertNotIn("Return one short token", rendered)


if __name__ == "__main__":
    unittest.main()
