from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tests" / "qualify_m1_86_multi_image_trace.py"
SPEC = importlib.util.spec_from_file_location(
    "qualify_m1_86_multi_image_trace", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


CASE_NAMES = (
    "models_262144_contract",
    "stream_one_image_cold",
    "stream_two_images_cold",
    "stream_two_images_warm",
    "stream_two_images_reversed",
    "stream_two_images_reversed_warm",
    "stream_palette_a_cold",
    "stream_palette_a_warm",
    "stream_palette_b_cold",
    "stream_palette_b_warm",
    "stream_transparency_cold",
    "stream_transparency_warm",
    "post_request_health",
)


def http_report(candidate: bool = True) -> dict:
    cached = {
        "stream_one_image_cold": 0,
        "stream_two_images_cold": 16,
        "stream_two_images_warm": 48,
        "stream_two_images_reversed": 16,
        "stream_two_images_reversed_warm": 48,
        "stream_palette_a_cold": 0,
        "stream_palette_a_warm": 48,
        "stream_palette_b_cold": 0,
        "stream_palette_b_warm": 48,
        "stream_transparency_cold": 0,
        "stream_transparency_warm": 48,
    }
    cases = []
    for name in CASE_NAMES:
        if name == "models_262144_contract":
            evidence = {"max_model_len": 262144}
        elif name == "post_request_health":
            evidence = {"http_status": 200}
        elif not candidate and name.startswith("stream_two_images"):
            evidence = {"skipped": True}
        else:
            evidence = {
                "http_status": 200,
                "prompt_tokens": (
                    16 if name == "stream_one_image_cold" else 48
                ),
                "cached_tokens": cached[name],
            }
        cases.append({"name": name, "ok": True, "evidence": evidence})
    return {
        "schema": "qwen36-diagnostic-multi-image-http-gate-v2",
        "version": 2,
        "qualified": True,
        "cases": cases,
    }


def block(value: int) -> bytes:
    return bytes([value]) * 32


def trace_record(
    ordinal: int,
    hashes: tuple[bytes, ...],
    *,
    raw_hit: int,
    effective_hit: int,
) -> dict:
    return {
        "version": 4,
        "trace_session_sha256": "1" * 16,
        "ordinal": ordinal,
        "request_id_sha256": f"{ordinal:016x}",
        "prompt_tokens": len(hashes) * 16,
        "block_size": 16,
        "capacity_blocks": 20000,
        "gdn_policy": "fine32",
        "initial_raw_kv_contiguous_hit_blocks": raw_hit,
        "raw_kv_contiguous_hit_blocks": raw_hit,
        "effective_gdn_hit_blocks": effective_hit,
        "gdn_restore_digest_base64": (
            base64.b64encode(hashes[effective_hit - 1]).decode("ascii")
            if effective_hit else None
        ),
        "full_blocks": len(hashes),
        "hash_encoding": "sha256_base64",
        "block_hashes": base64.b64encode(b"".join(hashes)).decode("ascii"),
        "observed_effective_cached_tokens": effective_hit * 16,
    }


def candidate_records() -> list[dict]:
    one = (block(1),)
    normal = (block(1), block(2), block(3))
    reversed_images = (block(1), block(4), block(5))
    palette_a = (block(10), block(11), block(12))
    palette_b = (block(20), block(21), block(22))
    transparency = (block(30), block(31), block(32))
    return [
        trace_record(1, one, raw_hit=0, effective_hit=0),
        trace_record(2, normal, raw_hit=1, effective_hit=1),
        trace_record(3, normal, raw_hit=3, effective_hit=3),
        trace_record(4, reversed_images, raw_hit=1, effective_hit=1),
        trace_record(5, reversed_images, raw_hit=3, effective_hit=3),
        trace_record(6, palette_a, raw_hit=0, effective_hit=0),
        trace_record(7, palette_a, raw_hit=3, effective_hit=3),
        trace_record(8, palette_b, raw_hit=0, effective_hit=0),
        trace_record(9, palette_b, raw_hit=3, effective_hit=3),
        trace_record(10, transparency, raw_hit=0, effective_hit=0),
        trace_record(11, transparency, raw_hit=3, effective_hit=3),
    ]


def control_records() -> list[dict]:
    one = (block(1),)
    palette_a = (block(10), block(11), block(12))
    palette_b = (block(20), block(21), block(22))
    transparency = (block(30), block(31), block(32))
    return [
        trace_record(1, one, raw_hit=0, effective_hit=0),
        trace_record(2, palette_a, raw_hit=0, effective_hit=0),
        trace_record(3, palette_a, raw_hit=3, effective_hit=3),
        trace_record(4, palette_b, raw_hit=0, effective_hit=0),
        trace_record(5, palette_b, raw_hit=3, effective_hit=3),
        trace_record(6, transparency, raw_hit=0, effective_hit=0),
        trace_record(7, transparency, raw_hit=3, effective_hit=3),
    ]


def qualify(records: list[dict], report: dict | None = None,
            mode: str = "candidate") -> dict:
    with tempfile.TemporaryDirectory() as temporary:
        log = Path(temporary) / "server.log"
        log.write_text(
            "".join(
                "[BI100_CACHE_TRACE] "
                + json.dumps(record, sort_keys=True)
                + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        return MODULE.qualify(
            log,
            report or http_report(candidate=mode == "candidate"),
            mode,
        )


class QualifyM186MultiImageTraceUnitTest(unittest.TestCase):

    def test_content_bounded_candidate_trace_qualifies(self):
        value = qualify(candidate_records())
        self.assertTrue(value["qualified"], value["reasons"])
        self.assertEqual(value["trace_count"], 11)
        self.assertEqual(
            value["content_isolation"][
                "reversed_initial_prior_common_blocks"],
            1,
        )
        self.assertEqual(
            value["palette_isolation"]["distinct_cold_chain_count"], 3)
        self.assertTrue(all(
            value["palette_isolation"]["warm_chain_exact"].values()))
        rendered = json.dumps(value)
        self.assertNotIn("block_hashes", rendered)
        self.assertNotIn("request_id_sha256", rendered)

    def test_reversed_overhit_fails(self):
        records = candidate_records()
        records[3]["initial_raw_kv_contiguous_hit_blocks"] = 2
        records[3]["raw_kv_contiguous_hit_blocks"] = 2
        records[3]["effective_gdn_hit_blocks"] = 2
        hashes = (block(1), block(4), block(5))
        records[3]["gdn_restore_digest_base64"] = base64.b64encode(
            hashes[1]).decode("ascii")
        records[3]["observed_effective_cached_tokens"] = 32
        report = http_report()
        report["cases"][4]["evidence"]["cached_tokens"] = 32
        value = qualify(records, report)
        self.assertFalse(value["qualified"])
        self.assertIn(
            "stream_two_images_reversed crossed the "
            "content-hash prefix boundary",
            value["reasons"],
        )

    def test_image_order_must_change_prompt_hash_chain(self):
        records = candidate_records()
        records[3]["block_hashes"] = records[1]["block_hashes"]
        records[4]["block_hashes"] = records[1]["block_hashes"]
        value = qualify(records)
        self.assertFalse(value["qualified"])
        self.assertIn(
            "normal and reversed images share one hash chain",
            value["reasons"],
        )

    def test_http_trace_accounting_mismatch_fails(self):
        report = http_report()
        report["cases"][3]["evidence"]["cached_tokens"] = 32
        value = qualify(candidate_records(), report)
        self.assertFalse(value["qualified"])
        self.assertIn(
            "stream_two_images_warm HTTP and trace accounting differ",
            value["reasons"],
        )

    def test_restore_digest_must_match_content_boundary(self):
        records = candidate_records()
        records[3]["gdn_restore_digest_base64"] = base64.b64encode(
            block(99)).decode("ascii")
        value = qualify(records)
        self.assertFalse(value["qualified"])
        self.assertIn(
            "stream_two_images_reversed GDN restore digest differs",
            value["reasons"],
        )

    def test_palette_cold_overhit_fails(self):
        records = candidate_records()
        records[7]["initial_raw_kv_contiguous_hit_blocks"] = 1
        records[7]["raw_kv_contiguous_hit_blocks"] = 1
        records[7]["effective_gdn_hit_blocks"] = 1
        records[7]["gdn_restore_digest_base64"] = base64.b64encode(
            block(20)).decode("ascii")
        records[7]["observed_effective_cached_tokens"] = 16
        report = http_report()
        report["cases"][8]["evidence"]["cached_tokens"] = 16
        value = qualify(records, report)
        self.assertFalse(value["qualified"])
        self.assertIn(
            "stream_palette_b_cold crossed the "
            "content-hash prefix boundary",
            value["reasons"],
        )

    def test_palette_variants_must_have_distinct_hash_chains(self):
        records = candidate_records()
        records[7]["block_hashes"] = records[5]["block_hashes"]
        records[8]["block_hashes"] = records[5]["block_hashes"]
        value = qualify(records)
        self.assertFalse(value["qualified"])
        self.assertIn(
            "palette or transparency variants share a prompt hash chain",
            value["reasons"],
        )

    def test_palette_warm_chain_mismatch_fails(self):
        records = candidate_records()
        replacement = (block(30), block(31), block(99))
        records[10]["block_hashes"] = base64.b64encode(
            b"".join(replacement)).decode("ascii")
        records[10]["gdn_restore_digest_base64"] = base64.b64encode(
            replacement[-1]).decode("ascii")
        value = qualify(records)
        self.assertFalse(value["qualified"])
        self.assertIn(
            "stream_transparency_warm prompt hash chain differs",
            value["reasons"],
        )

    def test_chunked_max_raw_hit_does_not_change_initial_boundary(self):
        records = candidate_records()
        records[3]["raw_kv_contiguous_hit_blocks"] = 1000
        value = qualify(records)
        self.assertTrue(value["qualified"], value["reasons"])

    def test_block_size_must_be_exactly_sixteen(self):
        records = candidate_records()
        records[0]["block_size"] = 8
        records[0]["prompt_tokens"] = 8
        value = qualify(records)
        self.assertFalse(value["qualified"])
        self.assertIn(
            "stream_one_image_cold effective cache accounting differs",
            value["reasons"],
        )

    def test_request_id_must_be_unique_lower_hex(self):
        records = candidate_records()
        records[1]["request_id_sha256"] = records[0]["request_id_sha256"]
        value = qualify(records)
        self.assertFalse(value["qualified"])
        self.assertIn(
            "cache trace request identities differ",
            value["reasons"],
        )

    def test_malformed_ordinal_fails_closed_without_crashing(self):
        records = candidate_records()
        records[2]["ordinal"] = "three"
        value = qualify(records)
        self.assertFalse(value["qualified"])
        self.assertIn(
            "cache trace ordinals are not contiguous",
            value["reasons"],
        )

    def test_control_trace_contains_one_image_and_palette_pairs(self):
        value = qualify(
            control_records(),
            http_report(False),
            "control",
        )
        self.assertTrue(value["qualified"], value["reasons"])
        self.assertEqual(value["trace_count"], 7)


if __name__ == "__main__":
    unittest.main()
