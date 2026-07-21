from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.summarize_prefill_path_profile import summarize


def _segments(context, prefill, block_size=16):
    strict = ((context + prefill - 1) // block_size) * block_size
    split = strict - context
    if 0 < split < prefill:
        return [(split, context), (prefill - split, strict)]
    return [(prefill, context)]


def _payload(
    index,
    prefill,
    context,
    paged,
    counter=True,
    capture_points=0,
):
    regions = {
        "model.forward": {"count": 1, "total_ms": 100.0},
        "model.embed": {"count": 1, "total_ms": 1.0},
        "layer.input_norm": {"count": 40, "total_ms": 2.0},
        "layer.gdn": {"count": 30, "total_ms": 20.0},
        "layer.full_attn": {"count": 10, "total_ms": 60.0},
        "layer.post_attn_norm": {"count": 40, "total_ms": 2.0},
        "layer.moe": {"count": 40, "total_ms": 14.0},
        "model.final_norm": {"count": 1, "total_ms": 1.0},
        "full_attn.project_qgkv": {"count": 10, "total_ms": 10.0},
        "full_attn.norm_rope": {"count": 10, "total_ms": 5.0},
        "full_attn.attention": {"count": 10, "total_ms": 35.0},
        "full_attn.gate": {"count": 10, "total_ms": 2.0},
        "full_attn.output_proj": {"count": 10, "total_ms": 8.0},
        "moe.router": {"count": 40, "total_ms": 1.0},
        "moe.routed": {"count": 40, "total_ms": 8.0},
        "moe.shared": {"count": 40, "total_ms": 2.0},
        "moe.combine": {"count": 40, "total_ms": 2.0},
        "moe.all_reduce": {"count": 40, "total_ms": 1.0},
        "xformers.kv_write": {"count": 10, "total_ms": 2.0},
    }
    if capture_points:
        regions["gdn_prefix.save"] = {
            "count": capture_points,
            "total_ms": float(capture_points),
        }
    if paged:
        segment_count = len(_segments(context, prefill))
        regions["xformers.paged_prefill"] = {
            "count": 10, "total_ms": 31.0}
        regions["paged_attn.prefix_pytorch"] = {
            "count": 10 * segment_count, "total_ms": 30.0}
    else:
        regions["xformers.dense_prefill"] = {
            "count": 10, "total_ms": 30.0}
    counters = []
    if paged and counter:
        for query_len, segment_context in _segments(context, prefill):
            counters.append({
                "block_size": 16,
                "context_len": segment_context,
                "count": 10,
                "head_dim": 256,
                "kv_heads": 1,
                "name": "paged_attn.prefix_dispatch",
                "path": "pytorch",
                "query_heads": 8,
                "query_len": query_len,
                "request_query_len": prefill,
            })
        counters.sort(key=lambda row: json.dumps(row, sort_keys=True))
    return {
        "schema": "bi100-profile-event-v1",
        "version": 1,
        "tp_rank": 0,
        "forward_index": index,
        "metadata": {
            "phase": "prefill",
            "prefill_tokens": prefill,
            "decode_tokens": 0,
            "context_len": context,
            "gdn_restore": False,
            "gdn_capture_points": capture_points,
            "gdn_evict_keys": 0,
        },
        "event_count": sum(item["count"] for item in regions.values()),
        "model_forward_event_count": 1,
        "regions": regions,
        "counters": counters,
        "host_model_start_to_flush_ms": 102.0,
        "host_gap_since_previous_flush_ms": 1.0,
    }


def _service(mode, ttft, output_sha="a" * 64):
    return {
        "schema": "bi100-m1-48-prefill-service-v1",
        "version": 1,
        "mode": mode,
        "run_id": "m148-unit",
        "protocol": {
            "stream": True,
            "max_tokens": 1,
            "min_tokens": 1,
            "temperature": 0,
            "seed": 20260722,
            "thinking": False,
            "target_prompt_tokens": 100,
            "max_model_len": 262144,
        },
        "request": {
            "elapsed_s": ttft + 0.01,
            "ttft_s": ttft,
            "prompt_tokens": 100,
            "cached_tokens": 0,
            "completion_tokens": 1,
            "finish_reason": "length",
            "output_sha256": output_sha,
        },
        "qualified_measurement": True,
        "reasons": [],
    }


class PrefillPathProfileSummaryTest(unittest.TestCase):
    def _write_log(
        self,
        directory: Path,
        mismatch: bool = False,
        extra_request: bool = False,
    ) -> Path:
        lines = []
        records = [
            _payload(0, 64, 0, True),
            _payload(1, 36, 64, True, capture_points=1),
        ]
        if extra_request:
            records = [
                _payload(0, 8, 0, False),
                _payload(1, 64, 0, True),
                _payload(2, 36, 64, True, capture_points=1),
            ]
        for process_index, pid in enumerate((101, 102, 103, 104)):
            for record in records:
                value = json.loads(json.dumps(record))
                value["tp_rank"] = process_index
                if mismatch and process_index == 3 and value["forward_index"] == 1:
                    value["counters"][0]["query_len"] -= 1
                lines.append(
                    f"(VllmWorkerProcess pid={pid}) [BI100_PROFILE_EVENT] "
                    + json.dumps(value, separators=(",", ":")))
        path = directory / "service.log"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _write_services(
        self,
        directory: Path,
        control_ttft=0.20,
        profile_ttft=0.21,
    ) -> tuple[Path, Path]:
        control = directory / "control.json"
        profile = directory / "profile.json"
        control.write_text(
            json.dumps(_service("control", control_ttft)), encoding="utf-8")
        profile.write_text(
            json.dumps(_service("profile", profile_ttft)), encoding="utf-8")
        return control, profile

    def _summarize(self, directory: Path, **kwargs):
        log = self._write_log(
            directory,
            kwargs.pop("mismatch", False),
            kwargs.pop("extra_request", False),
        )
        control, profile = self._write_services(
            directory,
            kwargs.pop("control_ttft", 0.20),
            kwargs.pop("profile_ttft", 0.21),
        )
        return summarize(
            log,
            expected_prefill_tokens=100,
            expected_processes=4,
            profile_service=profile,
            control_service=control,
            expected_chunk_size=64,
            block_size=16,
            **kwargs,
        )

    def test_closes_exact_request_and_nested_regions(self):
        with tempfile.TemporaryDirectory() as directory_text:
            report = self._summarize(Path(directory_text))

        self.assertTrue(report["qualified_profile"], report)
        self.assertEqual(report["request"]["forward_count"], 2)
        self.assertEqual(report["request"]["tp_ranks"], [0, 1, 2, 3])
        self.assertAlmostEqual(
            report["full_attention"]["paged_share_of_model_work"], 0.30)
        self.assertAlmostEqual(
            report["full_attention"]
                  ["paged_critical_upper_share_of_control_ttft"], 0.30)
        self.assertAlmostEqual(
            report["request"]["worker_critical_path_ms"], 205.0)
        self.assertAlmostEqual(
            report["request"]["worker_share_of_profiled_ttft"], 205 / 210)
        self.assertTrue(all(
            abs(value - 1.0) < 1e-9
            for value in report["coverage"]
                               ["exclusive_model_regions_by_rank"].values()))
        self.assertTrue(all(
            abs(value - 1.0) < 1e-9
            for value in report["coverage"]
                               ["full_attention_subregions_by_rank"].values()))

    def test_rank_counter_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            with self.assertRaisesRegex(ValueError, "counter mismatch"):
                self._summarize(directory, mismatch=True)

    def test_extra_profile_request_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            with self.assertRaisesRegex(
                    ValueError, "exactly one profiled request"):
                self._summarize(directory, extra_request=True)

    def test_cold_request_rejects_early_gdn_capture(self):
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            log = self._write_log(directory)
            rewritten = []
            for line in log.read_text(encoding="utf-8").splitlines():
                prefix, encoded = line.split("[BI100_PROFILE_EVENT] ", 1)
                payload = json.loads(encoded)
                if payload["forward_index"] == 0:
                    payload["metadata"]["gdn_capture_points"] = 1
                    payload["regions"]["gdn_prefix.save"] = {
                        "count": 1,
                        "total_ms": 1.0,
                    }
                    payload["event_count"] += 1
                rewritten.append(
                    prefix + "[BI100_PROFILE_EVENT] "
                    + json.dumps(payload, separators=(",", ":")))
            log.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
            control, profile = self._write_services(directory)

            report = summarize(
                log, 100, 4, profile, control,
                expected_chunk_size=64, block_size=16)

        self.assertFalse(report["qualified_profile"])
        self.assertTrue(any(
            "cold prefill capture count at offset 0 must be 0" in reason
            for reason in report["reasons"]), report)

    def test_duplicate_tp_rank_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            log = self._write_log(directory)
            text = log.read_text(encoding="utf-8")
            text = text.replace('"tp_rank":3', '"tp_rank":2')
            text = text.replace('"tp_rank": 3', '"tp_rank": 2')
            log.write_text(text, encoding="utf-8")
            control, profile = self._write_services(directory)
            with self.assertRaisesRegex(ValueError, "duplicate TP rank"):
                summarize(
                    log, 100, 4, profile, control,
                    expected_chunk_size=64, block_size=16)

    def test_large_frontend_gap_is_reported_not_rejected(self):
        with tempfile.TemporaryDirectory() as directory_text:
            report = self._summarize(
                Path(directory_text), control_ttft=0.49, profile_ttft=0.50)

        self.assertTrue(report["qualified_profile"], report)
        self.assertAlmostEqual(
            report["request"]["worker_share_of_profiled_ttft"], 0.41)
        self.assertAlmostEqual(
            report["request"]["model_share_of_profiled_ttft"], 0.4)

    def test_profile_overhead_above_fixed_bound_is_unqualified(self):
        with tempfile.TemporaryDirectory() as directory_text:
            report = self._summarize(
                Path(directory_text), control_ttft=0.20, profile_ttft=0.24)

        self.assertFalse(report["qualified_profile"])
        self.assertIn(
            "profile TTFT perturbation exceeds the fixed 15% bound",
            report["reasons"],
        )

    def test_alternating_slow_rank_cannot_hide_in_request_aggregate(self):
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            log = self._write_log(directory)
            rewritten = []
            for line in log.read_text(encoding="utf-8").splitlines():
                prefix, encoded = line.split("[BI100_PROFILE_EVENT] ", 1)
                payload = json.loads(encoded)
                if payload["forward_index"] in {0, 1}:
                    slow = ((payload["tp_rank"] + payload["forward_index"])
                            % 2 == 0)
                    factor = 1.8 if slow else 0.2
                    for region in payload["regions"].values():
                        region["total_ms"] *= factor
                    payload["host_model_start_to_flush_ms"] = (
                        182.0 if slow else 22.0)
                rewritten.append(
                    prefix + "[BI100_PROFILE_EVENT] "
                    + json.dumps(payload, separators=(",", ":")))
            log.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
            control, profile = self._write_services(
                directory, control_ttft=0.25, profile_ttft=0.25)

            report = summarize(
                log, 100, 4, profile, control,
                expected_chunk_size=64, block_size=16)

        self.assertAlmostEqual(
            report["request"]["model_rank_spread_fraction"], 0.0)
        self.assertGreater(
            report["request"]["max_forward_model_rank_spread_fraction"],
            1.0,
        )
        self.assertFalse(report["qualified_profile"])
        self.assertIn(
            "per-forward model TP-rank spread exceeds 10%",
            report["reasons"],
        )

    def test_unknown_metadata_field_fails_with_bounded_error(self):
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            log = self._write_log(directory)
            lines = log.read_text(encoding="utf-8").splitlines()
            prefix, encoded = lines[1].split("[BI100_PROFILE_EVENT] ", 1)
            payload = json.loads(encoded)
            payload["metadata"]["unexpected"] = 1
            lines[1] = (
                prefix + "[BI100_PROFILE_EVENT] "
                + json.dumps(payload, separators=(",", ":")))
            log.write_text("\n".join(lines) + "\n", encoding="utf-8")
            control, profile = self._write_services(directory)

            with self.assertRaisesRegex(ValueError, "metadata fields differ"):
                summarize(
                    log, 100, 4, profile, control,
                    expected_chunk_size=64, block_size=16)

    def test_region_event_count_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            log = self._write_log(directory)
            lines = log.read_text(encoding="utf-8").splitlines()
            prefix, encoded = lines[1].split("[BI100_PROFILE_EVENT] ", 1)
            payload = json.loads(encoded)
            payload["event_count"] = 1
            lines[1] = (
                prefix + "[BI100_PROFILE_EVENT] "
                + json.dumps(payload, separators=(",", ":")))
            log.write_text("\n".join(lines) + "\n", encoding="utf-8")
            control, profile = self._write_services(directory)
            with self.assertRaisesRegex(ValueError, "event_count"):
                summarize(
                    log, 100, 4, profile, control,
                    expected_chunk_size=64, block_size=16)


if __name__ == "__main__":
    unittest.main()
