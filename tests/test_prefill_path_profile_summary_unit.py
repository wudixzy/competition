from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.summarize_prefill_path_profile import summarize


def _payload(index, prefill, context, paged, counter=True):
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
    }
    if paged:
        regions["paged_attn.prefix_pytorch"] = {
            "count": 10, "total_ms": 30.0}
    counters = ([{
        "name": "paged_attn.prefix_dispatch",
        "path": "pytorch",
        "query_len": prefill,
        "context_len": context,
        "count": 10,
    }] if paged and counter else [])
    return {
        "forward_index": index,
        "metadata": {
            "phase": "prefill",
            "prefill_tokens": prefill,
            "decode_tokens": 0,
            "context_len": context,
            "gdn_restore": False,
            "gdn_capture_points": 1,
            "gdn_evict_keys": 0,
        },
        "event_count": 1,
        "regions": regions,
        "counters": counters,
        "host_model_to_flush_ms": 102.0,
        "host_gap_since_previous_flush_ms": 1.0,
    }


class PrefillPathProfileSummaryTest(unittest.TestCase):
    def _write_log(self, directory: Path, mismatch=False) -> Path:
        lines = []
        records = [
            _payload(0, 8, 0, False),
            _payload(1, 64, 0, False),
            _payload(2, 36, 64, True),
            _payload(3, 4, 96, True),
        ]
        for process_index, pid in enumerate((101, 102, 103, 104)):
            for record in records:
                value = json.loads(json.dumps(record))
                if mismatch and process_index == 3 and value["forward_index"] == 2:
                    value["counters"][0]["query_len"] = 35
                lines.append(
                    f"(VllmWorkerProcess pid={pid}) [BI100_PROFILE_EVENT] "
                    + json.dumps(value, separators=(",", ":")))
        path = directory / "service.log"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_selects_exact_request_and_closes_nested_regions(self):
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            log = self._write_log(directory)
            client = directory / "client.json"
            client.write_text(json.dumps({
                "first": {"elapsed_s": 0.21},
            }), encoding="utf-8")
            report = summarize(
                log, expected_prefill_tokens=100, expected_processes=4,
                client_summary=client, candidate_core_speedup=2.0)

        self.assertTrue(report["qualified_profile"])
        self.assertEqual(report["request"]["forward_count"], 2)
        self.assertAlmostEqual(
            report["full_attention"]["paged_share_of_model_forward"], 0.15)
        self.assertAlmostEqual(
            report["full_attention"]
                  ["amdahl_projected_service_improvement"], 0.075)
        self.assertAlmostEqual(report["coverage"]["exclusive_model_regions"], 1.0)
        self.assertAlmostEqual(
            report["coverage"]["full_attention_subregions"], 1.0)

    def test_rank_counter_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory_text:
            log = self._write_log(Path(directory_text), mismatch=True)
            with self.assertRaisesRegex(ValueError, "counter mismatch"):
                summarize(log, expected_prefill_tokens=100,
                          expected_processes=4)


if __name__ == "__main__":
    unittest.main()
