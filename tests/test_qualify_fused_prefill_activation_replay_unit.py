from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests" / "qualify_fused_prefill_activation_replay.py"
SPEC = importlib.util.spec_from_file_location("qualify_replay", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
CONTRACT = json.loads(
    (ROOT / "quality" / "experiment_funnel.v1.json").read_text())


def report(rank: int, buckets=(24576,), ordinals=(0,), speedup=1.5):
    records = []
    for bucket in buckets:
        for ordinal in ordinals:
            records.append({
                "rank": rank,
                "bucket_min_context_tokens": bucket,
                "call_ordinal": ordinal,
                "candidate_speedup": speedup,
                "numeric": {
                    "finite": True,
                    "lse_finite": True,
                    "qualified": True,
                },
            })
    return {
        "schema": MODULE.REPORT_SCHEMA,
        "version": 1,
        "capture_source_revision": "a" * 40,
        "candidate_source_revision": "c" * 40,
        "runtime_identity": "overlay",
        "instance": "instance",
        "rank": rank,
        "all_numeric_qualified": True,
        "candidate_extension": {"sha256": "b" * 64},
        "records": records,
    }


class QualifyActivationReplayTest(unittest.TestCase):

    def test_smoke_profile_passes_four_rank_subset_without_authorizing_tp4(self):
        result = MODULE.qualify(
            [report(rank) for rank in range(4)],
            CONTRACT,
            profile="smoke",
        )
        self.assertTrue(result["stage_qualified"], result)
        self.assertFalse(result["authorization"]["short_tp4_authorized"])

    def test_qualification_requires_full_frozen_matrix(self):
        buckets = (24576, 57344, 122880)
        ordinals = (0, 4, 9)
        result = MODULE.qualify(
            [
                report(rank, buckets=buckets, ordinals=ordinals)
                for rank in range(4)
            ],
            CONTRACT,
            profile="qualification",
        )
        self.assertTrue(result["stage_qualified"], result)
        self.assertTrue(result["authorization"]["short_tp4_authorized"])
        incomplete = MODULE.qualify(
            [report(rank) for rank in range(4)],
            CONTRACT,
            profile="qualification",
        )
        self.assertFalse(incomplete["stage_qualified"])
        self.assertTrue(incomplete["coverage_reasons"])

    def test_numeric_failure_cannot_be_waived_by_speed(self):
        reports = [report(rank, speedup=3.0) for rank in range(4)]
        reports[2]["records"][0]["numeric"]["qualified"] = False
        reports[2]["all_numeric_qualified"] = False
        result = MODULE.qualify(
            reports, CONTRACT, profile="smoke")
        self.assertFalse(result["stage_qualified"])
        self.assertFalse(result["execution_valid"])


if __name__ == "__main__":
    unittest.main()
