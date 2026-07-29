from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from tests import qualify_fused_prefill_shadow as qualification


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads(
    (ROOT / "quality/layered_quality_gate.v2.json").read_text(
        encoding="utf-8"))


def _record(index: int, bucket: int) -> dict:
    return {
        "index": index,
        "status": "pass",
        "bucket_min_context_tokens": bucket,
        "context_tokens": bucket + 8192,
        "query_shape": [8176, 4, 256],
        "query_heads": 4,
        "kv_heads": 1,
        "head_dim": 256,
        "block_size": 16,
        "candidate_finite": True,
        "reference_finite": True,
        "relative_l2": 6.0e-6,
        "max_abs": 0.00048828125,
        "error_stage": None,
        "error_type": None,
    }


def _report(rank: int) -> dict:
    records = [
        _record(0, 49152),
        _record(1, 49152),
        _record(2, 114688),
        _record(3, 114688),
    ]
    return {
        "schema": qualification.REPORT_SCHEMA,
        "version": 1,
        "run_id": "m1-136-unit",
        "pid": 1000 + rank,
        "rank": rank,
        "status": "pass",
        "selection": {
            "minimum_context_tokens": [49152, 114688],
            "max_calls_per_context": 2,
        },
        "thresholds": {
            "require_finite_candidate": True,
            "require_finite_reference": True,
            "maximum_relative_l2": 1.0e-5,
            "maximum_absolute_error": 0.001,
        },
        "observations": {
            "expected": 4,
            "reserved": 4,
            "completed": 4,
            "passed": 4,
            "failed": 0,
            "invalid": 0,
            "pending": 0,
            "maximum_relative_l2": 6.0e-6,
            "maximum_absolute_error": 0.00048828125,
        },
        "records": records,
        "privacy": {
            "contains_prompts": False,
            "contains_model_outputs": False,
            "contains_tensor_values": False,
            "contains_token_ids": False,
            "contains_credentials": False,
        },
    }


def _qualify(reports: list[dict]) -> dict:
    return qualification.qualify(
        reports,
        CONTRACT,
        run_id="m1-136-unit",
        source_revision="a" * 40,
        runtime_identity="bare-host-overlay-v1:" + "b" * 20,
    )


class FusedPrefillShadowQualificationTests(unittest.TestCase):

    def test_complete_tp4_real_activation_matrix_passes(self) -> None:
        result = _qualify([_report(rank) for rank in range(4)])
        self.assertEqual(result["status"], "pass", result)
        self.assertEqual(result["rank_count"], 4)
        self.assertEqual(result["observation_count"], 16)
        self.assertFalse(result["promotion_authorized"])

    def test_missing_rank_is_invalid(self) -> None:
        result = _qualify([_report(rank) for rank in range(3)])
        self.assertEqual(result["status"], "invalid")
        self.assertTrue(any(
            "missing rank" in reason for reason in result["invalid_reasons"]))

    def test_numeric_limit_is_hard_failure(self) -> None:
        reports = [_report(rank) for rank in range(4)]
        reports[2]["records"][1]["relative_l2"] = 1.01e-5
        reports[2]["records"][1]["status"] = "fail"
        reports[2]["status"] = "fail"
        result = _qualify(reports)
        self.assertEqual(result["status"], "fail")
        self.assertTrue(result["numeric_failures"])

    def test_reference_nonfinite_is_not_semantically_waivable(self) -> None:
        reports = [_report(rank) for rank in range(4)]
        record = reports[0]["records"][0]
        record["status"] = "invalid"
        record["reference_finite"] = False
        record["relative_l2"] = None
        record["max_abs"] = None
        reports[0]["status"] = "invalid"
        result = _qualify(reports)
        self.assertEqual(result["status"], "invalid")
        self.assertFalse(result["qualified"])

    def test_negative_error_metric_is_invalid(self) -> None:
        reports = [_report(rank) for rank in range(4)]
        reports[1]["records"][0]["relative_l2"] = -1.0e-6
        result = _qualify(reports)
        self.assertEqual(result["status"], "invalid")

    def test_high_context_cannot_fill_lower_bucket(self) -> None:
        reports = [_report(rank) for rank in range(4)]
        reports[0]["records"][0]["context_tokens"] = 122880
        result = _qualify(reports)
        self.assertEqual(result["status"], "invalid")

    def test_boolean_rank_is_invalid(self) -> None:
        reports = [_report(rank) for rank in range(4)]
        reports[1]["rank"] = True
        result = _qualify(reports)
        self.assertEqual(result["status"], "invalid")

    def test_input_reports_are_not_mutated(self) -> None:
        reports = [_report(rank) for rank in range(4)]
        before = copy.deepcopy(reports)
        _qualify(reports)
        self.assertEqual(reports, before)


if __name__ == "__main__":
    unittest.main()
