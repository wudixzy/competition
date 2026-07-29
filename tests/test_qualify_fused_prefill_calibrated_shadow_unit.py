from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from tests import qualify_fused_prefill_calibrated_shadow as M


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT / "quality/fused_prefill_numeric_adjudication.v1.json")
CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
RUN_ID = "m1-138-unit"
METRIC_NAMES = (
    "relative_l2",
    "max_abs",
    "candidate_to_fp32_relative_l2",
    "candidate_to_fp32_max_abs",
    "rounded_to_fp32_relative_l2",
    "rounded_to_fp32_max_abs",
    "relative_l2_baseline_ratio",
    "max_abs_baseline_ratio",
)


def record(index: int, bucket: int, context_tokens: int) -> dict:
    candidate_relative = 5.0e-4
    rounded_relative = 3.0e-4
    candidate_max = 0.0015
    rounded_max = 0.0008
    return {
        "index": index,
        "status": "pass",
        "bucket_min_context_tokens": bucket,
        "context_tokens": context_tokens,
        "query_shape": [8176, 4, 256],
        "query_heads": 4,
        "kv_heads": 1,
        "head_dim": 256,
        "block_size": 16,
        "candidate_finite": True,
        "reference_finite": True,
        "relative_l2": 7.1e-6,
        "max_abs": 0.001953125,
        "candidate_to_fp32_relative_l2": candidate_relative,
        "candidate_to_fp32_max_abs": candidate_max,
        "rounded_to_fp32_relative_l2": rounded_relative,
        "rounded_to_fp32_max_abs": rounded_max,
        "relative_l2_baseline_ratio": (
            candidate_relative / rounded_relative),
        "max_abs_baseline_ratio": candidate_max / rounded_max,
        "error_stage": None,
        "error_type": None,
    }


def observations(records: list[dict]) -> dict:
    completed = [
        item for item in records
        if item["status"] in {"pass", "fail", "invalid"}
    ]
    value = {
        "expected": 4,
        "reserved": len(records),
        "completed": len(completed),
        "passed": sum(item["status"] == "pass" for item in records),
        "failed": sum(item["status"] == "fail" for item in records),
        "invalid": sum(item["status"] == "invalid" for item in records),
        "pending": sum(item["status"] == "pending" for item in records),
    }
    for name in METRIC_NAMES:
        metrics = [
            item[name] for item in completed
            if isinstance(item.get(name), float)
        ]
        value[f"maximum_{name}"] = max(metrics) if metrics else None
    return value


def refresh_report(value: dict) -> None:
    records = value["records"]
    value["observations"] = observations(records)
    if any(item["status"] == "invalid" for item in records):
        value["status"] = "invalid"
    elif any(item["status"] == "fail" for item in records):
        value["status"] = "fail"
    elif len(records) == 4 and all(
        item["status"] == "pass" for item in records
    ):
        value["status"] = "pass"
    else:
        value["status"] = "collecting"


def report(rank: int) -> dict:
    records = [
        record(0, 49152, 49152),
        record(1, 49152, 49152),
        record(2, 114688, 114688),
        record(3, 114688, 114688),
    ]
    return {
        "schema": M.REPORT_SCHEMA,
        "version": 1,
        "run_id": RUN_ID,
        "pid": 100 + rank,
        "rank": rank,
        "status": "pass",
        "selection": {
            "minimum_context_tokens": [49152, 114688],
            "max_calls_per_context": 2,
        },
        "thresholds": {
            "require_finite_candidate": True,
            "require_finite_reference": True,
            "maximum_candidate_vs_rounded_relative_l2": 1.0e-5,
            "maximum_error_multiple_over_fp16_rounding": 2.0,
            "ratio_denominator_floor": 1.0e-12,
            "fixed_max_abs_role": "diagnostic_only",
            "finite_failure_action": "record",
        },
        "observations": observations(records),
        "records": records,
        "privacy": {
            "contains_prompts": False,
            "contains_model_outputs": False,
            "contains_tensor_values": False,
            "contains_token_ids": False,
            "contains_credentials": False,
        },
    }


def qualify(reports: list[dict]) -> dict:
    return M.qualify(
        reports,
        CONTRACT,
        run_id=RUN_ID,
        source_revision="a" * 40,
        runtime_identity="bare-host-overlay-v1:" + "b" * 20,
    )


class CalibratedFusedPrefillShadowQualificationTest(unittest.TestCase):

    def test_contract_digest_is_frozen(self) -> None:
        self.assertEqual(M.sha256(CONTRACT_PATH), M.CONTRACT_SHA256)

    def test_scale_aware_records_qualify_despite_fixed_max_abs(self) -> None:
        value = qualify([report(rank) for rank in range(4)])
        self.assertEqual(value["status"], "pass", value)
        self.assertTrue(value["operator_surface_authorized"])
        self.assertEqual(value["observation_count"], 16)
        self.assertEqual(value["maxima"]["max_abs"], 0.001953125)
        self.assertFalse(value["capability_evaluated"])
        self.assertFalse(value["performance_evaluated"])
        self.assertFalse(value["production_promotion_authorized"])

    def test_error_over_twice_rounding_baseline_fails(self) -> None:
        reports = [report(rank) for rank in range(4)]
        changed = reports[2]["records"][1]
        changed["candidate_to_fp32_max_abs"] = 0.0017
        changed["max_abs_baseline_ratio"] = 0.0017 / 0.0008
        changed["status"] = "fail"
        refresh_report(reports[2])
        value = qualify(reports)
        self.assertEqual(value["status"], "fail", value)
        self.assertFalse(value["operator_surface_authorized"])
        self.assertTrue(value["numeric_failures"])

    def test_inconsistent_ratio_is_invalid(self) -> None:
        reports = [report(rank) for rank in range(4)]
        reports[0]["records"][0]["max_abs_baseline_ratio"] = 1.0
        refresh_report(reports[0])
        value = qualify(reports)
        self.assertEqual(value["status"], "invalid", value)
        self.assertTrue(any(
            "ratio is inconsistent" in reason
            for reason in value["invalid_reasons"]
        ))

    def test_missing_rank_is_invalid(self) -> None:
        value = qualify([report(rank) for rank in range(3)])
        self.assertEqual(value["status"], "invalid", value)
        self.assertIn("missing rank reports: [3]", value["invalid_reasons"])

    def test_contract_threshold_change_is_invalid(self) -> None:
        changed = copy.deepcopy(CONTRACT)
        changed["hard_gates"][
            "maximum_error_multiple_over_fp16_rounding"] = 3.0
        value = M.qualify(
            [report(rank) for rank in range(4)],
            changed,
            run_id=RUN_ID,
            source_revision="a" * 40,
            runtime_identity="runtime",
        )
        self.assertEqual(value["status"], "invalid", value)

    def test_contract_extra_field_is_invalid(self) -> None:
        changed = copy.deepcopy(CONTRACT)
        changed["candidate_observation"] = "post-hoc"
        value = M.qualify(
            [report(rank) for rank in range(4)],
            changed,
            run_id=RUN_ID,
            source_revision="a" * 40,
            runtime_identity="runtime",
        )
        self.assertEqual(value["status"], "invalid", value)

    def test_observation_summary_tamper_is_invalid(self) -> None:
        reports = [report(rank) for rank in range(4)]
        reports[1]["observations"]["passed"] = 3
        value = qualify(reports)
        self.assertEqual(value["status"], "invalid", value)
        self.assertIn(
            "rank 1: observations are inconsistent",
            value["invalid_reasons"],
        )

    def test_malformed_evidence_outranks_numeric_failure(self) -> None:
        reports = [report(rank) for rank in range(4)]
        changed = reports[0]["records"][0]
        changed["candidate_to_fp32_max_abs"] = 0.0017
        changed["max_abs_baseline_ratio"] = 0.0017 / 0.0008
        changed["status"] = "fail"
        refresh_report(reports[0])
        reports[1]["observations"]["passed"] = 3
        value = qualify(reports)
        self.assertEqual(value["status"], "invalid", value)
        self.assertTrue(value["numeric_failures"])

    def test_malformed_contract_fails_closed_without_exception(self) -> None:
        value = M.qualify(
            [report(rank) for rank in range(4)],
            None,
            run_id=RUN_ID,
            source_revision="a" * 40,
            runtime_identity="runtime",
        )
        self.assertEqual(value["status"], "invalid", value)
        self.assertIn(
            "numeric adjudication contract identity is invalid",
            value["invalid_reasons"],
        )


if __name__ == "__main__":
    unittest.main()
