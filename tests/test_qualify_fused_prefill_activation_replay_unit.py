from __future__ import annotations

import hashlib
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
NUMERIC_CONTRACT = json.loads(
    (
        ROOT / "quality" / "fused_prefill_numeric_adjudication.v1.json"
    ).read_text())


def report(rank: int, buckets=(24576,), ordinals=(0,), speedup=1.5):
    records = []
    for bucket in buckets:
        for ordinal in ordinals:
            reference_trials = [3.0, 3.1, 3.2]
            candidate_ms = 3.1 / speedup
            records.append({
                "rank": rank,
                "bucket_min_context_tokens": bucket,
                "call_ordinal": ordinal,
                "context_tokens": bucket,
                "query_length": 8176,
                "case_sha256": hashlib.sha256(
                    f"{rank}:{bucket}:{ordinal}".encode("ascii")
                ).hexdigest(),
                "load_elapsed_s": 0.5,
                "reference_timing": {
                    "warmups": 1,
                    "trials": 3,
                    "cuda_trials_ms": reference_trials,
                    "cuda_median_ms": 3.1,
                },
                "candidate_timing": {
                    "warmups": 1,
                    "trials": 3,
                    "cuda_trials_ms": [candidate_ms] * 3,
                    "cuda_median_ms": candidate_ms,
                },
                "candidate_speedup": speedup,
                "numeric": {
                    "candidate_finite": True,
                    "reference_finite": True,
                    "finite": True,
                    "candidate_vs_rounded_relative_l2": 7.0e-6,
                    "candidate_vs_rounded_max_abs_diagnostic": 0.001953125,
                    "candidate_to_fp32_relative_l2": 5.0e-4,
                    "candidate_to_fp32_max_abs": 0.0015,
                    "rounded_to_fp32_relative_l2": 3.0e-4,
                    "rounded_to_fp32_max_abs": 0.0008,
                    "candidate_lse_finite": True,
                    "reference_lse_finite": True,
                    "lse_finite": True,
                    "lse_relative_l2": 1.0e-8,
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
        "visible_physical_gpu": rank,
        "rank": rank,
        "device_name": "Iluvatar BI-V100",
        "torch_version": "2.1.0",
        "bank": {
            "manifest": f"/tmp/rank-{rank}.manifest.json",
            "manifest_sha256": hashlib.sha256(
                f"manifest:{rank}".encode("ascii")
            ).hexdigest(),
            "run_id": "capture",
            "record_count": len(records),
        },
        "all_numeric_qualified": True,
        "candidate_extension": {
            "path": "/tmp/candidate.so",
            "sha256": "b" * 64,
            "size_bytes": 1024,
        },
        "records": records,
        "privacy": dict(MODULE.PRIVACY_CONTRACT),
        "authorization": dict(MODULE.AUTHORIZATION_CONTRACT),
    }


class QualifyActivationReplayTest(unittest.TestCase):

    def test_smoke_profile_passes_four_rank_subset_without_authorizing_tp4(self):
        result = MODULE.qualify(
            [report(rank) for rank in range(4)],
            CONTRACT,
            NUMERIC_CONTRACT,
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
            NUMERIC_CONTRACT,
            profile="qualification",
        )
        self.assertTrue(result["stage_qualified"], result)
        self.assertTrue(result["authorization"]["short_tp4_authorized"])
        self.assertEqual(
            result["candidate_extension"],
            {"sha256": "b" * 64, "size_bytes": 1024},
        )
        self.assertEqual(result["capture_source_revision"], "a" * 40)
        self.assertEqual(result["candidate_source_revision"], "c" * 40)
        self.assertEqual(result["activation_run_id"], "capture")
        self.assertEqual(len(result["bank_manifest_sha256s"]), 4)
        incomplete = MODULE.qualify(
            [report(rank) for rank in range(4)],
            CONTRACT,
            NUMERIC_CONTRACT,
            profile="qualification",
        )
        self.assertFalse(incomplete["stage_qualified"])
        self.assertTrue(incomplete["coverage_reasons"])

    def test_numeric_failure_cannot_be_waived_by_speed(self):
        reports = [report(rank, speedup=3.0) for rank in range(4)]
        numeric = reports[2]["records"][0]["numeric"]
        numeric["candidate_to_fp32_max_abs"] = 0.0017
        numeric["qualified"] = False
        reports[2]["all_numeric_qualified"] = False
        result = MODULE.qualify(
            reports, CONTRACT, NUMERIC_CONTRACT, profile="smoke")
        self.assertFalse(result["stage_qualified"])
        self.assertFalse(result["execution_valid"])
        self.assertTrue(result["numeric_reasons"])

    def test_reported_boolean_cannot_mask_numeric_failure(self):
        reports = [report(rank, speedup=3.0) for rank in range(4)]
        reports[1]["records"][0]["numeric"][
            "candidate_to_fp32_max_abs"] = 0.0017
        result = MODULE.qualify(
            reports, CONTRACT, NUMERIC_CONTRACT, profile="smoke")
        self.assertFalse(result["stage_qualified"])
        self.assertTrue(result["numeric_reasons"])
        self.assertTrue(any(
            "reported numeric qualification is inconsistent" in reason
            for reason in result["invalid_reasons"]
        ))

    def test_timing_and_duplicate_evidence_fail_closed(self):
        reports = [report(rank) for rank in range(4)]
        reports[0]["records"][0]["candidate_speedup"] = 99.0
        duplicate = dict(reports[1]["records"][0])
        reports[1]["records"].append(duplicate)
        reports[1]["bank"]["record_count"] += 1
        result = MODULE.qualify(
            reports, CONTRACT, NUMERIC_CONTRACT, profile="smoke")
        self.assertFalse(result["stage_qualified"])
        self.assertTrue(any(
            "speedup is inconsistent" in reason
            for reason in result["invalid_reasons"]
        ))
        self.assertTrue(any(
            "duplicate matrix cell" in reason
            for reason in result["coverage_reasons"]
        ))

    def test_malformed_contracts_fail_without_exception(self):
        malformed = (
            (None, None),
            ({"schema": MODULE.CONTRACT_SCHEMA, "version": 1,
              "stages": [{"id": "L2", "capture": [],
                          "continuation_screen": "invalid"}]}, {
                              "schema": MODULE.NUMERIC_CONTRACT_SCHEMA,
                              "version": 1,
                              "execution": [],
                              "promotion": [],
                          }),
        )
        for contract, numeric_contract in malformed:
            with self.subTest(contract=contract):
                result = MODULE.qualify(
                    [report(rank) for rank in range(4)],
                    contract,
                    numeric_contract,
                    profile="qualification",
                )
                self.assertFalse(result["stage_qualified"])
                self.assertTrue(result["invalid_reasons"])

    def test_bank_and_case_identity_must_be_unique(self):
        reports = [report(rank) for rank in range(4)]
        reports[3]["bank"]["manifest_sha256"] = reports[2]["bank"][
            "manifest_sha256"]
        reports[1]["records"][0]["case_sha256"] = reports[0]["records"][0][
            "case_sha256"]
        result = MODULE.qualify(
            reports, CONTRACT, NUMERIC_CONTRACT, profile="smoke")
        self.assertFalse(result["stage_qualified"])
        self.assertTrue(any(
            "four distinct bank manifests" in reason
            for reason in result["invalid_reasons"]
        ))
        self.assertTrue(any(
            "duplicate activation case" in reason
            for reason in result["coverage_reasons"]
        ))

    def test_shape_and_gpu_boolean_values_fail_closed(self):
        reports = [report(rank) for rank in range(4)]
        reports[1]["visible_physical_gpu"] = True
        reports[2]["records"][0]["context_tokens"] = 262144
        result = MODULE.qualify(
            reports, CONTRACT, NUMERIC_CONTRACT, profile="smoke")
        self.assertFalse(result["stage_qualified"])
        self.assertTrue(result["invalid_reasons"])

    def test_runner_binds_both_frozen_contracts(self):
        source = (
            ROOT / "scripts" / "run_m1_140_activation_replay.sh"
        ).read_text(encoding="ascii")
        self.assertIn("--numeric-contract", source)
        self.assertIn(
            "fused_prefill_numeric_adjudication.v1.json",
            source,
        )
        self.assertIn("trap '' INT TERM", source)


if __name__ == "__main__":
    unittest.main()
