from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests" / "qualify_selected_dataset_trace_smoke.py"
SPEC = importlib.util.spec_from_file_location("trace_smoke_qualifier", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _record(ordinal: int) -> dict:
    first = hashlib.sha256(f"first-{ordinal}".encode()).digest()
    second = hashlib.sha256(first + b"second").digest()
    return {
        "version": 4,
        "trace_session_sha256": "0123456789abcdef",
        "ordinal": ordinal,
        "request_id_sha256": f"{ordinal:016x}",
        "prompt_tokens": 16,
        "prompt_allocated_blocks": 1,
        "block_size": 16,
        "capacity_blocks": 100,
        "gdn_policy": "admission64",
        "raw_kv_contiguous_hit_blocks": 0,
        "effective_gdn_hit_blocks": 0,
        "gdn_admissions": [],
        "gdn_evictions": [],
        "ttft_s": 1.0,
        "request_latency_s": 2.0,
        "time_in_queue_s": 0.1,
        "observed_effective_cached_tokens": 0,
        "total_tokens": 32,
        "allocated_blocks": 2,
        "full_blocks": 2,
        "hash_encoding": "sha256_base64",
        "block_hashes": base64.b64encode(first + second).decode("ascii"),
        "generated_tokens": 16,
        "observed_input_tps": 16.0,
        "observed_output_tps": 15.0,
        "_hashes": [first, second],
        "_prompt_full_blocks": 1,
    }


def _sources() -> tuple[list[dict], dict, dict, bytes, bytes, dict[str, str]]:
    records = [_record(ordinal) for ordinal in range(1, 14)]
    serializable = [
        {key: value for key, value in record.items() if not key.startswith("_")}
        for record in records
    ]
    log_bytes = b"".join(
        MODULE.TRACE_MARKER
        + json.dumps(record, sort_keys=True).encode("ascii")
        + b"\n"
        for record in serializable
    )
    timing = {
        "per_request_timing_projection_complete": True,
        "projected_ttft_p90_s": 1.0,
        "projected_sequential_wall_s": 26.0,
    }
    analysis = {
        "requests": 13,
        "trace_version": 4,
        "qualification_trace": False,
        "trace_session_sha256": "0123456789abcdef",
        "trace_ordinals": {"first": 1, "last": 13, "contiguous": True},
        "prompt_tokens": 208,
        "generated_tokens": 208,
        "source_logs": [{"sha256": hashlib.sha256(log_bytes).hexdigest()}],
        "policy_metrics": {
            name: dict(timing) for name in MODULE.POLICY_NAMES
        },
        "control_policy_metrics": {
            name: {} for name in MODULE.CONTROL_POLICY_NAMES
        },
    }
    replay = {
        "schema": "bi100-selected-dataset-replay-v1",
        "dataset": {
            "sha256": MODULE.EXPECTED_DATASET_SHA256,
            "turn_count": 13,
        },
        "validation": {"complete_replay": True, "all_successful": True},
        "aggregate": {
            "prompt_tokens": 208,
            "cached_tokens": 0,
            "completion_tokens": 208,
        },
    }
    dataset_bytes = (ROOT / "chat_dataset_v0.json").read_bytes()
    source_names = {"analysis": "a" * 64, "replay": "b" * 64}
    return records, analysis, replay, log_bytes, dataset_bytes, source_names


class SelectedDatasetTraceSmokeQualificationTest(unittest.TestCase):

    def test_fixed_live_trace_smoke_qualifies_without_881_authority(self):
        records, analysis, replay, log, dataset, names = _sources()
        report = MODULE.qualify(
            records=records,
            analysis=analysis,
            replay=replay,
            log_bytes=log,
            dataset_bytes=dataset,
            source_names=names,
        )
        self.assertTrue(report["qualified"], report)
        self.assertFalse(report["qualification_authorized"])
        self.assertEqual(report["trace"]["requests"], 13)
        self.assertTrue(report["trace"]["all_policy_timing_complete"])
        self.assertTrue(report["privacy"]["known_dataset_text_absent"])
        self.assertGreater(report["size"]["linear_881_request_estimate_bytes"], 0)

    def test_unknown_trace_field_fails_privacy_allowlist(self):
        records, analysis, replay, log, dataset, names = _sources()
        records[0]["messages"] = [{"role": "user", "content": "forbidden"}]
        report = MODULE.qualify(
            records=records,
            analysis=analysis,
            replay=replay,
            log_bytes=log,
            dataset_bytes=dataset,
            source_names=names,
        )
        self.assertFalse(report["qualified"])
        self.assertTrue(any(
            "fields unexpected" in reason for reason in report["reasons"]))

    def test_known_dataset_text_in_log_fails(self):
        records, analysis, replay, log, dataset, names = _sources()
        first_question = json.loads(dataset)[0]["user_questions"][0]
        report = MODULE.qualify(
            records=records,
            analysis=analysis,
            replay=replay,
            log_bytes=log + first_question.encode("utf-8"),
            dataset_bytes=dataset,
            source_names=names,
        )
        self.assertFalse(report["qualified"])
        self.assertFalse(report["privacy"]["known_dataset_text_absent"])

    def test_13_request_analysis_cannot_claim_qualification(self):
        records, analysis, replay, log, dataset, names = _sources()
        analysis = copy.deepcopy(analysis)
        analysis["qualification_trace"] = True
        analysis["qualification"] = {"admission64": {"ok": True}}
        report = MODULE.qualify(
            records=records,
            analysis=analysis,
            replay=replay,
            log_bytes=log,
            dataset_bytes=dataset,
            source_names=names,
        )
        self.assertFalse(report["qualified"])
        self.assertFalse(report["qualification_authorized"])


if __name__ == "__main__":
    unittest.main()
