import base64
import hashlib
import json
import pathlib
import tempfile
import unittest
from unittest import mock
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "scripts"))
import analyze_prefix_cache_trace as sim


def runtime_contract_value():
    contract = sim.baseline_contract.runtime_contract
    model_path = "/model"
    environment = contract.service_environment(
        "/runtime/site-packages",
        gdn_cache_policy="fine32",
        gdn_restore_mode="direct",
        fused_prefill="0",
        kv_eviction_policy="lru",
    )
    return {
        "schema": "bi100-quality-runtime-contract-v1",
        "version": 1,
        "source_revision": "a" * 40,
        "runtime_identity": "bare-host-overlay-v1:" + "b" * 20,
        "runtime_overlay_sha256": "b" * 64,
        "instance": "unit-instance",
        "gpu_count": 4,
        "tensor_parallel_size": 4,
        "max_model_len": 262144,
        "model_path": model_path,
        "tokenizer_path": model_path,
        "served_model_name": "llm",
        "base_image": contract.BASE_IMAGE,
        "command": contract.service_command(model_path),
        "environment": environment,
        "cache_trace_enabled": True,
        "optimization_label": "fine32-direct-lru-control",
    }


def attested_baseline(records, trace_path):
    contract = sim.baseline_contract
    trace = contract.trace_identity(records, [trace_path])
    runtime = runtime_contract_value()
    runtime_sha = contract.validate_runtime_contract(runtime)
    workload = {
        "schema": contract.WORKLOAD_SCHEMA,
        "version": 1,
        "workload_kind": "restricted_official_881",
        "name": "unit official workload",
        "author_or_org": "unit operator",
        "source_url": None,
        "license": "restricted",
        "revision": "unit-run-1",
        "captured_at_utc": "2026-07-25T00:00:00Z",
        "split": "all",
        "request_count": 881,
        "request_order_sha256": trace["request_order_sha256"],
        "source_artifact_sha256": trace["records_sha256"],
        "source_artifact_kind": "privacy_safe_cache_trace_v4_records",
        "selection_rule": "all requests in fixed order",
        "transformation": "privacy-safe cache trace v4",
        "redistribution_allowed": False,
        "contains_restricted_evaluation_data": True,
        "snapshot_redistributed": False,
    }
    workload_sha = contract.validate_workload_manifest(
        workload, expected_trace=trace)
    metrics = {
        "score_kind": "local_881_proxy",
        "aggregation": "fixed-order sequential wall-clock aggregate",
        "attempted_requests": 881,
        "successful_requests": 881,
        "error_requests": 0,
        "output_tps_p10": 21.0,
        "input_tps": 2800.0,
        "cache_tps": 100.0,
        "ttft_p90_s": 4.0,
        "cache_hit_rate": 0.5,
        "success_rate": 1.0,
        "weighted_score": 0.0,
        "formula": contract.SCORE_FORMULA,
    }
    metrics["weighted_score"] = contract.weighted_score(metrics)
    value = {
        "schema": contract.BASELINE_SCHEMA,
        "version": 1,
        "run_id": "unit-run-1",
        "runtime_contract": {
            "sha256": runtime_sha,
            "file_sha256": "c" * 64,
            "value": runtime,
        },
        "workload_manifest": {
            "sha256": workload_sha,
            "file_sha256": "d" * 64,
            "value": workload,
        },
        "trace": trace,
        "metrics": metrics,
        "metrics_source": {"bytes": 2, "sha256": "e" * 64},
        "metrics_transformation": "unit metrics copied by exact field name",
        "attestation": {
            "trace_metrics_same_service_run_asserted": True,
            "metrics_cover_exact_trace_request_order_asserted": True,
            "contains_raw_requests_or_outputs": False,
            "qualification_scope": "offline_cache_phase_gate_only",
        },
    }
    contract.validate_baseline_contract(value, expected_trace=trace)
    return value


def digest(value: int) -> bytes:
    return hashlib.sha256(value.to_bytes(4, "big")).digest()


def record(values, capacity=4, block_size=16, request_id=0,
           session="0123456789abcdef", ordinal=1, prompt_tokens=None,
           total_tokens=None):
    raw = b"".join(digest(value) for value in values)
    if total_tokens is None:
        total_tokens = len(values) * block_size
    if prompt_tokens is None:
        prompt_tokens = total_tokens
    return {
        "version": 4,
        "trace_session_sha256": session,
        "ordinal": ordinal,
        "request_id_sha256": f"{request_id:016x}",
        "prompt_tokens": prompt_tokens,
        "prompt_allocated_blocks": (prompt_tokens + block_size - 1) // block_size,
        "total_tokens": total_tokens,
        "allocated_blocks": (total_tokens + block_size - 1) // block_size,
        "block_size": block_size,
        "capacity_blocks": capacity,
        "full_blocks": len(values),
        "hash_encoding": "sha256_base64",
        "block_hashes": base64.b64encode(raw).decode("ascii"),
    }


def decoded(values, capacity=4, ordinal=1, prompt_tokens=None, total_tokens=None):
    item = record(values, capacity=capacity, ordinal=ordinal,
                  prompt_tokens=prompt_tokens, total_tokens=total_tokens)
    item["_hashes"] = [digest(value) for value in values]
    item["_prompt_full_blocks"] = item["prompt_tokens"] // item["block_size"]
    return item


class AnalyzerTest(unittest.TestCase):
    def test_heap_eviction_matches_scan_order_with_stale_entries(self):
        blocks = [digest(value) for value in range(1, 9)]
        for candidate in (False, True):
            cache = set(blocks)
            last = {
                block: (index // 3, index % 3)
                for index, block in enumerate(blocks)
            }
            frequency = {
                block: (index * 7) % 5
                for index, block in enumerate(blocks)
            }
            heap = []
            for block in blocks:
                sim._index_eviction_candidate(
                    heap, block, last, frequency, candidate)

            # Leave one stale entry in the heap and index its new state.
            last[blocks[2]] = (9, 4)
            frequency[blocks[2]] = 11
            sim._index_eviction_candidate(
                heap, blocks[2], last, frequency, candidate)

            while cache:
                if candidate:
                    expected = min(cache, key=lambda block: (
                        frequency[block], last[block][0],
                        -last[block][1], block))
                else:
                    expected = min(cache, key=lambda block: (
                        last[block][0], -last[block][1], block))
                self.assertEqual(
                    sim._evict(cache, last, frequency, candidate, heap),
                    expected)

    def test_version_4_and_hash_encoding_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            bad_version = record([1, 2], request_id=1)
            bad_version["version"] = 3
            log = root / "trace.log"
            log.write_text(sim.MARKER + json.dumps(bad_version) + "\n")
            with self.assertRaisesRegex(ValueError, "unsupported trace version"):
                sim.read([str(log)])

    def test_malformed_hash_is_rejected(self):
        bad = record([1])
        bad["block_hashes"] = "not-base64%%"
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            log = root / "trace.log"
            log.write_text(sim.MARKER + json.dumps(bad) + "\n")
            with self.assertRaisesRegex(ValueError, "invalid base64 block_hashes"):
                sim.read([str(log)])

        wrong_len = record([1])
        wrong_len["full_blocks"] = 2
        wrong_len["total_tokens"] = 32
        wrong_len["allocated_blocks"] = 2
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            log = root / "trace.log"
            log.write_text(sim.MARKER + json.dumps(wrong_len) + "\n")
            with self.assertRaisesRegex(ValueError, "block_hashes length"):
                sim.read([str(log)])

    def test_request_id_and_ordinal_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            duplicate = root / "dup.log"
            duplicate.write_text("\n".join([
                sim.MARKER + json.dumps(record([1], request_id=1)),
                sim.MARKER + json.dumps(record([1], request_id=1, ordinal=2)),
            ]) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                sim.read([str(duplicate)])

            mixed = root / "mixed.log"
            mixed.write_text("\n".join([
                sim.MARKER + json.dumps(record([1], request_id=1,
                                                session="0123456789abcdef", ordinal=1)),
                sim.MARKER + json.dumps(record([1], request_id=2,
                                                session="fedcba9876543210", ordinal=2)),
            ]) + "\n")
            with self.assertRaisesRegex(ValueError, "multiple runtime sessions"):
                sim.read([str(mixed)])

            missing = root / "missing.log"
            missing.write_text("\n".join([
                sim.MARKER + json.dumps(record([1], request_id=1, ordinal=1)),
                sim.MARKER + json.dumps(record([1], request_id=3, ordinal=3)),
            ]) + "\n")
            with self.assertRaisesRegex(ValueError, "contiguous"):
                sim.read([str(missing)])

            non_hex = record([1], request_id=1)
            non_hex["request_id_sha256"] = "not-a-hex-string"
            with tempfile.TemporaryDirectory() as invalid:
                bad = pathlib.Path(invalid) / "bad.log"
                bad.write_text(sim.MARKER + json.dumps(non_hex) + "\n")
                with self.assertRaisesRegex(ValueError, "request_id_sha256"):
                    sim.read([str(bad)])

    def test_policy_raw_and_gdn_counters(self):
        records = [
            decoded([1, 2], capacity=2, ordinal=1),
            decoded([1, 2], capacity=2, ordinal=2),
            decoded([1, 2], capacity=2, ordinal=3),
        ]
        off = sim.simulate(records, 2, policy="off")
        fine = sim.simulate(records, 2, policy="fine32")
        admission = sim.simulate(records, 2, policy="admission64")

        self.assertEqual(off["raw_kv_contiguous_hit_tokens"],
                         fine["raw_kv_contiguous_hit_tokens"])
        self.assertEqual(off["raw_kv_contiguous_hit_tokens"],
                         admission["raw_kv_contiguous_hit_tokens"])
        self.assertEqual(off["raw_kv_contiguous_hit_tokens"], 32)
        self.assertEqual(fine["usable_gdn_state_avoided_tokens"], 32)
        self.assertEqual(admission["usable_gdn_state_avoided_tokens"], 32)
        self.assertEqual(admission["residual_prefill_tokens"], 64)
        self.assertEqual(len(admission["request_results"]), 3)

    def test_tail64_restores_first_sibling_at_previous_chunk_boundary(self):
        records = [
            decoded(list(range(1, 9)), capacity=16, ordinal=1,
                    prompt_tokens=128),
            decoded([1, 2, 3, 4, 9, 10, 11, 12], capacity=16, ordinal=2,
                    prompt_tokens=128),
        ]

        admission = sim.simulate(
            records, 16, policy="admission64", gdn_chunk_tokens=64)
        tail = sim.simulate(
            records, 16, policy="tail64", gdn_chunk_tokens=64)

        self.assertEqual(
            admission["request_results"][1]["effective_hit_tokens"], 0)
        self.assertEqual(
            admission["request_results"][1]["residual_prefill_tokens"], 128)
        self.assertEqual(
            tail["request_results"][1]["effective_hit_tokens"], 64)
        self.assertEqual(
            tail["request_results"][1]["residual_prefill_tokens"], 64)
        self.assertLessEqual(tail["gdn_policy_cache_size"], 64)

    def test_gdn_state_without_live_kv_cannot_avoid_tokens(self):
        records = [
            decoded([1, 2], capacity=2, ordinal=1),
            decoded([3, 4], capacity=2, ordinal=2),
            decoded([1, 2], capacity=2, ordinal=3),
        ]
        fine = sim.simulate(records, 2, policy="fine32")
        admission = sim.simulate(records, 2, policy="admission64")
        self.assertEqual(fine["raw_kv_contiguous_hit_tokens"], 0)
        self.assertEqual(fine["usable_gdn_state_avoided_tokens"], 0)
        self.assertEqual(admission["usable_gdn_state_avoided_tokens"], 0)
        self.assertEqual(fine["combined_hit_tokens"], 0)

    def test_fine32_does_not_capture_a_boundary_with_one_replay_token(self):
        records = [
            decoded([1], capacity=4, ordinal=1,
                    prompt_tokens=17, total_tokens=17),
            decoded([1], capacity=4, ordinal=2,
                    prompt_tokens=17, total_tokens=17),
        ]

        result = sim.simulate(records, 4, policy="fine32")

        self.assertEqual(
            result["request_results"][1]["raw_kv_contiguous_hit_tokens"],
            16,
        )
        self.assertEqual(
            result["request_results"][1]["effective_hit_tokens"],
            0,
        )
        self.assertEqual(result["usable_gdn_state_avoided_tokens"], 0)

    def test_chunk64_mode_uses_only_native_recurrence_boundaries(self):
        records = [
            decoded(list(range(1, 9)), capacity=16, ordinal=1,
                    prompt_tokens=128),
            decoded(list(range(1, 9)), capacity=16, ordinal=2,
                    prompt_tokens=128),
        ]
        direct = sim.simulate(
            records, 16, policy="admission64", restore_mode="direct")
        hybrid64 = sim.simulate(
            records, 16, policy="admission64", restore_mode="hybrid64")
        chunk64 = sim.simulate(
            records, 16, policy="admission64", restore_mode="chunk64")

        self.assertEqual(direct["usable_gdn_state_avoided_tokens"], 112)
        self.assertEqual(hybrid64["usable_gdn_state_avoided_tokens"], 112)
        self.assertEqual(chunk64["usable_gdn_state_avoided_tokens"], 64)
        self.assertEqual(chunk64["gdn_restore_mode"], "chunk64")

    def test_main_reports_policies_and_optional_baseline_projection(self):
        records = [
            decoded([1, 2], capacity=2, ordinal=1),
            decoded([1, 2], capacity=2, ordinal=2),
        ]
        records[0]["request_id_sha256"] = "0" * 16
        records[1]["request_id_sha256"] = "1" * 16
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            path = root / "trace.log"
            lines = [
                sim.MARKER + json.dumps({k: v for k, v in item.items()
                                         if not k.startswith("_")})
                for item in records
            ]
            path.write_text("\n".join(lines) + "\n")
            out = root / "report.json"

            sim.main([
                str(path),
                "--expected-requests", "2",
                "--expected-block-size", "16",
                "--out", str(out),
            ])
            report = json.loads(out.read_text())
            self.assertIn("policy_metrics", report)
            self.assertIn("control_policy_metrics", report)
            self.assertEqual(report["trace_version"], 4)
            self.assertFalse(report["qualification_trace"])
            self.assertEqual(report["candidate_gdn_restore_mode"], "direct")
            self.assertIn("off", report["policy_metrics"])
            self.assertIn("fine32", report["policy_metrics"])
            self.assertIn("admission64", report["policy_metrics"])
            self.assertIn("tail64", report["policy_metrics"])
            self.assertIn("admission64_m1_29", report["policy_metrics"])
            self.assertEqual(report["tail64"]["policy"], "tail64")
            self.assertFalse(
                report["policy_metrics"]["admission64"]
                ["per_request_timing_projection_complete"])

            chunk64_out = root / "chunk64.json"
            sim.main([
                str(path),
                "--expected-requests", "2",
                "--expected-block-size", "16",
                "--gdn-restore-mode", "chunk64",
                "--out", str(chunk64_out),
            ])
            chunk64_report = json.loads(chunk64_out.read_text())
            self.assertEqual(
                chunk64_report["policy_metrics"]["fine32"]
                ["gdn_restore_mode"], "direct")
            self.assertEqual(
                chunk64_report["policy_metrics"]["admission64"]
                ["gdn_restore_mode"], "chunk64")
            self.assertEqual(
                chunk64_report["policy_metrics"]["tail64"]
                ["gdn_restore_mode"], "chunk64")

            pair_out = root / "tail64-pair.json"
            with mock.patch.object(
                    sim, "_simulate", wraps=sim._simulate) as replay:
                sim.main([
                    str(path),
                    "--expected-requests", "2",
                    "--expected-block-size", "16",
                    "--gdn-restore-mode", "hybrid64",
                    "--tail64-pair-only",
                    "--out", str(pair_out),
                ])
            pair_report = json.loads(pair_out.read_text())
            self.assertEqual(replay.call_count, 2)
            self.assertEqual(
                pair_report["schema"],
                "bi100-tail64-trace-diagnostic-v1")
            self.assertEqual(
                set(pair_report["policy_metrics"]),
                {"admission64", "tail64"})
            self.assertFalse(
                pair_report["promotion"]["main_or_yaml_change_authorized"])

            with self.assertRaisesRegex(
                    ValueError, "does not accept qualification"):
                sim.main([
                    str(path),
                    "--expected-requests", "2",
                    "--tail64-pair-only",
                    "--qualification-trace",
                    "--out", str(root / "invalid-pair.json"),
                ])

            with self.assertRaisesRegex(
                    ValueError, "aggregate hit-rate scaling is disabled"):
                sim.main([
                    str(path),
                    "--out", str(root / "out.json"),
                    "--expected-requests", "2",
                    "--expected-block-size", "16",
                    "--baseline-cache-tps", "100",
                    "--baseline-metrics", str(path),
                ])

            with self.assertRaisesRegex(
                    ValueError, "aggregate hit-rate scaling is disabled"):
                sim.main([
                    str(path),
                    "--out", str(out),
                    "--expected-requests", "2",
                    "--expected-block-size", "16",
                    "--baseline-cache-tps", "100",
                    "--baseline-weighted-score", "1000",
                ])

    def test_main_reports_cpu_candidate_against_zero_cpu_control(self):
        records = [
            record([1, 2], capacity=2, request_id=0, ordinal=1),
            record([3, 4], capacity=2, request_id=1, ordinal=2),
            record([1, 2], capacity=2, request_id=2, ordinal=3),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            trace = root / "trace.log"
            trace.write_text("\n".join(
                sim.MARKER + json.dumps(item) for item in records) + "\n")
            out = root / "report.json"
            sim.main([
                str(trace), "--out", str(out),
                "--expected-requests", "3",
                "--expected-block-size", "16",
                "--cpu-capacity-blocks", "2",
            ])
            report = json.loads(out.read_text())
            self.assertEqual(
                report["control_policy_metrics"]["admission64"]
                ["cpu_hit_blocks"], 0)
            self.assertEqual(
                report["policy_metrics"]["admission64"]
                ["cpu_hit_blocks"], 2)
            self.assertGreater(
                report[
                    "cpu_tier_admission64_effective_hit_gain_percentage_points"],
                0)

    def test_per_request_residual_projection_uses_trace_timings(self):
        records = [
            record([1, 2], request_id=0, ordinal=1),
            record([1, 2], request_id=1, ordinal=2),
        ]
        records[0].update({
            "ttft_s": 2.0,
            "request_latency_s": 3.0,
            "time_in_queue_s": 0.1,
            "observed_effective_cached_tokens": 0,
        })
        records[1].update({
            "ttft_s": 1.0,
            "request_latency_s": 2.0,
            "time_in_queue_s": 0.1,
            "observed_effective_cached_tokens": 16,
        })
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            trace = root / "trace.log"
            trace.write_text("\n".join(
                sim.MARKER + json.dumps(item) for item in records) + "\n")
            baseline = root / "baseline.json"
            baseline.write_text(json.dumps({
                "run_id": "timed",
                "trace_session_sha256": "0123456789abcdef",
                "cache_tps": 100.0,
                "weighted_score": 1000.0,
                "output_tps_p10": 21.0,
                "success_rate": 1.0,
            }))
            out = root / "report.json"
            sim.main([
                str(trace), "--out", str(out),
                "--expected-requests", "2",
                "--expected-block-size", "16",
                "--baseline-metrics", str(baseline),
            ])
            report = json.loads(out.read_text())
            metrics = report["policy_metrics"]["admission64"]
            self.assertTrue(metrics["per_request_timing_projection_complete"])
            self.assertIsNotNone(metrics["projected_ttft_p90_s"])
            qualification = report["qualification"]["admission64"]
            self.assertEqual(
                qualification["projection_model"],
                "per_request_residual_prefill_directional_v1")
            self.assertIsNotNone(qualification["projected_weighted_score"])
            self.assertFalse(qualification["ok"])
            self.assertFalse(
                qualification["gates"]["attested_baseline_contract"])
            self.assertFalse(
                report["promotion"]["main_or_yaml_change_authorized"])

    def test_qualification_rejects_legacy_baseline_metrics(self):
        records = [
            record([1, 2], request_id=0, ordinal=1),
            record([1, 2], request_id=1, ordinal=2),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            trace = root / "trace.log"
            trace.write_text("\n".join(
                sim.MARKER + json.dumps(item) for item in records) + "\n")
            baseline = root / "legacy.json"
            baseline.write_text(json.dumps({
                "run_id": "legacy",
                "trace_session_sha256": "0123456789abcdef",
                "cache_tps": 100.0,
                "weighted_score": 1000.0,
                "output_tps_p10": 21.0,
                "success_rate": 1.0,
            }))
            with self.assertRaisesRegex(
                    ValueError, "rejects legacy baseline metrics"):
                sim.main([
                    str(trace),
                    "--expected-requests", "2",
                    "--qualification-trace",
                    "--baseline-metrics", str(baseline),
                    "--out", str(root / "out.json"),
                ])

    def test_attested_881_trace_is_only_an_offline_phase_gate(self):
        raw_records = []
        for index in range(881):
            item = record(
                [1, 2], capacity=4, request_id=index, ordinal=index + 1)
            item.update({
                "ttft_s": 1.0,
                "request_latency_s": 2.0,
                "time_in_queue_s": 0.0,
                "observed_effective_cached_tokens": 0,
            })
            raw_records.append(item)
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            trace = root / "trace.log"
            trace.write_text("\n".join(
                sim.MARKER + json.dumps(item) for item in raw_records) + "\n")
            records = sim.read([str(trace)])
            baseline = root / "baseline.json"
            baseline.write_text(json.dumps(
                attested_baseline(records, trace), indent=2, sort_keys=True))
            out = root / "report.json"

            sim.main([
                str(trace),
                "--qualification-trace",
                "--baseline-metrics", str(baseline),
                "--out", str(out),
            ])

            report = json.loads(out.read_text())
            self.assertTrue(report["qualification_trace"])
            self.assertTrue(report["qualification_evidence_attested"])
            self.assertTrue(
                report["evidence"]["baseline_contract_attested"])
            self.assertFalse(
                report["promotion"]["main_or_yaml_change_authorized"])
            self.assertFalse(
                report["promotion"]["official_score_claim_authorized"])
            for candidate in ("admission64", "admission64_m1_29"):
                self.assertFalse(
                    report["qualification"][candidate]
                    ["main_or_yaml_change_authorized"])
                self.assertEqual(
                    report["qualification"][candidate]["decision_scope"],
                    "offline_cache_phase_gate_only",
                )

    def test_docker_json_wrapper_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            log = root / "trace.json.log"
            lines = []
            for index in range(2):
                payload = {
                    k: v for k, v in record([1, 2], request_id=index,
                                             ordinal=index + 1).items()
                    if not k.startswith("_")
                }
                lines.append(json.dumps({
                    "log": sim.MARKER + json.dumps(payload),
                    "time": f"2026-07-18T00:00:0{index}Z",
                }))
            log.write_text("\n".join(lines) + "\n")
            out = root / "report.json"
            sim.main([
                str(log),
                "--out", str(out),
                "--expected-requests", "2",
                "--expected-block-size", "16",
            ])
            report = json.loads(out.read_text())
            self.assertEqual(report["source_logs"][0]["bytes"], log.stat().st_size)
            self.assertEqual(len(report["source_logs"][0]["sha256"]), 64)

    def test_cpu_recovers_gpu_evicted_prefix(self):
        records = [
            decoded([1, 2], capacity=2, ordinal=1),
            decoded([3, 4], capacity=2, ordinal=2),
            decoded([1, 2], capacity=2, ordinal=3),
        ]
        result = sim.simulate(records, 2, policy="fine32", cpu_capacity=2)
        recovered = result["request_results"][2]
        self.assertEqual(recovered["cpu_hit_blocks"], 2)
        self.assertEqual(recovered["h2d_blocks"], 2)
        self.assertEqual(recovered["d2h_blocks"], 0)
        self.assertEqual(recovered["effective_hit_blocks"], 1)
        self.assertEqual(recovered["residual_prefill_tokens"], 16)
        self.assertEqual(result["d2h_blocks"], 2)

    def test_gpu_first_avoids_cpu_promotion(self):
        records = [
            decoded([1, 2], capacity=2, ordinal=1),
            decoded([1, 2], capacity=2, ordinal=2),
        ]
        result = sim.simulate(records, 2, policy="fine32", cpu_capacity=2)
        warm = result["request_results"][1]
        self.assertEqual(warm["cpu_hit_blocks"], 0)
        self.assertEqual(warm["h2d_blocks"], 0)
        self.assertEqual(warm["raw_kv_contiguous_hit_tokens"], 16)

    def test_cpu_lru_replaces_oldest_copy(self):
        records = [
            decoded([1, 2], capacity=2, ordinal=1),
            decoded([3, 4], capacity=2, ordinal=2),
            decoded([1, 3], capacity=2, ordinal=3),
            decoded([5, 6], capacity=2, ordinal=4),
        ]
        result = sim._simulate(records, 2, False, "off", cpu_capacity=2)
        self.assertEqual(
            list(result["final_cpu_cache"]), [digest(1), digest(3)])

    def test_saturated_promotion_preserves_later_cpu_source(self):
        records = [
            decoded([1, 2], capacity=2, ordinal=1),
            decoded([3, 4], capacity=2, ordinal=2),
            decoded([1, 2], capacity=2, ordinal=3),
        ]
        result = sim._simulate(records, 2, False, "fine32", cpu_capacity=2)
        promoted = result["request_results"][2]
        self.assertEqual(promoted["cpu_hit_blocks"], 2)
        self.assertEqual(promoted["h2d_blocks"], 2)
        self.assertEqual(promoted["d2h_blocks"], 0)
        self.assertEqual(list(result["final_cpu_cache"]),
                         [digest(1), digest(2)])

    def test_store_before_later_promotion_is_deferred(self):
        records = [
            decoded([1, 2], capacity=2, ordinal=1),
            decoded([3, 4], capacity=2, ordinal=2),
            decoded([5, 1], capacity=2, ordinal=3),
        ]
        result = sim._simulate(records, 2, False, "fine32", cpu_capacity=2)
        mixed = result["request_results"][2]
        self.assertEqual(mixed["cpu_hit_blocks"], 1)
        self.assertEqual(mixed["h2d_blocks"], 1)
        self.assertEqual(mixed["d2h_blocks"], 0)
        self.assertGreaterEqual(result["d2h_skipped_blocks"], 1)
        self.assertEqual(set(result["final_cpu_cache"]),
                         {digest(1), digest(2)})

    def test_same_step_d2h_destination_is_not_a_cpu_hit(self):
        records = [
            decoded([1, 2], capacity=2, ordinal=1),
            decoded([3, 2], capacity=2, ordinal=2),
        ]
        result = sim.simulate(records, 2, policy="off", cpu_capacity=2)
        second = result["request_results"][1]
        self.assertEqual(second["cpu_hit_blocks"], 0)
        self.assertEqual(second["h2d_blocks"], 0)
        self.assertEqual(second["d2h_blocks"], 2)

    def test_per_request_ttft_includes_transfer_cost(self):
        records = [
            decoded([1, 2], capacity=2, ordinal=1),
            decoded([3, 4], capacity=2, ordinal=2),
            decoded([1, 2], capacity=2, ordinal=3),
        ]
        records[2].update({
            "ttft_s": 1.0,
            "request_latency_s": 2.0,
            "time_in_queue_s": 0.2,
            "observed_effective_cached_tokens": 0,
        })
        result = sim.simulate(
            records, 2, policy="fine32", cpu_capacity=2,
            h2d_ms_per_block=10.0, d2h_ms_per_block=20.0)
        timing = result["request_results"][2]
        self.assertEqual(timing["h2d_blocks"], 2)
        self.assertEqual(timing["d2h_blocks"], 0)
        self.assertAlmostEqual(timing["baseline_prefill_s"], 0.8)
        self.assertAlmostEqual(timing["projected_prefill_s"], 0.4)
        self.assertAlmostEqual(timing["h2d_transfer_s"], 0.02)
        self.assertAlmostEqual(timing["projected_ttft_s"], 0.62)

    def test_decode_d2h_affects_latency_but_not_ttft(self):
        records = [
            decoded([1, 2], capacity=2, ordinal=1),
            decoded([3, 4], capacity=2, ordinal=2,
                    prompt_tokens=16, total_tokens=32),
        ]
        records[1].update({
            "ttft_s": 1.0,
            "request_latency_s": 2.0,
            "time_in_queue_s": 0.0,
            "observed_effective_cached_tokens": 0,
        })
        result = sim.simulate(
            records, 2, policy="off", cpu_capacity=2,
            h2d_ms_per_block=10.0, d2h_ms_per_block=10.0)
        timing = result["request_results"][1]
        self.assertEqual(timing["prefill_d2h_blocks"], 1)
        self.assertEqual(timing["decode_d2h_blocks"], 1)
        self.assertAlmostEqual(timing["projected_ttft_s"], 1.01)
        self.assertAlmostEqual(timing["projected_request_latency_s"], 2.02)
        self.assertAlmostEqual(timing["d2h_transfer_s"], 0.02)

    def test_qualification_trace_requires_explicit_complete_881(self):
        records = [{"ordinal": ordinal} for ordinal in range(1, 882)]
        self.assertFalse(sim._qualification_trace(records))
        self.assertTrue(sim._qualification_trace(
            records, explicitly_declared=True))
        self.assertFalse(sim._qualification_trace(
            records[:-1], explicitly_declared=True))

    def test_cpu_capacity_zero_regression(self):
        records = [
            decoded([1, 2], capacity=2, ordinal=1),
            decoded([3, 4], capacity=2, ordinal=2),
            decoded([1, 2], capacity=2, ordinal=3),
        ]
        result = sim.simulate(records, 2, policy="fine32", cpu_capacity=0)
        self.assertEqual(result["raw_kv_contiguous_hit_tokens"], 0)
        self.assertEqual(result["usable_gdn_state_avoided_tokens"], 0)
        self.assertEqual(result["residual_prefill_tokens"], 96)
        self.assertEqual(result["cpu_hit_blocks"], 0)
        self.assertEqual(result["h2d_blocks"], 0)
        self.assertEqual(result["d2h_blocks"], 0)


if __name__ == "__main__":
    unittest.main()
