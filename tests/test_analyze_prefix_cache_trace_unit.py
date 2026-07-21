import base64
import hashlib
import json
import pathlib
import tempfile
import unittest
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "scripts"))
import analyze_prefix_cache_trace as sim


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

    def test_chunk64_mode_uses_only_native_recurrence_boundaries(self):
        records = [
            decoded(list(range(1, 9)), capacity=16, ordinal=1,
                    prompt_tokens=128),
            decoded(list(range(1, 9)), capacity=16, ordinal=2,
                    prompt_tokens=128),
        ]
        direct = sim.simulate(
            records, 16, policy="admission64", restore_mode="direct")
        chunk64 = sim.simulate(
            records, 16, policy="admission64", restore_mode="chunk64")

        self.assertEqual(direct["usable_gdn_state_avoided_tokens"], 112)
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
            self.assertIn("admission64_m1_29", report["policy_metrics"])
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
                "per_request_residual_prefill")
            self.assertIsNotNone(qualification["projected_weighted_score"])
            self.assertFalse(qualification["ok"])

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
