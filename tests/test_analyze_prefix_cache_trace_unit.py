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
            self.assertEqual(report["trace_version"], 4)
            self.assertIn("off", report["policy_metrics"])
            self.assertIn("fine32", report["policy_metrics"])
            self.assertIn("admission64", report["policy_metrics"])

            with self.assertRaisesRegex(
                    ValueError, "baseline metrics file cannot be combined"):
                sim.main([
                    str(path),
                    "--out", str(root / "out.json"),
                    "--expected-requests", "2",
                    "--expected-block-size", "16",
                    "--baseline-cache-tps", "100",
                    "--baseline-metrics", str(path),
                ])

            sim.main([
                str(path),
                "--out", str(out),
                "--expected-requests", "2",
                "--expected-block-size", "16",
                "--baseline-cache-tps", "100",
                "--baseline-weighted-score", "1000",
                "--cache-coefficient", "0.5",
            ])
            projected = json.loads(out.read_text())
            self.assertIn("weighted_cache_tps_upper_bound", projected)

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


if __name__ == "__main__":
    unittest.main()
