#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import qualify_qwen36_diagnostic_components as qualify  # noqa: E402


class Qwen36DiagnosticComponentGateUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.paths: dict[str, Path] = {}
        self._write_fixtures()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self, name: str, value: object) -> Path:
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        self.paths[name] = path
        return path

    def _write_fixtures(self) -> None:
        for rank in range(4):
            self._write(f"qgkv{rank}", {
                "tp_rank": rank,
                "loaded": [True, True, True],
                "weight_exact": [True, True, True],
                "output_checks": [
                    {"exact": True, "max_abs": 0.0}
                    for _ in range(3)
                ],
                "ok": True,
            })
        self._write("moe", {
            "shape": {
                "experts": 256,
                "top_k": 8,
                "hidden": 2048,
                "intermediate": 128,
                "dtype": "torch.float16",
            },
            "extension_capabilities": {
                "w13": True,
                "w2_reduce": True,
                "w13_silu": False,
            },
            "numerics": {
                name: {"finite": True, "relative_l2": 1.0e-6}
                for name in ("direct_w13", "direct_w2_reduce", "staged")
            },
            "sequence": {
                "staged": {
                    "steps": 500,
                    "finite_steps": 500,
                    "relative_l2": 1.0e-6,
                },
            },
            "timings": {
                "staged_fixed": {"speedup_vs_baseline": 1.6},
                "staged_routed": {"speedup_vs_baseline": 1.3},
            },
        })
        self._write("gdn", {
            "config": {"shape": [1, 4, 8, 128]},
            "sequence": {
                "steps": 1000,
                "finite_steps": 1000,
                "output_relative_l2": 1.0e-6,
                "state_relative_l2": 1.0e-6,
            },
            "performance": {
                "candidate_median_ms": 0.05,
                "speedup": 2.0,
            },
            "ok": True,
        })
        self._write("paged", {
            "shape": {
                "head_size": 256,
                "block_size": 16,
                "dtype": "float16",
            },
            "results": {
                str(length): {
                    "checks": {
                        "key_exact": True,
                        "value_exact": True,
                        "output_exact": True,
                        "output_max_abs": 0.0,
                    },
                }
                for length in qualify.PAGED_LENGTHS
            },
            "ok": True,
        })
        self._write("cache", {
            "round_trip_byte_exact": True,
            "same_slot_preserved_victim_exact": True,
            "same_slot_promoted_request_exact": True,
            "invalid_mapping_fail_fast": True,
            "invalid_mapping_zero_write": True,
            "invalid_selector_fail_fast": True,
            "gate": {"qualified": True, "reasons": []},
        })
        preflight = {
            "gpus": [3],
            "ok": True,
            "results": [{"gpu": 3, "ok": True, "free": 30 << 30}],
        }
        self._write("before", preflight)
        self._write("after", preflight)
        self._write("runtime", {
            "qualified": True,
            "source_revision": "a" * 40,
            "runtime_tree_sha256": "b" * 64,
        })
        self.log = self.root / "probe.stderr"
        self.log.write_text("", encoding="utf-8")

    def _run(self, qgkv_names: list[str] | None = None) -> tuple[int, dict]:
        qgkv_names = qgkv_names or [
            "qgkv0", "qgkv1", "qgkv2", "qgkv3"]
        out = self.root / "qualification.json"
        argv = ["qualify_qwen36_diagnostic_components.py"]
        for name in qgkv_names:
            argv.extend(["--qgkv", str(self.paths[name])])
        argv.extend([
            "--moe", str(self.paths["moe"]),
            "--gdn", str(self.paths["gdn"]),
            "--paged", str(self.paths["paged"]),
            "--cache", str(self.paths["cache"]),
            "--preflight-before", str(self.paths["before"]),
            "--preflight-after", str(self.paths["after"]),
            "--runtime-identity", str(self.paths["runtime"]),
            "--log", str(self.log),
            "--source-revision", "a" * 40,
            "--source-branch", "exp/test",
            "--instance", "unit",
            "--physical-gpu", "3",
            "--out", str(out),
        ])
        with mock.patch.object(sys, "argv", argv), \
                contextlib.redirect_stdout(io.StringIO()):
            rc = qualify.main()
        return rc, json.loads(out.read_text(encoding="utf-8"))

    def test_complete_fixture_qualifies_without_promotion(self) -> None:
        rc, report = self._run()
        self.assertEqual(rc, 0)
        self.assertTrue(report["qualified"])
        self.assertFalse(report["semantic_quality_evaluated"])
        self.assertFalse(report["full_model_tp4_evaluated"])
        self.assertFalse(report["production_promotion_authorized"])

    def test_moe_relative_l2_above_limit_rejects(self) -> None:
        report = json.loads(self.paths["moe"].read_text(encoding="utf-8"))
        report["numerics"]["direct_w13"]["relative_l2"] = 1.1e-5
        self._write("moe", report)
        rc, result = self._run()
        self.assertEqual(rc, 1)
        self.assertTrue(any(
            "MoE direct_w13 relative L2" in reason
            for reason in result["reasons"]))

    def test_gdn_state_relative_l2_above_limit_rejects(self) -> None:
        report = json.loads(self.paths["gdn"].read_text(encoding="utf-8"))
        report["sequence"]["state_relative_l2"] = 2.0e-5
        self._write("gdn", report)
        rc, result = self._run()
        self.assertEqual(rc, 1)
        self.assertTrue(any(
            "GDN state relative L2" in reason
            for reason in result["reasons"]))

    def test_duplicate_qgkv_rank_rejects(self) -> None:
        rc, report = self._run(
            ["qgkv0", "qgkv1", "qgkv2", "qgkv2"])
        self.assertEqual(rc, 1)
        self.assertTrue(any(
            "duplicate rank" in reason for reason in report["reasons"]))

    def test_fatal_log_pattern_rejects_without_copying_log(self) -> None:
        self.log.write_text(
            "Traceback (most recent call last):\nsecret payload\n",
            encoding="utf-8",
        )
        rc, report = self._run()
        self.assertEqual(rc, 1)
        encoded = json.dumps(report)
        self.assertNotIn("secret payload", encoded)
        self.assertEqual(
            report["fatal_scan"]["hits"][0]["pattern"], "traceback")


class Qwen36DiagnosticComponentStaticTest(unittest.TestCase):
    def test_runner_covers_target_shape_components(self) -> None:
        script = (
            ROOT / "scripts"
            / "run_qwen36_diagnostic_component_gates.sh"
        ).read_text(encoding="utf-8")
        for marker in (
            "bi100_full_attention_qgkv_runtime.py",
            "bench_moe_direct_routed.py",
            "bench_gdn_packed_production_boundary.py",
            "bench_paged_kv_gather.py",
            "bench_m1_57_cache_engine_integration.py",
            "32768,65536,131072,235000",
            "verify_bare_host_runtime_identity.py",
            "preflight_before",
            "preflight_after",
        ):
            self.assertIn(marker, script)
        self.assertNotIn("computility-run.yaml", script)
        self.assertNotIn("git push", script)

    def test_benchmarks_expose_fail_closed_numerics(self) -> None:
        moe = (
            ROOT / "tests" / "bench_moe_direct_routed.py"
        ).read_text(encoding="utf-8")
        gdn = (
            ROOT / "tests" / "bench_gdn_packed_production_boundary.py"
        ).read_text(encoding="utf-8")
        paged = (
            ROOT / "tests" / "bench_paged_kv_gather.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"relative_l2"', moe)
        self.assertIn('hasattr(direct, "w13_silu")', moe)
        self.assertIn('"output_relative_l2"', gdn)
        self.assertIn('"state_relative_l2"', gdn)
        self.assertIn('return 0 if report["ok"] else 1', gdn)
        self.assertIn('return 0 if report["ok"] else 1', paged)


if __name__ == "__main__":
    unittest.main()
