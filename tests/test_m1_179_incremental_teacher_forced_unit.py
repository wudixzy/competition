from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import run_m1_179_incremental_teacher_forced as runner


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _arm(root: Path, label: str) -> tuple[Path, str]:
    root.mkdir()
    variant = runner.ARM_VARIANTS[label]
    selector = runner.ARM_SELECTORS[label]
    extension = Path(f"/tmp/{variant}.so")
    digest = "b" * 64 if label == "candidate" else "a" * 64
    status = {
        "schema": "bi100-attention-operator-tp4-arm-v1", "version": 1,
        "workload_mode": "teacher_forced", "selector": selector,
        "fused_variant": variant, "extension_path": str(extension),
        "extension_sha256": digest, "qualified": True,
        "result_status": "pass", "returncode": 0,
        "terminal_stage": "complete", "targets": list(runner.TARGETS),
        "repetitions": 1, "service_startups": 1,
        "request_population": {"expected": 4, "attempted": 4,
                               "completed": 4, "failed": 0},
        "dispatch_count": 4, "gates": {"all": 0},
        "source_revision": "a" * 40, "source_dirty_summary": "clean",
        "runtime_identity": "runtime-1", "instance": "instance-1",
        "model_path": "/model", "workload_id": "m1-179",
        "session_preflight_id": "preflight-1",
    }
    manifest = {
        "schema": "bi100-attention-operator-runtime-v1", "version": 1,
        "workload_mode": "teacher_forced", "source_revision": "a" * 40,
        "runtime_identity": "runtime-1", "instance": "instance-1",
        "model_path": "/model", "tokenizer_path": "/model",
        "tensor_parallel_size": 4, "dtype": "float16",
        "max_model_len": 262144, "block_size": 16,
        "command": ["launch_service"], "compiler": "clang 16",
        "fused_variant": variant,
        "extension_identity": {"module_path": str(extension),
                               "runtime_loaded_module": str(extension),
                               "sha256": digest},
        "environment": {
            "BI100_ATTN_COREX_FUSED_PREFILL": "1",
            "BI100_ATTN_COREX_FUSED_PREFILL_VARIANT": variant,
            "BI100_ATTN_COREX_FUSED_PREFILL_EXTENSION": str(extension),
            "BI100_ATTN_COREX_FUSED_PREFILL_EXTENSION_SHA256": digest,
            "BI100_CACHE_TRACE": "0", "OTHER": "same"},
    }
    values = {
        "runner_status.json": status, "runtime_manifest.json": manifest,
        "measurement.json": {"cases": [
            {"cached_tokens": 0} for _ in runner.TARGETS]},
        "fatal_scan.json": {"qualified": True, "category_counts": {}},
        "postflight_after.json": {"qualified": True,
            "api_server_pids": [], "worker_pids": [], "gpu_processes": []},
        "scoped_cleanup.json": {"qualified": True},
    }
    for name, value in values.items():
        _write(root / name, value)
    return extension, digest


class M1179RunnerTests(unittest.TestCase):

    def test_three_arm_variant_and_cross_arm_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arms = {}
            for label in runner.ARM_VARIANTS:
                extension, digest = _arm(root / label, label)
                reasons, arms[label] = runner.validate_arm(
                    root / label, label, extension, digest)
                self.assertEqual(reasons, [])
            self.assertEqual(runner.cross_arm_reasons(arms), [])

    def test_fused_off_or_wrong_variant_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extension, digest = _arm(root / "control_a", "control_a")
            manifest_path = root / "control_a/runtime_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["environment"]["BI100_ATTN_COREX_FUSED_PREFILL"] = "0"
            _write(manifest_path, manifest)
            reasons, _ = runner.validate_arm(
                root / "control_a", "control_a", extension, digest)
            self.assertTrue(any("runtime extension" in reason
                                for reason in reasons))

    def test_drift_runs_control_b_but_invalid_evidence_stops(self) -> None:
        self.assertTrue(runner.control_b_required(True, True))
        self.assertTrue(runner.control_b_required(True, False))
        self.assertFalse(runner.control_b_required(False, True))


if __name__ == "__main__":
    unittest.main()
