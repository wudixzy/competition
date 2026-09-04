from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import run_m1_178_attention_teacher_forced as runner


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _arm(root: Path, selector: str) -> None:
    root.mkdir()
    status = {
        "schema": "bi100-attention-operator-tp4-arm-v1", "version": 1,
        "workload_mode": "teacher_forced", "selector": selector,
        "qualified": True, "result_status": "pass", "returncode": 0,
        "terminal_stage": "complete", "targets": list(runner.TARGETS),
        "repetitions": 1, "service_startups": 1,
        "request_population": {"expected": 4, "attempted": 4,
                               "completed": 4, "failed": 0},
        "gates": {"all": 0},
        "dispatch_count": 4 if selector == "candidate" else 0,
        "source_revision": "a" * 40, "source_dirty_summary": "clean",
        "runtime_identity": "runtime-1", "instance": "instance-1",
        "model_path": "/root/public-storage/models/Qwen/Qwen3.6-35B-A3B",
        "workload_id": "m1-178", "session_preflight_id": "preflight-1",
    }
    manifest = {
        "schema": "bi100-attention-operator-runtime-v1", "version": 1,
        "workload_mode": "teacher_forced", "source_revision": "a" * 40,
        "runtime_identity": "runtime-1", "instance": "instance-1",
        "model_path": "/root/public-storage/models/Qwen/Qwen3.6-35B-A3B",
        "tokenizer_path": "/root/public-storage/models/Qwen/Qwen3.6-35B-A3B",
        "tensor_parallel_size": 4, "dtype": "float16",
        "max_model_len": 262144, "block_size": 16,
        "command": ["launch_service"],
        "environment": {
            "BI100_ATTN_COREX_FUSED_PREFILL": (
                "1" if selector == "candidate" else "0"),
            "BI100_CACHE_TRACE": "0", "OTHER": "same",
        },
    }
    values = {
        "runner_status.json": status,
        "runtime_manifest.json": manifest,
        "measurement.json": {"cases": [
            {"cached_tokens": 0} for _ in runner.TARGETS]},
        "fatal_scan.json": {"qualified": True, "category_counts": {}},
        "postflight_after.json": {
            "qualified": True, "api_server_pids": [], "worker_pids": [],
            "gpu_processes": []},
        "scoped_cleanup.json": {"qualified": True},
    }
    for name, value in values.items():
        _write(root / name, value)


class M1178RunnerTests(unittest.TestCase):

    def test_valid_arm_and_cross_arm_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _arm(root / "control_a", "control")
            _arm(root / "candidate", "candidate")
            control_reasons, control = runner.validate_arm(
                root / "control_a", "control")
            candidate_reasons, candidate = runner.validate_arm(
                root / "candidate", "candidate")
            self.assertEqual(control_reasons, [])
            self.assertEqual(candidate_reasons, [])
            self.assertEqual(runner.cross_arm_reasons({
                "control_a": control, "candidate": candidate}), [])

    def test_missing_dispatch_or_dirty_identity_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _arm(root / "control_a", "control")
            _arm(root / "candidate", "candidate")
            status_path = root / "candidate/runner_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["dispatch_count"] = 0
            _write(status_path, status)
            reasons, _ = runner.validate_arm(root / "candidate", "candidate")
            self.assertTrue(any("dispatch" in reason for reason in reasons))

            status["dispatch_count"] = 4
            status["source_dirty_summary"] = "different"
            _write(status_path, status)
            _, control = runner.validate_arm(root / "control_a", "control")
            _, candidate = runner.validate_arm(root / "candidate", "candidate")
            reasons = runner.cross_arm_reasons({
                "control_a": control, "candidate": candidate})
            self.assertTrue(any("source_dirty_summary" in reason
                                for reason in reasons))


if __name__ == "__main__":
    unittest.main()
