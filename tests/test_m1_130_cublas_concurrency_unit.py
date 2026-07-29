from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "tests/bench_m1_130_cublas_concurrency.py"
RUNNER = ROOT / "scripts/run_m1_130_cublas_concurrency.sh"
DOC = ROOT / "docs/experiments/M1_130_CUBLAS_CONCURRENCY_20260729.md"


class M1130CublasConcurrencyUnitTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.bench = BENCH.read_text(encoding="utf-8")
        cls.runner = RUNNER.read_text(encoding="utf-8")
        cls.doc = DOC.read_text(encoding="utf-8")

    def test_probe_uses_fixed_fp32_production_shapes(self) -> None:
        for marker in (
            "HEADS = 4",
            "HEAD_DIM = 256",
            "KEY_TOKENS = 512",
            '"q8176": 8176',
            '"q5616": 5616',
            "dtype=torch.float32",
            "torch.bmm(qk_left, qk_right, out=",
            "torch.bmm(pv_left, pv_right, out=",
            "torch.backends.cuda.matmul.allow_tf32 = False",
        ):
            self.assertIn(marker, self.bench)

    def test_probe_has_independent_streams_and_strict_numerical_gate(
            self) -> None:
        for marker in (
            "qk_stream = torch.cuda.Stream()",
            "pv_stream = torch.cuda.Stream()",
            "with torch.cuda.stream(qk_stream)",
            "with torch.cuda.stream(pv_stream)",
            "RELATIVE_L2_LIMIT = 1e-7",
            "MAX_ABS_LIMIT = 1e-5",
            "MIN_CELL_SPEEDUP = 1.05",
            "sequential_over_concurrent_speedup",
            "qk_concurrent_vs_sequential",
            "pv_concurrent_vs_sequential",
            '"double_buffer_pipeline_authorized": False',
            '"tp4_service_authorized": False',
        ):
            self.assertIn(marker, self.bench)
        ast.parse(self.bench)

    def test_probe_report_excludes_sensitive_payloads(self) -> None:
        for marker in (
            '"contains_raw_tensors": False',
            '"contains_model_outputs": False',
            '"contains_credentials": False',
        ):
            self.assertIn(marker, self.bench)
        for forbidden in (
            "request_contract_sha256",
            "semantic_output_sha256",
            "session_token",
            "input_ids",
            "prompt",
        ):
            self.assertNotIn(forbidden, self.bench)

    def test_runner_freezes_cells_and_scoped_lifecycle(self) -> None:
        for marker in (
            "CASES=(q8176 q5616)",
            "GPUS=(0 1)",
            "exec_bi100_session.py",
            "bi100_stop_process_group",
            '"$pgid" "$leader" 60 20',
            "bench_m1_130_cublas_concurrency.py",
            "--gpus 0,1,2,3",
            "service_postflight_gate.py",
            "compare_bi100_preflights.py",
            "fatal_scan.rc",
            "timeout_scan.rc",
            "MIN_MEDIAN_SPEEDUP = 1.10",
            "MIN_CELL_SPEEDUP = 1.05",
            "RELATIVE_L2_LIMIT = 1e-7",
            "MAX_ABS_LIMIT = 1e-5",
            '"qk_concurrent_vs_sequential"',
            '"pv_concurrent_vs_sequential"',
            "fixed shape or dtype differs",
            "timing trial contract differs",
            "report is unavailable or invalid",
            "cells_execution.rc",
            '"double_buffer_pipeline_authorized": qualified',
            '"tp4_service_authorized": False',
        ):
            self.assertIn(marker, self.runner)
        self.assertNotIn("pkill", self.runner)
        self.assertNotIn(
            "one or more M1-130 probe cells failed to execute",
            self.runner,
        )
        completed = subprocess.run(
            ["bash", "-n", str(RUNNER)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_runner_embedded_python_is_valid(self) -> None:
        fragments = self.runner.split("<<'PY'\n")[1:]
        self.assertGreaterEqual(len(fragments), 4)
        for fragment in fragments:
            source, separator, _ = fragment.partition("\nPY\n")
            self.assertEqual(separator, "\nPY\n")
            ast.parse(source)

    def test_invalid_invocations_fail_without_gpu_access(self) -> None:
        for command in (
            ["python3", str(BENCH)],
            ["bash", str(RUNNER)],
        ):
            completed = subprocess.run(
                command,
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)

    def test_document_limits_probe_authority(self) -> None:
        for marker in (
            "diagnostic-only",
            "1.10x",
            "1.05x",
            "does not authorize TP4",
            "does not modify the runtime overlay",
            "does not modify `computility-run.yaml`",
        ):
            self.assertIn(marker, self.doc)


if __name__ == "__main__":
    unittest.main()
