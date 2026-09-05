from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

import compare_m1_181_ifeval as comparison
import m1_181_ifeval_distribution_api as workload
import run_m1_181_adjudication as orchestrator
import run_m1_181_m1_109_numeric as numeric_runner
from test_compare_teacher_forced_logprobs_v2_unit import _report


def _teacher(label: str) -> dict:
    value = _report("candidate" if label == "m1_109" else "control")
    value.update({"source_revision": "a" * 40,
                  "runtime_identity": "runtime", "instance": "instance",
                  "model_path": "/model"})
    value["optimization"]["fused_prefill"] = "1" if label == "m1_109" else "0"
    if label == "m1_109":
        value.update({"schema": "bi100-teacher-forced-topk-observation-v2",
                      "version": 2, "fused_variant": "m1_109_fp32_qk",
                      "extension_identity": {
                          "module_path": "/tmp/m109.so",
                          "runtime_loaded_module": "/tmp/m109.so",
                          "sha256": "b" * 64}})
    return value


def _arm(label: str, *, stopped: bool = False) -> dict:
    cases = [{
        "key": key, "instruction_id_list": ["family:test"],
        "strict": [True], "loose": [True], "http_status": 200,
        "finish_reason": "stop", "usage": {"prompt_tokens": 10,
        "completion_tokens": 2, "total_tokens": 12, "cached_tokens": 0},
        "elapsed_s": 1.0, "all_values_finite": True,
    } for key in range(16 if stopped else 64)]
    ifeval = None if label == "fused_off_b" else {
        "selected": 64, "completed": len(cases), "complete": not stopped,
        "stopped_after_smoke": stopped, "cases": cases,
        "smoke_baseline_only": {"strict": 3 if stopped else 0,
                                "loose": 0},
    }
    teacher = None if stopped else _teacher(label)
    count = (len(cases) if ifeval else 0) + (4 if teacher else 0)
    return {
        "schema": "bi100-m1-181-arm-observation-v1", "version": 1,
        "arm": label,
        "algorithm_variant": "m1_109_fp32_qk" if label == "m1_109" else "fused_off",
        "source_revision": "a" * 40, "runtime_identity": "runtime",
        "instance": "instance", "model_path": "/model", "workload_id": "w",
        "ifeval": ifeval, "teacher_forced": teacher,
        "request_population": {"attempted": count, "completed": count,
                               "failed": 0}, "wall_s": 1.0,
    }


class M1181Tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.bootstrap = comparison.distribution.BOOTSTRAP_SAMPLES
        comparison.distribution.BOOTSTRAP_SAMPLES = 100

    @classmethod
    def tearDownClass(cls) -> None:
        comparison.distribution.BOOTSTRAP_SAMPLES = cls.bootstrap

    def test_smoke_selection_and_original_completion_budget_are_frozen(self) -> None:
        self.assertEqual(len(workload.SMOKE_ORDINALS), 16)
        self.assertEqual(workload.SMOKE_ORDINALS, tuple(range(0, 64, 4)))
        manifest, _, _ = workload.ifeval.load_manifest(
            workload.ifeval.DEFAULT_MANIFEST)
        self.assertEqual(manifest["request_conversion"]["max_tokens"], 8192)

    def test_ifeval64_pass_is_separate_from_distribution(self) -> None:
        result = comparison.compare(_arm("fused_off"), _arm("m1_109"))
        self.assertEqual(result["ifeval_statistical_capability"]["status"],
                         "pass")
        self.assertEqual(result["fused_off_vs_m1_109_distribution"]["status"],
                         "inconclusive")
        self.assertFalse(result["fused_off_vs_m1_109_distribution"]["calibrated"])

    def test_length_limited_reasoning_only_response_is_scoreable_failure(self) -> None:
        body = {
            "choices": [{"finish_reason": "length", "message": {
                "content": None, "reasoning_content": "unfinished reasoning",
                "tool_calls": None}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 8192,
                      "total_tokens": 8199,
                      "prompt_tokens_details": {"cached_tokens": 0}},
        }
        normalized = workload.normalize_capability_response(body, 12.5)
        self.assertEqual(normalized["content"], "")
        self.assertTrue(normalized["empty_final_content"])
        self.assertEqual(normalized["finish_reason"], "length")

    def test_empty_final_content_is_invalid_without_length_limited_reasoning(self) -> None:
        body = {
            "choices": [{"finish_reason": "stop", "message": {
                "content": "", "reasoning_content": "reasoning",
                "tool_calls": None}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 1,
                      "total_tokens": 8,
                      "prompt_tokens_details": {"cached_tokens": 0}},
        }
        with self.assertRaisesRegex(ValueError, "nonempty text"):
            workload.normalize_capability_response(body, 1.0)

    def test_fused_off_b_calibrates_only_fused_off_comparison(self) -> None:
        result = comparison.compare(
            _arm("fused_off"), _arm("m1_109"), _arm("fused_off_b"))
        self.assertEqual(result["fused_off_aa_distribution"]["top1_flip_count"], 0)
        self.assertTrue(result["fused_off_vs_m1_109_distribution"]["calibrated"])
        self.assertEqual(result["fused_off_vs_m1_109_distribution"]
                         ["aa_control_variant"], "fused_off")

    def test_predeclared_smoke_regression_stops_candidate(self) -> None:
        result = comparison.compare(
            _arm("fused_off"), _arm("m1_109", stopped=True))
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["ifeval_statistical_capability"]["classification"],
                         "smoke_regression_stop")

    def test_numeric_aggregate_requires_all_four_two_cell_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for rank in range(4):
                (root / f"rank-{rank}.json").write_text(json.dumps({
                    "schema": "bi100-m1-181-m1-109-rank-replay-v1",
                    "logical_tp_rank": rank, "physical_gpu": rank,
                    "all_qualified": True, "wall_s": 2.0,
                    "records": [{"numeric": {"all_finite": True,
                        "relative_l2_error_ratio": 1.0,
                        "maximum_absolute_error_ratio": 1.0,
                        "candidate_lse_relative_l2": 0.0},
                        "repeat_exact": {"output": True, "lse": True}}
                        for _ in range(2)]}))
            result = numeric_runner.aggregate(root, [0, 0, 0, 0], 3.0)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["cell_count"], 8)

    def test_orchestrator_commands_bind_variants_and_baseline(self) -> None:
        args = argparse.Namespace(
            instance="instance", run_root=Path("/tmp/m181"), pair_id="pair",
            session_preflight=Path("/tmp/preflight.json"),
            m1_109_extension=Path("/tmp/m109.so"), m1_109_sha256="b" * 64)
        fused = orchestrator.arm_command(args, "fused_off")
        candidate = orchestrator.arm_command(args, "m1_109")
        control_b = orchestrator.arm_command(args, "fused_off_b")
        self.assertNotIn("--fused-variant", fused)
        self.assertIn("m1_109_fp32_qk", candidate)
        self.assertIn("--reference-fused-off", candidate)
        self.assertNotIn("--fused-variant", control_b)


if __name__ == "__main__":
    unittest.main()
