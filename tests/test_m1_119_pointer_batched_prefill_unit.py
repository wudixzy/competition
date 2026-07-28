from __future__ import annotations

from pathlib import Path
import json
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests" / "corex_fused_paged_prefill_batched16_ext.cu"
DEFAULT_SOURCE = (
    ROOT / "qwen3_6_scripts" / "corex_fused_paged_prefill_split4.cu"
)
BUILD = ROOT / "tests" / "build_corex_fused_prefill_batched16.sh"
DEFAULT_BUILD = (
    ROOT / "qwen3_6_scripts" / "build_corex_fused_paged_prefill_split4.sh"
)
PATCH_OPS = ROOT / "qwen3_6_scripts" / "patch_ops.sh"
WRAPPER = ROOT / "scripts" / "run_m1_119_pointer_batched_component_ab.sh"
RUNNER = ROOT / "scripts" / "run_m1_109_fused_softmax_component_ab.sh"
EVIDENCE = (
    ROOT
    / "docs"
    / "experiments"
    / "evidence"
    / "M1_119_POINTER_BATCHED_PREFILL_20260729"
    / "compile_qualification.json"
)


class M1119PointerBatchedPrefillUnitTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.default_source = DEFAULT_SOURCE.read_text(encoding="utf-8")
        cls.build = BUILD.read_text(encoding="utf-8")
        cls.default_build = DEFAULT_BUILD.read_text(encoding="utf-8")
        cls.patch_ops = PATCH_OPS.read_text(encoding="utf-8")
        cls.wrapper = WRAPPER.read_text(encoding="utf-8")
        cls.runner = RUNNER.read_text(encoding="utf-8")
        cls.evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_candidate_is_compile_time_only(self) -> None:
        marker = "BI100_PREFILL_BATCHED16_EXPERIMENT"
        self.assertIn(f"#if defined({marker})", self.source)
        self.assertIn(f"-D{marker}=1", self.build)
        self.assertNotIn(marker, self.default_source)
        self.assertNotIn(marker, self.default_build)
        self.assertNotIn(marker, self.patch_ops)

    def test_candidate_preserves_m1_108_softmax_boundary(self) -> None:
        for marker in (
            "auto block_max = std::get<0>(at::max(",
            "active_scores.sub_(new_maxes.unsqueeze(-1)).exp_();",
            "auto split_sums = at::sum(active_scores, {-1}, false);",
            "merge_split_sums_kernel<<<",
            "kTileTokens = 512",
        ):
            self.assertIn(marker, self.source)

    def test_candidate_batches_split_head_gemms_without_flattening(self) -> None:
        for marker in (
            "kMaxBatchedGemmCount = kSplitCount * kNumQueryHeads",
            "build_split_head_pointer_arrays",
            "cublasSgemmBatched",
            "qk_split_head_batched",
            "pv_split_head_batched",
            "active_splits * kNumQueryHeads",
        ):
            self.assertIn(marker, self.source)
        self.assertNotIn("group_tokens, rows, kHeadDim", self.source)

    def test_wrapper_selects_private_component_schema(self) -> None:
        self.assertIn(
            "BI100_COMPONENT_AB_VARIANT=m1-119-pointer-batched",
            self.wrapper,
        )
        self.assertIn(
            "bi100-m1-119-pointer-batched-component-ab-v1",
            self.runner,
        )
        self.assertIn('"main_or_yaml_change_authorized": False', self.runner)

    def test_build_and_wrapper_shell_syntax(self) -> None:
        for script in (BUILD, WRAPPER, RUNNER):
            result = subprocess.run(
                ["bash", "-n", str(script)],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_variant_fails_before_runtime_access(self) -> None:
        result = subprocess.run(
            ["bash", str(RUNNER)],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            env={"BI100_COMPONENT_AB_VARIANT": "invalid"},
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("BI100_COMPONENT_AB_VARIANT is invalid", result.stderr)

    def test_compile_evidence_keeps_gpu_and_promotion_gates_closed(self) -> None:
        self.assertEqual(
            self.evidence["schema"],
            "bi100-m1-119-pointer-batched-compile-v1",
        )
        self.assertTrue(
            self.evidence["candidate_binary"]["runtime_linkage_complete"])
        self.assertFalse(self.evidence["component_gpu_qualified"])
        self.assertFalse(
            self.evidence["remote_validation"]["gpu_kernel_executed"])
        self.assertFalse(
            self.evidence["decision"]["tp4_service_experiment_authorized"])
        self.assertFalse(
            self.evidence["decision"]["main_or_yaml_change_authorized"])


if __name__ == "__main__":
    unittest.main()
