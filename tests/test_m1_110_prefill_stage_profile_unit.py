from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_m1_110_prefill_stage_profile.sh"
BUILD = ROOT / "tests" / "build_corex_fused_prefill_stage_profile.sh"
SOURCE = ROOT / "tests" / "corex_fused_paged_prefill_stage_profile_ext.cu"
PROFILE = ROOT / "tests" / "profile_m1_110_fused_prefill_stages.py"


def load_profile_module():
    sys.path.insert(0, str(ROOT / "tests"))
    spec = importlib.util.spec_from_file_location(
        "m1_110_profile_unit", PROFILE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load M1-110 profile module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class M1110PrefillStageProfileUnitTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = RUNNER.read_text(encoding="utf-8")
        cls.build = BUILD.read_text(encoding="utf-8")
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.profile_source = PROFILE.read_text(encoding="utf-8")
        cls.profile = load_profile_module()

    def test_stage_matrix_helpers_fail_closed(self) -> None:
        self.assertEqual(
            self.profile.median_rows([
                [1.0, 20.0, 3.0],
                [3.0, 10.0, 1.0],
                [2.0, 30.0, 2.0],
            ]),
            [2.0, 20.0, 2.0],
        )
        for rows in ([], [[1.0], [1.0, 2.0]]):
            with self.assertRaises(ValueError):
                self.profile.median_rows(rows)
        for value in (True, -1.0, float("inf"), float("nan")):
            self.assertFalse(self.profile.finite_nonnegative(value))

    def test_profile_source_records_fixed_pipeline_boundaries(self) -> None:
        for marker in (
            "constexpr int kProfileStageCount = 8;",
            "ProfileStage::kInit",
            "ProfileStage::kGather",
            "ProfileStage::kQk",
            "ProfileStage::kMask",
            "ProfileStage::kSoftmax",
            "ProfileStage::kPv",
            "ProfileStage::kMerge",
            "ProfileStage::kFinalize",
            'module.def("forward"',
            'module.def("profile"',
        ):
            self.assertIn(marker, self.source)
        self.assertIn(
            "2\n        + 6 * (",
            self.profile_source,
        )
        self.assertIn("MAX_RELATIVE_L2 = 1e-5", self.profile_source)
        self.assertIn(
            '"main_or_yaml_change_authorized": False',
            self.profile_source,
        )

    def test_build_has_a_distinct_profile_module(self) -> None:
        self.assertIn(
            "-DTORCH_EXTENSION_NAME=corex_fused_paged_prefill_profile",
            self.build,
        )
        self.assertIn(
            "corex_fused_paged_prefill_stage_profile_ext.cu",
            self.build,
        )
        self.assertNotIn("computility-run.yaml", self.build)

    def test_runner_fixes_one_production_case_per_gpu(self) -> None:
        for marker in (
            "production_dense_q8176",
            "production_65k_q8176",
            "production_128k_q8176",
            "production_235k_q5616",
            "for gpu in 0 1 2 3",
            'CUDA_VISIBLE_DEVICES="$GPU"',
            "profile_m1_110_fused_prefill_stages.py",
            "bi100-m1-110-fused-prefill-stage-profile-matrix-v1",
        ):
            self.assertIn(marker, self.runner)
        self.assertIn(
            '"deeper_fusion_design_selection_authorized": not reasons',
            self.runner,
        )
        self.assertIn(
            '"tp4_service_experiment_authorized": False',
            self.runner,
        )

    def test_runner_uses_scoped_graceful_cleanup(self) -> None:
        for marker in (
            "setsid",
            "bi100_stop_process_group",
            '"$pid" "$pid" 60 20',
            "trap finish EXIT",
            "timeout --foreground --signal=TERM --kill-after=60s",
            "service_postflight_gate.py",
            "bi100_preflight.py",
            "compare_bi100_preflights.py",
            "fatal_scan.rc",
        ):
            self.assertIn(marker, self.runner)
        for forbidden in ("pkill", "killall", "git push"):
            self.assertNotIn(forbidden, self.runner)

    def test_invalid_invocation_fails_before_runtime_access(self) -> None:
        result = subprocess.run(
            ["bash", str(RUNNER)],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)


if __name__ == "__main__":
    unittest.main()
