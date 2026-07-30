from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_m1_136_fused_prefill_shadow.sh"
WRAPPER = ROOT / "scripts" / "run_m1_163_fp16_qk_calibrated_shadow.sh"
BUILD = (
    ROOT
    / "qwen3_6_scripts"
    / "build_corex_fused_paged_prefill_fp16_qk_runtime.sh"
)
CONTRACT = (
    ROOT / "quality/fused_prefill_real_activation_adjudication.v2.json")
CONTRACT_SHA256 = (
    "ba37338f4d4112a1bd90e3e700334652a66ebb048f3cea7379ed21cdd3f3aceb"
)


class M1163Fp16QkShadowRunnerTest(unittest.TestCase):
    def test_contract_is_frozen_and_keeps_semantic_gates_separate(self):
        self.assertEqual(
            hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
            CONTRACT_SHA256,
        )
        value = json.loads(CONTRACT.read_text(encoding="utf-8"))
        hard = value["hard_gates"]
        self.assertEqual(
            hard["candidate_vs_rounded_relative_l2_role"],
            "diagnostic_only",
        )
        self.assertFalse(hard["semantic_evidence_may_waive_failure"])
        self.assertTrue(value["execution"]["task_capability_still_required"])
        self.assertFalse(
            value["promotion"]["production_promotion_authorized"])

    def test_runtime_build_keeps_candidate_math_and_loader_module_name(self):
        source = BUILD.read_text(encoding="utf-8")
        self.assertIn(
            "corex_fused_paged_prefill_fp16_qk.cu", source)
        self.assertIn(
            "-DTORCH_EXTENSION_NAME=corex_fused_paged_prefill", source)
        self.assertIn(
            "OUTPUT=${OUTPUT_DIR}/corex_fused_paged_prefill.so", source)

    def test_wrapper_selects_only_v2_and_binds_private_extension(self):
        source = WRAPPER.read_text(encoding="utf-8")
        self.assertIn(
            "BI100_FUSED_PREFILL_SHADOW_VARIANT=calibrated_v2", source)
        self.assertIn(
            "BI100_FUSED_PREFILL_SHADOW_EXTENSION=$2", source)
        self.assertIn(
            'exec "$ROOT/scripts/run_m1_136_fused_prefill_shadow.sh"',
            source,
        )

    def test_runner_hash_binds_extension_and_fails_closed(self):
        source = RUNNER.read_text(encoding="utf-8")
        for fragment in (
            "EXPERIMENT_LABEL=M1-163",
            "NUMERIC_MODE=calibrated_v2",
            "REQUIRE_EXTERNAL_EXTENSION=1",
            "fused_prefill_real_activation_adjudication.v2.json",
            "QUALIFIER_VERSION=2",
            "candidate extension must be a non-writable file under /tmp",
            "BI100_ATTN_COREX_FUSED_PREFILL_EXTENSION=",
            "BI100_ATTN_COREX_FUSED_PREFILL_EXTENSION_SHA256=",
            '--contract-version "$QUALIFIER_VERSION"',
            '"production_promotion_authorized": False',
            '"yaml_change_authorized": False',
            '"main_merge_authorized": False',
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)


if __name__ == "__main__":
    unittest.main()
