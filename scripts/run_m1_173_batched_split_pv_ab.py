#!/usr/bin/env python3
"""Screen one FP32 batched split-PV call against the FP16-QK baseline."""

from __future__ import annotations

import run_m1_157_fp16_qk_ab as runner


runner.CELL_SCRIPT = (
    runner.ROOT / "tests" / "bench_m1_162_calibrated_fp16_qk_ab.py")
runner.RUNNER_SCHEMA = "bi100-m1-173-batched-split-pv-ab-runner-v1"
runner.SCREEN_SCHEMA = "bi100-m1-173-batched-split-pv-ab-screen-v1"
runner.RUNTIME_IDENTITY = "corex-3.2.3-m1-173-batched-split-pv"
runner.BASELINE_MODULE_NAME = "corex_fused_paged_prefill_fp16_qk"


def _authorization(qualified: bool) -> dict[str, bool]:
    return {
        "real_activation_replay_authorized": qualified,
        "short_tp4_screen_authorized": False,
        "long_context_or_quality_authorized": False,
        "main_or_yaml_change_authorized": False,
    }


runner.screen_authorization = _authorization


if __name__ == "__main__":
    raise SystemExit(runner.main())
