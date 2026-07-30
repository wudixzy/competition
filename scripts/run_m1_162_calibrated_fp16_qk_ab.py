#!/usr/bin/env python3
"""Run the calibrated M1-162 FP16-QK screen on three healthy BI100 cards."""

from __future__ import annotations

import run_m1_157_fp16_qk_ab as runner


runner.CELL_SCRIPT = (
    runner.ROOT / "tests" / "bench_m1_162_calibrated_fp16_qk_ab.py")
runner.RUNNER_SCHEMA = "bi100-m1-162-calibrated-fp16-qk-ab-runner-v1"
runner.SCREEN_SCHEMA = "bi100-m1-162-calibrated-fp16-qk-ab-screen-v1"
runner.RUNTIME_IDENTITY = "corex-3.2.3-m1-162-calibrated"


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
