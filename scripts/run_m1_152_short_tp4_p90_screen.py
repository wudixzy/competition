#!/usr/bin/env python3
"""Run one P90-oriented TP4 arm with cold and partial-prefix requests."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import sys

from record_experiment_timeline import summarize
from run_m1_140_activation_capture import _atomic_json, _run_to_files, _sha256
from run_m1_141_short_tp4_screen import (
    ShortTp4Runner,
    _hex,
    _mapping,
    _sha256_if_file,
)


TARGETS = [8192, 16384, 24576, 32768, 49152, 65536]
PARTIAL_TARGETS = [16384, 32768, 49152, 65536]
PARTIAL_RESIDUAL_TOKENS = 8192


def validate_p90_operator_authorization(
    status_path: Path,
    *,
    candidate_extension_sha256: str,
    kernel_source_sha256: str,
) -> dict:
    status_path = status_path.resolve(strict=True)
    if (
        not status_path.is_relative_to(Path("/tmp"))
        or status_path.stat().st_mode & 0o077
    ):
        raise ValueError(
            "P90 operator authorization must be private under /tmp")
    status = _mapping(json.loads(
        status_path.read_text(encoding="ascii")))
    screen = _mapping(status.get("screen"))
    authorization = _mapping(status.get("authorization"))
    lifecycle = _mapping(status.get("lifecycle"))
    privacy = _mapping(status.get("privacy"))
    rows = screen.get("rows")
    expected_cases = [
        f"p90_total_{total // 1024:02d}k_q8176"
        for total in range(8192, 65537, 8192)
    ]
    expected_lengths = list(range(8176, 65521, 8192))
    if (
        status.get("schema")
        != "bi100-m1-149-ttft-p90-prefill-grid-v1"
        or status.get("version") != 1
        or status.get("qualified") is not True
        or not _hex(status.get("source_revision"), 40)
        or status.get("extension_sha256")
        != candidate_extension_sha256
        or status.get("fixed_cases") != expected_cases
        or status.get("gpu_count", 0) < 1
        or status.get("gpu_count") != len(status.get("gpus", []))
        or len(set(status.get("gpus", []))) != status.get("gpu_count")
        or screen.get("qualified") is not True
        or screen.get("reasons") != []
        or not isinstance(rows, list)
        or len(rows) != len(expected_cases)
        or [row.get("case") for row in rows] != expected_cases
        or [row.get("total_kv_len") for row in rows] != expected_lengths
        or any(
            row.get("qualified") is not True
            or row.get("finite") is not True
            or not isinstance(row.get("speedup"), (int, float))
            or isinstance(row.get("speedup"), bool)
            or not math.isfinite(float(row["speedup"]))
            or row["speedup"] < 1.2
            or not isinstance(
                row.get("output_relative_l2"), (int, float))
            or isinstance(row.get("output_relative_l2"), bool)
            or not math.isfinite(float(row["output_relative_l2"]))
            or row["output_relative_l2"] > 1.0e-5
            or not isinstance(
                row.get("lse_relative_l2"), (int, float))
            or isinstance(row.get("lse_relative_l2"), bool)
            or not math.isfinite(float(row["lse_relative_l2"]))
            or row["lse_relative_l2"] > 1.0e-5
            or not isinstance(row.get("output_max_abs"), (int, float))
            or isinstance(row.get("output_max_abs"), bool)
            or not math.isfinite(float(row["output_max_abs"]))
            or row["output_max_abs"] > 1.0e-3
            for row in rows
        )
        or not isinstance(screen.get("minimum_speedup"), (int, float))
        or isinstance(screen.get("minimum_speedup"), bool)
        or screen["minimum_speedup"] < 1.2
        or not isinstance(screen.get("median_speedup"), (int, float))
        or isinstance(screen.get("median_speedup"), bool)
        or not math.isfinite(float(screen["median_speedup"]))
        or screen["median_speedup"] < screen["minimum_speedup"]
        or lifecycle != {
            "after_preflight_qualified": True,
            "cleanup_reaped": True,
            "fatal_scan_qualified": True,
            "postflight_qualified": True,
            "preflight_comparison_qualified": True,
            "source_unchanged": True,
        }
        or privacy != {
            "credentials_recorded": False,
            "model_outputs_recorded": False,
            "prompts_recorded": False,
            "token_ids_recorded": False,
        }
        or authorization != {
            "short_tp4_p90_screen_authorized": True,
            "l2_capture_authorized": False,
            "main_or_yaml_change_authorized": False,
            "official_score_claim_authorized": False,
        }
    ):
        raise ValueError(
            "P90 operator evidence does not authorize the TP4 screen")
    identity_path = (
        status_path.parent / "identity.json").resolve(strict=True)
    if (
        identity_path.parent != status_path.parent
        or identity_path.stat().st_mode & 0o077
    ):
        raise ValueError("P90 operator identity is not private and adjacent")
    identity = _mapping(json.loads(
        identity_path.read_text(encoding="ascii")))
    if (
        identity.get("extension_sha256")
        != candidate_extension_sha256
        or identity.get("kernel_source_sha256")
        != kernel_source_sha256
    ):
        raise ValueError("P90 operator artifact identity differs")
    return {
        "runner_status_sha256": _sha256(status_path),
        "identity_sha256": _sha256(identity_path),
        "source_revision": status["source_revision"],
        "candidate_extension_sha256": candidate_extension_sha256,
        "kernel_source_sha256": kernel_source_sha256,
        "minimum_speedup": screen["minimum_speedup"],
        "median_speedup": screen["median_speedup"],
        "case_count": len(rows),
    }


class P90ShortTp4Runner(ShortTp4Runner):

    def validate(self) -> None:
        super().validate()
        self.run_id = (
            f"m1-152-{self.args.selector}-"
            f"{self.source_revision[:12]}")
        kernel_sha = _sha256(
            self.root / "qwen3_6_scripts"
            / "corex_fused_paged_prefill_split4.cu")
        self.p90_authorization = validate_p90_operator_authorization(
            self.args.p90_operator_status,
            candidate_extension_sha256=self.l2_authorization[
                "candidate_extension_sha256"],
            kernel_source_sha256=kernel_sha,
        )

    def run_requests(self) -> None:
        rc = _run_to_files(
            [
                sys.executable,
                str(
                    self.root / "tests"
                    / "short_tp4_p90_funnel_service.py"),
                "--base", "http://127.0.0.1:8000",
                "--model-path", str(self.model_path.resolve()),
                "--timeout-s", "1800",
                "--run-id", self.run_id,
                "--prompt-set-id", self.args.pair_id,
                "--selector", self.args.selector,
                "--out", str(self.run_root / "measurement.json"),
            ],
            self.run_root / "measurement.stdout",
            self.run_root / "measurement.stderr",
            environment=self.base_environment(),
            timeout_s=7200,
        )
        if rc:
            raise RuntimeError("P90-oriented TP4 request matrix failed")

    def write_status(self, returncode: int) -> None:
        timeline_report = summarize(
            self.timeline, expected_run_id=self.run_id)
        _atomic_json(self.run_root / "timeline_report.json", timeline_report)
        status = {
            "schema": "bi100-m1-152-short-tp4-p90-screen-runner-v3",
            "version": 3,
            "qualified": returncode == 0,
            "returncode": returncode,
            "terminal_stage": self.current_stage,
            "error_type": self.error_type,
            "run_id": self.run_id,
            "source_revision": self.source_revision,
            "source_branch": self.source_branch,
            "instance": self.args.instance,
            "selector": self.args.selector,
            "pair_id": self.args.pair_id,
            "runtime_identity": self.runtime_identity,
            "gpu_count": 4,
            "tensor_parallel_size": 4,
            "service_startups": 1,
            "targets": TARGETS,
            "partial_targets": PARTIAL_TARGETS,
            "partial_residual_tokens": PARTIAL_RESIDUAL_TOKENS,
            "repetitions": 1,
            "gates": self.gates,
            "artifact_sha256": {
                name: _sha256_if_file(self.run_root / name)
                for name in (
                    "runtime_identity.json", "measurement.json",
                    "dispatch_count.txt", "fatal_scan.json",
                    "postflight_after.json", "preflight_comparison.json",
                    "timeline_report.json",
                )
            },
            "candidate_extension": {
                "sha256": self.candidate_extension_sha256,
                "size_bytes": (
                    self.candidate_extension.stat().st_size
                    if self.candidate_extension is not None else None
                ),
                "external_override_active": (
                    self.args.selector == "candidate"),
            },
            "dispatch_count": self.dispatch_count,
            "kernel_source_sha256": _sha256(
                self.root / "qwen3_6_scripts"
                / "corex_fused_paged_prefill_split4.cu"),
            "l2_authorization": self.l2_authorization,
            "p90_operator_authorization": self.p90_authorization,
            "timing": {
                "wall_span_s": timeline_report["wall_span_s"],
                "summed_stage_s": timeline_report["summed_stage_s"],
            },
            "authorization": {
                "long_context_confirmation_authorized": False,
                "full_capability_authorized": False,
                "main_or_yaml_change_authorized": False,
            },
            "privacy": {
                "prompts_recorded": False,
                "model_outputs_recorded": False,
                "token_ids_recorded": False,
                "credentials_recorded": False,
            },
        }
        _atomic_json(self.run_root / "runner_status.json", status)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("instance")
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--selector", choices=("control", "candidate"), required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--candidate-extension", type=Path)
    parser.add_argument("--expected-candidate-sha256")
    parser.add_argument("--l2-qualification", type=Path, required=True)
    parser.add_argument("--l2-runner-status", type=Path, required=True)
    parser.add_argument("--p90-operator-status", type=Path, required=True)
    args = parser.parse_args()
    args.profile = "short"
    args.targets = ",".join(map(str, TARGETS))
    args.contexts = ""
    args.ordinals = ""
    runner = P90ShortTp4Runner(args)

    def _interrupt(signum, _frame):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    try:
        return runner.run()
    except BaseException as exc:
        if runner.run_root.is_dir() and runner.run_id:
            runner.error_type = type(exc).__name__
            try:
                runner.cleanup_service()
            except BaseException:
                pass
            try:
                runner.write_status(1)
            except BaseException:
                pass
        print(
            f"M1-152 P90 TP4 runner failed: {type(exc).__name__}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
