#!/usr/bin/env python3
"""Run one short TP4 arm with 4K/32K/65K batched in one service."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from run_m1_140_activation_capture import CaptureRunner, _atomic_json, _sha256
from record_experiment_timeline import summarize


class ShortTp4Runner(CaptureRunner):

    def validate(self) -> None:
        if (
            self.run_root == Path("/tmp")
            or not self.run_root.is_relative_to(Path("/tmp"))
            or self.run_root.exists()
        ):
            raise ValueError("run root must be a new private path under /tmp")
        status = subprocess.run(
            [
                "git", "-C", str(self.root), "status", "--porcelain",
                "--untracked-files=all", "--", ".",
                ":(exclude)bench_runs/**",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if status:
            raise RuntimeError("short TP4 runner requires a clean source tree")
        self.source_revision = subprocess.run(
            ["git", "-C", str(self.root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        self.source_branch = subprocess.run(
            ["git", "-C", str(self.root), "branch", "--show-current"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        self.run_id = (
            f"m1-141-{self.args.selector}-"
            f"{self.source_revision[:12]}")
        if (
            not self.runtime_site.is_absolute()
            or not (self.runtime_site / "vllm").is_dir()
            or not (self.runtime_site / "transformers").is_dir()
        ):
            raise RuntimeError("immutable runtime overlay is missing")
        if not self.runtime_install.is_file():
            candidate = self.runtime_site.parent / "install.json"
            if candidate.is_file():
                self.runtime_install = candidate
            else:
                raise RuntimeError("runtime install report is missing")
        if not self.model_path.is_dir():
            raise RuntimeError("model path is missing")
        from run_m1_140_activation_capture import _port_free
        if not _port_free():
            raise RuntimeError("API port 8000 is already occupied")

    def prepare(self) -> None:
        self.run_root.mkdir(mode=0o700, parents=True)
        (self.run_root / "runtime-workdir").mkdir(mode=0o700)
        for name, value in {
            "source_revision.txt": self.source_revision,
            "source_branch.txt": self.source_branch,
            "instance.txt": self.args.instance,
            "model_path.txt": str(self.model_path.resolve()),
            "runtime_site_packages.txt": str(self.runtime_site.resolve()),
        }.items():
            (self.run_root / name).write_text(
                value + "\n", encoding="ascii")

    def service_environment(self) -> dict[str, str]:
        environment = self.base_environment()
        environment.update({
            "BI100_RUNTIME_SITE_PACKAGES": str(self.runtime_site),
            "BI100_RUNTIME_INSTALL_REPORT": str(self.runtime_install),
            "BI100_RUNTIME_WORKDIR": str(
                self.run_root / "runtime-workdir"),
            "MODEL_PATH": str(self.model_path.resolve()),
            "HOST": "0.0.0.0",
            "PORT": "8000",
            "ENABLE_CUSTOM_IPC": "1",
            "VLLM_ENGINE_ITERATION_TIMEOUT_S": "3600",
            "BI100_MOE_COREX_DIRECT_ROUTED": "1",
            "BI100_GDN_COREX_PACKED_DECODE": "1",
            "BI100_GDN_COMBINED_QK_NORM": "0",
            "BI100_GDN_CACHE_POLICY": "admission64",
            "BI100_GDN_RESTORE_MODE": "hybrid64",
            "BI100_HYBRID_KV_ACCOUNTING": "full_attention",
            "BI100_CPU_KV_OFFLOAD": "0",
            "BI100_BLOCK_MAJOR_CPU_KV": "0",
            "BI100_CACHE_TRACE": "0",
            "BI100_ATTN_COREX_FUSED_PREFILL": (
                "1" if self.args.selector == "candidate" else "0"),
            "BI100_ATTN_COREX_FUSED_PREFILL_SHADOW": "0",
            "BI100_ATTN_CAPTURE_REPLAY": "0",
            "BI100_PROFILE": "0",
            "BI100_PROFILE_INCLUDE_STARTUP": "0",
            "BI100_PAGED_ATTN_DIAGNOSTICS": "0",
            "BI100_GDN_ALLOW_NAN_ZERO": "0",
            "BI100_GDN_FINITE_CHECK": "0",
        })
        return environment

    def run_requests(self) -> None:
        from run_m1_140_activation_capture import _run_to_files

        rc = _run_to_files(
            [
                sys.executable,
                str(self.root / "tests" / "short_tp4_funnel_service.py"),
                "--base", "http://127.0.0.1:8000",
                "--model-path", str(self.model_path.resolve()),
                "--targets", "4096,32768,65536",
                "--max-tokens", "8",
                "--timeout-s", "1800",
                "--run-id", self.run_id,
                "--selector", self.args.selector,
                "--out", str(self.run_root / "measurement.json"),
            ],
            self.run_root / "measurement.stdout",
            self.run_root / "measurement.stderr",
            environment=self.base_environment(),
            timeout_s=7200,
        )
        if rc:
            raise RuntimeError("short TP4 request matrix failed")

    def verify_dispatch(self) -> None:
        log = (self.run_root / "server.log").read_text(
            encoding="utf-8", errors="replace")
        count = log.count("path=corex_split4")
        expected_candidate = self.args.selector == "candidate"
        if expected_candidate and count < 2:
            raise RuntimeError("candidate fused-prefill dispatch is absent")
        if not expected_candidate and count:
            raise RuntimeError("control unexpectedly used fused prefill")
        (self.run_root / "dispatch_count.txt").write_text(
            f"{count}\n", encoding="ascii")

    def write_status(self, returncode: int) -> None:
        timeline_report = summarize(
            self.timeline, expected_run_id=self.run_id)
        _atomic_json(self.run_root / "timeline_report.json", timeline_report)
        status = {
            "schema": "bi100-m1-141-short-tp4-screen-runner-v1",
            "version": 1,
            "qualified": returncode == 0,
            "returncode": returncode,
            "terminal_stage": self.current_stage,
            "error_type": self.error_type,
            "source_revision": self.source_revision,
            "source_branch": self.source_branch,
            "instance": self.args.instance,
            "selector": self.args.selector,
            "gpu_count": 4,
            "tensor_parallel_size": 4,
            "service_startups": 1,
            "targets": [4096, 32768, 65536],
            "gates": self.gates,
            "artifact_sha256": {
                name: _sha256(self.run_root / name)
                for name in (
                    "runtime_identity.json", "measurement.json",
                    "fatal_scan.json", "postflight_after.json",
                    "preflight_comparison.json", "timeline_report.json",
                )
            },
            "timing": {
                "wall_span_s": timeline_report["wall_span_s"],
                "summed_stage_s": timeline_report["summed_stage_s"],
            },
            "authorization": {
                "long_context_authorized": False,
                "full_capability_authorized": False,
                "main_or_yaml_change_authorized": False,
            },
        }
        _atomic_json(self.run_root / "runner_status.json", status)

    def run(self) -> int:
        self.validate()
        self.prepare()
        primary_error = None
        try:
            with self.stage("postflight_before"):
                self.run_postflight("postflight_before")
            with self.stage("preflight_before"):
                self.run_preflight("preflight_before")
            with self.stage("runtime_identity"):
                self.verify_runtime()
            with self.stage("service_startup"):
                self.start_service()
            with self.stage("request_matrix"):
                self.run_requests()
            with self.stage("dispatch"):
                self.verify_dispatch()
            with self.stage("health_after_requests"):
                from run_m1_140_activation_capture import _health
                if not _health():
                    raise RuntimeError("service health failed")
        except BaseException as exc:
            primary_error = exc
            self.error_type = type(exc).__name__
        try:
            with self.stage("scoped_cleanup"):
                self.cleanup_service()
        except BaseException as exc:
            if primary_error is None:
                primary_error = exc
                self.error_type = type(exc).__name__
        try:
            with self.stage("fatal_scan"):
                if any(self.scan_fatal().values()):
                    raise RuntimeError("fatal log categories were observed")
            with self.stage("postflight_after"):
                self.run_postflight("postflight_after")
            with self.stage("preflight_after"):
                self.run_preflight("preflight_after")
            with self.stage("preflight_comparison"):
                self.compare_preflights()
            with self.stage("source_unchanged"):
                self.source_unchanged()
        except BaseException as exc:
            if primary_error is None:
                primary_error = exc
                self.error_type = type(exc).__name__
        returncode = 0 if primary_error is None else 1
        self.current_stage = "complete" if returncode == 0 else self.current_stage
        self.write_status(returncode)
        return returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("instance")
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--selector", choices=("control", "candidate"), required=True)
    args = parser.parse_args()
    args.profile = "short"
    args.targets = "4096,32768,65536"
    args.contexts = ""
    args.ordinals = ""
    runner = ShortTp4Runner(args)

    def _interrupt(signum, _frame):
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
            f"M1-141 short TP4 runner failed: {type(exc).__name__}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
