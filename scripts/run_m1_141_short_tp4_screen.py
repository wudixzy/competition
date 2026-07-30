#!/usr/bin/env python3
"""Run one short TP4 arm with 4K/32K/65K batched in one service."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

from run_m1_140_activation_capture import CaptureRunner, _atomic_json, _sha256
from record_experiment_timeline import summarize


def _hex(value, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(value) -> dict:
    return value if isinstance(value, dict) else {}


def _finite_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _sha256_if_file(path: Path) -> str | None:
    return _sha256(path) if path.is_file() else None


def validate_l2_authorization(
    qualification_path: Path,
    runner_status_path: Path,
    *,
    experiment_contract_sha256: str,
    numeric_contract_sha256: str,
) -> dict:
    qualification_path = qualification_path.resolve(strict=True)
    runner_status_path = runner_status_path.resolve(strict=True)
    if (
        not qualification_path.is_relative_to(Path("/tmp"))
        or not runner_status_path.is_relative_to(Path("/tmp"))
        or qualification_path.stat().st_mode & 0o077
        or runner_status_path.stat().st_mode & 0o077
    ):
        raise ValueError("L2 authorization files must be private under /tmp")
    qualification = _mapping(json.loads(
        qualification_path.read_text(encoding="ascii")))
    status = _mapping(json.loads(
        runner_status_path.read_text(encoding="ascii")))
    extension = _mapping(qualification.get("candidate_extension"))
    authorization = _mapping(qualification.get("authorization"))
    manifests = qualification.get("bank_manifest_sha256s")
    identity_fields = (
        qualification.get("capture_source_revision"),
        qualification.get("candidate_source_revision"),
        qualification.get("runtime_identity"),
        qualification.get("instance"),
        qualification.get("activation_run_id"),
    )
    if (
        qualification.get("schema")
        != "bi100-fused-prefill-activation-replay-qualification-v1"
        or qualification.get("version") != 1
        or qualification.get("profile") != "qualification"
        or qualification.get("execution_valid") is not True
        or qualification.get("stage_qualified") is not True
        or any(
            qualification.get(name)
            for name in (
                "invalid_reasons", "numeric_reasons",
                "performance_reasons", "coverage_reasons",
            )
        )
        or qualification.get("report_count") != 4
        or qualification.get("record_count") != 36
        or qualification.get("ranks") != [0, 1, 2, 3]
        or not _hex(identity_fields[0], 40)
        or not _hex(identity_fields[1], 40)
        or not all(
            isinstance(value, str) and value
            for value in identity_fields[2:]
        )
        or not isinstance(manifests, list)
        or len(manifests) != 4
        or len(set(manifests)) != 4
        or not all(_hex(value, 64) for value in manifests)
        or not _hex(extension.get("sha256"), 64)
        or not isinstance(extension.get("size_bytes"), int)
        or isinstance(extension["size_bytes"], bool)
        or extension["size_bytes"] <= 0
        or not _finite_number(
            qualification.get("median_candidate_speedup"))
        or qualification["median_candidate_speedup"] < 1.05
        or not _finite_number(
            qualification.get("minimum_case_speedup"))
        or qualification["minimum_case_speedup"] < 0.98
        or qualification.get("contract_sha256")
        != experiment_contract_sha256
        or qualification.get("numeric_contract_sha256")
        != numeric_contract_sha256
        or authorization != {
            "short_tp4_authorized": True,
            "long_context_authorized": False,
            "main_or_yaml_change_authorized": False,
        }
    ):
        raise ValueError("L2 replay qualification does not authorize L3")

    status_authorization = _mapping(status.get("authorization"))
    artifact_sha = _mapping(status.get("artifact_sha256"))
    if (
        status.get("schema")
        != "bi100-m1-140-activation-replay-runner-v1"
        or status.get("version") != 1
        or status.get("qualified") is not True
        or not isinstance(status.get("returncode"), int)
        or isinstance(status["returncode"], bool)
        or status.get("returncode") != 0
        or status.get("terminal_stage") != "complete"
        or status.get("profile") != "qualification"
        or status.get("gpu_count") != 4
        or status.get("parallel_rank_replays") != 4
        or status.get("capture_source_revision") != identity_fields[0]
        or status.get("candidate_source_revision") != identity_fields[1]
        or status.get("runtime_identity") != identity_fields[2]
        or status.get("instance") != identity_fields[3]
        or status.get("candidate_extension_sha256")
        != extension["sha256"]
        or artifact_sha.get("qualification.json")
        != _sha256(qualification_path)
        or not all(
            _hex(artifact_sha.get(name), 64)
            for name in (
                "timeline_report.json", "preflight_comparison.json",
                "final_postflight.json",
            )
        )
        or status_authorization != {
            "short_tp4_authorized": True,
            "long_context_authorized": False,
            "main_or_yaml_change_authorized": False,
        }
        or _mapping(status.get("privacy")).get(
            "credentials_recorded") is not False
    ):
        raise ValueError("L2 replay runner identity does not authorize L3")
    return {
        "qualification_sha256": _sha256(qualification_path),
        "runner_status_sha256": _sha256(runner_status_path),
        "candidate_extension_sha256": extension["sha256"],
        "candidate_extension_size_bytes": extension["size_bytes"],
        "capture_source_revision": identity_fields[0],
        "replay_source_revision": identity_fields[1],
        "runtime_identity": identity_fields[2],
        "activation_run_id": identity_fields[4],
    }


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
        if (
            not 1 <= len(self.args.pair_id) <= 128
            or any(
                character not in (
                    "abcdefghijklmnopqrstuvwxyz"
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                )
                for character in self.args.pair_id
            )
        ):
            raise ValueError("short TP4 pair identity is invalid")
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
        self.l2_authorization = validate_l2_authorization(
            self.args.l2_qualification,
            self.args.l2_runner_status,
            experiment_contract_sha256=_sha256(
                self.root / "quality" / "experiment_funnel.v1.json"),
            numeric_contract_sha256=_sha256(
                self.root / "quality"
                / "fused_prefill_numeric_adjudication.v1.json"),
        )
        self.dispatch_count: int | None = None
        self.candidate_extension: Path | None = None
        self.candidate_extension_sha256: str | None = None
        if self.args.selector == "candidate":
            if (
                self.args.candidate_extension is None
                or self.args.expected_candidate_sha256 is None
            ):
                raise ValueError(
                    "candidate selector requires an extension and SHA-256")
            expected = self.args.expected_candidate_sha256
            if (
                len(expected) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in expected
                )
            ):
                raise ValueError("candidate extension SHA-256 is invalid")
            if (
                expected
                != self.l2_authorization[
                    "candidate_extension_sha256"]
            ):
                raise ValueError(
                    "candidate extension was not authorized by L2")
            extension = self.args.candidate_extension.resolve(strict=True)
            if (
                not extension.is_file()
                or not extension.is_relative_to(Path("/tmp"))
                or extension.stat().st_size <= 0
                or extension.stat().st_mode & 0o022
                or _sha256(extension) != expected
            ):
                raise ValueError(
                    "candidate extension identity or permissions differ")
            self.candidate_extension = extension
            self.candidate_extension_sha256 = expected
        elif (
            self.args.candidate_extension is not None
            or self.args.expected_candidate_sha256 is not None
        ):
            raise ValueError(
                "control selector cannot receive a candidate extension")

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
        if self.args.selector == "candidate":
            environment.update({
                "BI100_ATTN_COREX_FUSED_PREFILL_EXTENSION": str(
                    self.candidate_extension),
                "BI100_ATTN_COREX_FUSED_PREFILL_EXTENSION_SHA256": str(
                    self.candidate_extension_sha256),
            })
        else:
            environment.pop(
                "BI100_ATTN_COREX_FUSED_PREFILL_EXTENSION", None)
            environment.pop(
                "BI100_ATTN_COREX_FUSED_PREFILL_EXTENSION_SHA256", None)
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
                "--repetitions", "3",
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
        self.dispatch_count = count

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
            "targets": [4096, 32768, 65536],
            "repetitions": 3,
            "gates": self.gates,
            "artifact_sha256": {
                name: _sha256_if_file(self.run_root / name)
                for name in (
                    "runtime_identity.json", "measurement.json",
                    "dispatch_count.txt",
                    "fatal_scan.json", "postflight_after.json",
                    "preflight_comparison.json", "timeline_report.json",
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
            "timing": {
                "wall_span_s": timeline_report["wall_span_s"],
                "summed_stage_s": timeline_report["summed_stage_s"],
            },
            "authorization": {
                "long_context_authorized": False,
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
        primary_error = self.run_postconditions(primary_error)
        returncode = 0 if primary_error is None else 1
        self.current_stage = (
            "complete"
            if returncode == 0
            else (self.failed_stage or self.current_stage)
        )
        self.write_status(returncode)
        return returncode


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
    args = parser.parse_args()
    args.profile = "short"
    args.targets = "4096,32768,65536"
    args.contexts = ""
    args.ordinals = ""
    runner = ShortTp4Runner(args)

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
            f"M1-141 short TP4 runner failed: {type(exc).__name__}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
