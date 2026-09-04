#!/usr/bin/env python3
"""Run one lean full-model TP4 arm for an attention-only candidate."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Callable

from record_experiment_timeline import summarize
from run_m1_140_activation_capture import (
    CaptureRunner, _atomic_json, _health, _run_to_files,
)


EXPECTED_MODEL_PATH = Path(
    "/root/public-storage/models/Qwen/Qwen3.6-35B-A3B")
TARGETS = (16384, 32768, 65536)
REPETITIONS = 3
MAX_TOKENS = 8
TEACHER_FORCED_TARGETS = (4096, 16384, 32768, 65536)


def _identifier(value: str) -> bool:
    return 1 <= len(value) <= 128 and all(
        item.isalnum() or item in "._-" for item in value)


def _workload_config(
    raw_targets: str,
    repetitions: int,
) -> tuple[tuple[int, ...], int]:
    try:
        targets = tuple(int(value) for value in raw_targets.split(","))
    except ValueError as exc:
        raise ValueError("targets must be comma-separated integers") from exc
    if (not targets or targets != tuple(sorted(set(targets)))
            or any(target <= MAX_TOKENS
                   or target + MAX_TOKENS > 262144 for target in targets)
            or not 1 <= repetitions <= 3):
        raise ValueError("focused workload configuration is invalid")
    return targets, repetitions


def _api_listener_absent(port: int = 8000) -> bool:
    with socket.socket() as connection:
        connection.settimeout(0.25)
        return connection.connect_ex(("127.0.0.1", port)) != 0


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True,
        capture_output=True, text=True).stdout.strip()


def _load_session_preflight(path: Path, instance: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(value, dict)
            or value.get("schema") != "bi100-session-preflight-v1"
            or value.get("version") != 1
            or value.get("qualified") is not True
            or value.get("instance") != instance
            or not isinstance(value.get("session_preflight_id"), str)
            or not value["session_preflight_id"]
            or value.get("gpu_indices") != [0, 1, 2, 3]
            or value.get("gpu_health_qualified") is not True
            or value.get("fp16_matmul_qualified") is not True
            or value.get("collective_qualified") is not True):
        raise ValueError("reusable four-card session preflight is invalid")
    return value


class AttentionOperatorTp4Runner(CaptureRunner):

    def validate(self) -> None:
        if self.run_root.exists() or not self.run_root.is_absolute():
            raise ValueError("run root must be a new absolute path")
        if not _identifier(self.args.pair_id):
            raise ValueError("pair identity is invalid")
        self.targets, self.repetitions = _workload_config(
            self.args.targets, self.args.repetitions)
        self.workload_mode = getattr(self.args, "workload", "performance")
        if (self.workload_mode == "teacher_forced"
                and (self.targets != TEACHER_FORCED_TARGETS
                     or self.repetitions != 1)):
            raise ValueError(
                "teacher-forced workload requires fixed targets and one request")
        if not _api_listener_absent():
            raise RuntimeError("API port 8000 has an active listener")
        self.source_revision = _git(self.root, "rev-parse", "HEAD")
        self.source_branch = _git(self.root, "branch", "--show-current")
        self.source_dirty_summary = _git(
            self.root, "status", "--short", "--untracked-files=all") or "clean"
        self.run_id = (
            f"attention-tp4-{self.workload_mode}-{self.args.selector}-"
            f"{self.source_revision[:10]}")
        if (self.model_path != EXPECTED_MODEL_PATH
                or not self.model_path.is_dir()):
            raise RuntimeError("focused TP4 requires the fixed full model")
        if (not self.runtime_site.is_absolute()
                or not (self.runtime_site / "vllm").is_dir()
                or not (self.runtime_site / "transformers").is_dir()):
            raise RuntimeError("runtime overlay is missing")
        if not self.runtime_install.is_file():
            candidate = self.runtime_site.parent / "install.json"
            if not candidate.is_file():
                raise RuntimeError("runtime install report is missing")
            self.runtime_install = candidate
        self.session_preflight = _load_session_preflight(
            self.args.session_preflight, self.args.instance)
        self.session_preflight_id = self.session_preflight[
            "session_preflight_id"]
        self.dispatch_count: int | None = None

    def prepare(self) -> None:
        self.run_root.mkdir(parents=True)
        (self.run_root / "runtime-workdir").mkdir()
        _atomic_json(self.run_root / "session_preflight.json",
                     self.session_preflight)

    def verify_runtime(self) -> None:
        install = json.loads(self.runtime_install.read_text(encoding="utf-8"))
        install_revision = install.get("source_revision")
        pairs = (
            (self.root / "qwen3_6_scripts/paged_attn.py",
             self.runtime_site / "vllm/attention/ops/paged_attn.py"),
            (self.root / "qwen3_6_scripts/qwen3_5.py",
             self.runtime_site / "vllm/model_executor/models/qwen3_5.py"),
        )
        if (not isinstance(install_revision, str) or not install_revision
                or not all(left.is_file() and right.is_file()
                           and left.read_bytes() == right.read_bytes()
                           for left, right in pairs)):
            raise RuntimeError("runtime overlay differs from experiment source")
        probe = subprocess.run([
            sys.executable, "-c",
            "import json,sys,torch,vllm,transformers;"
            "import vllm.corex_fused_paged_prefill as ext;"
            "print(json.dumps({'python':sys.version.split()[0],"
            "'torch':torch.__version__,'vllm':vllm.__version__,"
            "'transformers':transformers.__version__,"
            "'candidate_module':ext.__file__}))",
        ], check=False, capture_output=True, text=True,
            cwd=self.run_root / "runtime-workdir",
            env=self.base_environment())
        (self.run_root / "runtime_probe.stdout").write_text(
            probe.stdout, encoding="utf-8")
        (self.run_root / "runtime_probe.stderr").write_text(
            probe.stderr, encoding="utf-8")
        if probe.returncode:
            raise RuntimeError(
                f"runtime import probe failed with rc={probe.returncode}")
        versions = json.loads(probe.stdout.strip().splitlines()[-1])
        compiler = subprocess.run(
            ["/usr/local/corex-3.2.3/bin/clang++", "--version"],
            capture_output=True, text=True)
        compiler_line = (compiler.stdout or compiler.stderr).splitlines()
        compiler_version = compiler_line[0] if compiler_line else "unavailable"
        self.runtime_identity = (
            f"overlay-{install_revision[:12]}-"
            f"torch-{versions['torch']}-vllm-{versions['vllm']}")
        self.runtime_versions = versions
        self.compiler_version = compiler_version
        _atomic_json(self.run_root / "runtime_identity.json", {
            "schema": "bi100-attention-runtime-identity-v1",
            "version": 1,
            "source_revision": self.source_revision,
            "source_dirty_summary": self.source_dirty_summary,
            "runtime_install_source_revision": install_revision,
            "runtime_identity": self.runtime_identity,
            "versions": versions,
            "compiler": compiler_version,
            "relevant_overlay_files_byte_equal": True,
            "tree_hash_used": False,
            "file_hashes_used": False,
        })

    def service_environment(self) -> dict[str, str]:
        environment = self.base_environment()
        environment.update({
            "BI100_RUNTIME_SITE_PACKAGES": str(self.runtime_site),
            "BI100_RUNTIME_INSTALL_REPORT": str(self.runtime_install),
            "BI100_RUNTIME_WORKDIR": str(self.run_root / "runtime-workdir"),
            "MODEL_PATH": str(self.model_path),
            "HOST": "0.0.0.0",
            "PORT": "8000",
            "ENABLE_CUSTOM_IPC": "1",
            "VLLM_ENGINE_ITERATION_TIMEOUT_S": "3600",
            "BI100_MOE_COREX_DIRECT_ROUTED": "1",
            "BI100_GDN_COREX_PACKED_DECODE": "1",
            "BI100_GDN_COMBINED_QK_NORM": "0",
            "BI100_GDN_CACHE_POLICY": "admission64",
            "BI100_GDN_RESTORE_MODE": "hybrid64",
            "BI100_KV_EVICTION_POLICY": "lru",
            "BI100_HYBRID_KV_ACCOUNTING": "full_attention",
            "BI100_CPU_KV_OFFLOAD": "0",
            "BI100_BLOCK_MAJOR_CPU_KV": "0",
            "BI100_CACHE_TRACE": (
                "1" if getattr(self.args, "workload", "performance")
                == "teacher_forced" else "0"),
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
        environment.pop("BI100_ATTN_COREX_FUSED_PREFILL_EXTENSION", None)
        environment.pop("BI100_ATTN_COREX_FUSED_PREFILL_EXTENSION_SHA256", None)
        return environment

    def write_runtime_manifest(self) -> None:
        environment = self.service_environment()
        relevant_names = (
            "ENABLE_CUSTOM_IPC", "VLLM_ENGINE_ITERATION_TIMEOUT_S",
            "BI100_MOE_COREX_DIRECT_ROUTED", "BI100_GDN_COREX_PACKED_DECODE",
            "BI100_GDN_COMBINED_QK_NORM", "BI100_GDN_CACHE_POLICY",
            "BI100_GDN_RESTORE_MODE", "BI100_KV_EVICTION_POLICY",
            "BI100_HYBRID_KV_ACCOUNTING", "BI100_CPU_KV_OFFLOAD",
            "BI100_BLOCK_MAJOR_CPU_KV", "BI100_CACHE_TRACE",
            "BI100_ATTN_COREX_FUSED_PREFILL",
            "BI100_ATTN_COREX_FUSED_PREFILL_SHADOW",
            "BI100_ATTN_CAPTURE_REPLAY", "BI100_GDN_ALLOW_NAN_ZERO",
            "BI100_GDN_FINITE_CHECK",
        )
        command = [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--host", "0.0.0.0", "--port", "8000",
            "--model", str(self.model_path), "--served-model-name", "llm",
            "--max-model-len", "262144", "--gpu-memory-utilization", "0.9",
            "--trust-remote-code", "--tensor-parallel-size", "4",
            "--max-num-seqs", "1", "--disable-log-requests",
            "--disable-frontend-multiprocessing", "--max-num-batched-tokens",
            "8192", "--enable-chunked-prefill", "--max-seq-len-to-capture",
            "32768", "--enable-auto-tool-choice", "--tool-call-parser",
            "qwen3_coder", "--reasoning-parser", "qwen3",
            "--enable-prefix-caching",
        ]
        _atomic_json(self.run_root / "runtime_manifest.json", {
            "schema": "bi100-attention-operator-runtime-v1",
            "version": 1,
            "change_scope": "attention_operator",
            "workload_mode": self.workload_mode,
            "source_revision": self.source_revision,
            "source_dirty_summary": self.source_dirty_summary,
            "runtime_identity": self.runtime_identity,
            "runtime_versions": self.runtime_versions,
            "compiler": self.compiler_version,
            "instance": self.args.instance,
            "model_path": str(self.model_path),
            "tokenizer_path": str(self.model_path),
            "gpu_count": 4,
            "tensor_parallel_size": 4,
            "dtype": "float16",
            "max_model_len": 262144,
            "block_size": 16,
            "served_model_name": "llm",
            "command": command,
            "environment": {name: environment[name] for name in relevant_names},
            "candidate_build_provenance": {
                "module_path": self.runtime_versions["candidate_module"],
                "compiler": self.compiler_version,
                "sha256_used": False,
            },
        })

    def run_requests(self) -> None:
        self.write_runtime_manifest()
        if self.workload_mode == "performance":
            command = [
                sys.executable,
                str(self.root / "tests/attention_operator_tp4_service.py"),
                "--base", "http://127.0.0.1:8000",
                "--model-path", str(self.model_path),
                "--timeout-s", "1800", "--run-id", self.run_id,
                "--workload-id", self.args.pair_id,
                "--selector", self.args.selector,
                "--targets", ",".join(map(str, self.targets)),
                "--repetitions", str(self.repetitions),
                "--out", str(self.run_root / "measurement.json"),
            ]
            environment = self.base_environment()
        else:
            key = os.environ.get("BI100_TEACHER_FORCED_HMAC_KEY", "")
            if (len(key) != 64 or any(character not in "0123456789abcdef"
                                      for character in key)):
                raise RuntimeError("teacher token identity key is unavailable")
            command = [
                sys.executable,
                str(self.root / "tests/teacher_forced_topk_api.py"),
                "--base", "http://127.0.0.1:8000",
                "--model-path", str(self.model_path),
                "--attention-runtime-manifest",
                str(self.run_root / "runtime_manifest.json"),
                "--runtime-identity", self.runtime_identity,
                "--source-revision", self.source_revision,
                "--instance", self.args.instance,
                "--mode", self.args.selector,
                "--targets", ",".join(map(str, self.targets)),
                "--timeout-s", "3600",
                "--out", str(self.run_root / "measurement.json"),
            ]
            environment = self.base_environment()
            environment["BI100_TEACHER_FORCED_HMAC_KEY"] = key
        rc = _run_to_files(
            command, self.run_root / "measurement.stdout",
            self.run_root / "measurement.stderr",
            environment=environment,
            cwd=self.run_root / "runtime-workdir", timeout_s=14400)
        if rc:
            raise RuntimeError("focused attention request population failed")

    def verify_dispatch(self) -> None:
        log = (self.run_root / "server.log").read_text(
            encoding="utf-8", errors="replace")
        count = log.count("path=corex_split4")
        if self.args.selector == "candidate" and count <= 0:
            raise RuntimeError("candidate dispatch marker is absent")
        if self.args.selector == "control" and count != 0:
            raise RuntimeError("control unexpectedly dispatched candidate")
        self.dispatch_count = count
        (self.run_root / "dispatch_count.txt").write_text(
            f"{count}\n", encoding="ascii")

    def source_revision_stable(self) -> None:
        if _git(self.root, "rev-parse", "HEAD") != self.source_revision:
            raise RuntimeError("source revision changed during service arm")
        self.source_dirty_summary_after = _git(
            self.root, "status", "--short", "--untracked-files=all") or "clean"

    def run_postconditions(
        self, primary_error: BaseException | None,
    ) -> BaseException | None:
        def check(name: str, action: Callable[[], None]) -> None:
            nonlocal primary_error
            try:
                with self.stage(name):
                    action()
            except BaseException as exc:
                primary_error = self.retain_error(primary_error, exc)

        check("scoped_cleanup", self.cleanup_service)
        check("postflight_after", lambda: self.run_postflight("postflight_after"))

        def fatal() -> None:
            if any(self.scan_fatal().values()):
                raise RuntimeError("fatal/OOM/collective/worker-loss scan failed")

        check("fatal_scan", fatal)
        check("source_revision_stable", self.source_revision_stable)
        return primary_error

    def write_status(self, returncode: int) -> None:
        timeline = summarize(self.timeline, expected_run_id=self.run_id)
        _atomic_json(self.run_root / "timeline_report.json", timeline)
        hard_candidate_stage = self.failed_stage in {
            "request_population", "dispatch", "health_after_requests",
            "fatal_scan"}
        result_status = (
            "pass" if returncode == 0 else
            "fail" if self.args.selector == "candidate" and hard_candidate_stage
            else "invalid")
        measurement = self.run_root / "measurement.json"
        measured = (json.loads(measurement.read_text(encoding="utf-8"))
                    if measurement.is_file() else {})
        if self.workload_mode == "teacher_forced":
            completed = len(measured.get("cases") or [])
            attempted = completed if measurement.is_file() else 0
            failed = 0 if completed == len(self.targets) else max(
                0, len(self.targets) - completed)
        else:
            attempted = measured.get("attempted_requests", 0)
            completed = measured.get("completed_requests", 0)
            failed = measured.get("failed_requests", 0)
        _atomic_json(self.run_root / "runner_status.json", {
            "schema": "bi100-attention-operator-tp4-arm-v1",
            "version": 1,
            "change_scope": "attention_operator",
            "workload_mode": self.workload_mode,
            "qualified": returncode == 0,
            "result_status": result_status,
            "returncode": returncode,
            "terminal_stage": self.current_stage,
            "failed_stage": self.failed_stage,
            "error_type": self.error_type,
            "run_id": self.run_id,
            "source_revision": self.source_revision,
            "source_branch": self.source_branch,
            "source_dirty_summary": self.source_dirty_summary,
            "source_dirty_summary_after": getattr(
                self, "source_dirty_summary_after", None),
            "runtime_identity": self.runtime_identity,
            "instance": self.args.instance,
            "model_path": str(self.model_path),
            "selector": self.args.selector,
            "workload_id": self.args.pair_id,
            "session_preflight_id": self.session_preflight_id,
            "targets": list(self.targets),
            "repetitions": self.repetitions,
            "service_startups": 1,
            "gpu_count": 4,
            "tensor_parallel_size": 4,
            "request_population": {
                "expected": len(self.targets) * self.repetitions,
                "attempted": attempted,
                "completed": completed,
                "failed": failed,
            },
            "dispatch_count": self.dispatch_count,
            "gates": self.gates,
            "timing": {"wall_span_s": timeline["wall_span_s"],
                       "summed_stage_s": timeline["summed_stage_s"]},
            "artifacts_present": {
                name: (self.run_root / name).is_file() for name in (
                    "session_preflight.json", "runtime_identity.json",
                    "runtime_manifest.json", "measurement.json",
                    "dispatch_count.txt", "scoped_cleanup.json",
                    "postflight_after.json", "fatal_scan.json",
                    "timeline_report.json")},
            "hashes_required": False,
            "capability_run": False,
            "cache_matrix_run": False,
        })

    def run(self) -> int:
        self.validate()
        self.prepare()
        primary_error: BaseException | None = None
        try:
            with self.stage("session_preflight"):
                _load_session_preflight(
                    self.args.session_preflight, self.args.instance)
            with self.stage("runtime_identity"):
                self.verify_runtime()
            with self.stage("service_startup"):
                self.start_service()
            with self.stage("request_population"):
                self.run_requests()
            with self.stage("dispatch"):
                self.verify_dispatch()
            with self.stage("health_after_requests"):
                if not _health():
                    raise RuntimeError("service health failed after requests")
        except BaseException as exc:
            primary_error = self.retain_error(primary_error, exc)
        primary_error = self.run_postconditions(primary_error)
        returncode = 0 if primary_error is None else 1
        self.current_stage = "complete" if returncode == 0 else (
            self.failed_stage or self.current_stage)
        self.write_status(returncode)
        return returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("instance")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--selector", choices=("control", "candidate"),
                        required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--session-preflight", type=Path, required=True)
    parser.add_argument("--targets", default=",".join(map(str, TARGETS)))
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument("--workload", choices=("performance", "teacher_forced"),
                        default="performance")
    args = parser.parse_args()
    args.profile = "attention_operator"
    args.contexts = ""
    args.ordinals = ""
    runner = AttentionOperatorTp4Runner(args)

    def interrupt(signum: int, _frame: Any) -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
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
        print(f"attention TP4 runner failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
