#!/usr/bin/env python3
"""Run one fixed M1-176 short-TP4 v2 arm with a single service startup."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any

from record_experiment_timeline import summarize
from run_m1_140_activation_capture import (
    CaptureRunner, _atomic_json, _health, _port_free, _run_to_files,
)


TARGETS = (4096, 16384, 32768, 65536)
PROJECTED_GAIN_FLOOR = 0.02


def _load_private_json(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if not path.is_relative_to(Path("/tmp")) or path.stat().st_mode & 0o077:
        raise ValueError("L2 authorization must be private under /tmp")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("L2 authorization is not an object")
    return value


def validate_l2_authorization(
    replay_path: Path,
    capture_path: Path,
) -> dict[str, Any]:
    replay = _load_private_json(replay_path)
    capture = _load_private_json(capture_path)
    aggregate = replay.get("aggregate") or {}
    authorization = replay.get("authorization") or {}
    rows = aggregate.get("rows")
    if (
        replay.get("schema")
        != "bi100-m1-176-four-rank-real-activation-replay-v2"
        or replay.get("version") != 2
        or replay.get("qualified") is not True
        or replay.get("result_status") != "pass"
        or replay.get("terminal_stage") != "parallel_four_rank_replay"
        or authorization.get("l3_short_tp4_authorized") is not True
        or authorization.get("long_context_or_formal_score_authorized")
        is not False
        or not isinstance(rows, list) or len(rows) != 4
        or [row.get("logical_tp_rank") for row in rows] != [0, 1, 2, 3]
        or any(row.get("all_g2_qualified") is not True
               or row.get("record_count") != 3 for row in rows)
        or aggregate.get("four_rank_replay_complete") is not True
        or aggregate.get("g2_reasons") != []
        or aggregate.get("invalid_reasons") != []
    ):
        raise ValueError("L2 replay does not authorize short TP4")
    if (
        capture.get("schema") != "qwen36-diagnostic-service-gate-v2"
        or capture.get("version") != 2
        or capture.get("qualified") is not True
        or capture.get("workload_scope")
        != "m1-176-activation-capture-only"
        or (capture.get("activation_capture_summary") or {}).get(
            "request_count") != 3
    ):
        raise ValueError("L2 capture population is invalid")
    return {
        "capture_qualified": True,
        "replay_qualified": True,
        "source_revision": replay.get("source_revision"),
        "runtime_identity": replay.get("runtime_identity"),
        "four_rank_cells": 12,
        "raw_activation_reused": True,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ShortTp4V2Runner(CaptureRunner):

    def validate(self) -> None:
        if (self.run_root == Path("/tmp")
                or not self.run_root.is_relative_to(Path("/tmp"))
                or self.run_root.exists()):
            raise ValueError("run root must be a new private path under /tmp")
        status = subprocess.run([
            "git", "-C", str(self.root), "status", "--porcelain",
            "--untracked-files=all", "--", ".", ":(exclude)bench_runs/**",
        ], check=True, capture_output=True, text=True).stdout.strip()
        if status:
            raise RuntimeError("short TP4 runner requires a clean source tree")
        if (not 1 <= len(self.args.pair_id) <= 128
                or any(not (item.isalnum() or item in "._-")
                       for item in self.args.pair_id)):
            raise ValueError("pair identity is invalid")
        if self.args.projected_gain < PROJECTED_GAIN_FLOOR:
            raise ValueError("Amdahl projected gain is below the L3 entry floor")
        self.source_revision = subprocess.run([
            "git", "-C", str(self.root), "rev-parse", "HEAD",
        ], check=True, capture_output=True, text=True).stdout.strip()
        self.source_branch = subprocess.run([
            "git", "-C", str(self.root), "branch", "--show-current",
        ], check=True, capture_output=True, text=True).stdout.strip()
        self.run_id = (
            f"m1-176-l3-{self.args.selector}-{self.source_revision[:10]}")
        if not _port_free():
            raise RuntimeError("API port 8000 is occupied")
        if (not self.runtime_site.is_absolute()
                or not (self.runtime_site / "vllm").is_dir()
                or not (self.runtime_site / "transformers").is_dir()):
            raise RuntimeError("runtime overlay is missing")
        if not self.runtime_install.is_file():
            candidate = self.runtime_site.parent / "install.json"
            if not candidate.is_file():
                raise RuntimeError("runtime install report is missing")
            self.runtime_install = candidate
        if not self.model_path.is_dir():
            raise RuntimeError("model path is missing")
        self.l2_authorization = validate_l2_authorization(
            self.args.l2_replay_status, self.args.l2_capture_status)
        l2_revision = self.l2_authorization["source_revision"]
        ancestry = subprocess.run([
            "git", "-C", str(self.root), "merge-base", "--is-ancestor",
            l2_revision, self.source_revision,
        ]).returncode
        if ancestry:
            raise ValueError("L2 source is not an ancestor of the L3 harness")
        extension = self.args.candidate_extension.resolve(strict=True)
        if (not extension.is_file() or not extension.is_relative_to(Path("/tmp"))
                or extension.stat().st_size <= 0
                or extension.stat().st_mode & 0o022):
            raise ValueError("candidate extension path or permissions are invalid")
        observed = _file_sha256(extension)
        if observed != self.args.expected_candidate_sha256:
            raise ValueError("candidate extension identity differs")
        self.candidate_extension = extension
        self.candidate_extension_sha256 = observed
        key_path = self.args.teacher_hmac_key_file.resolve(strict=True)
        if (not key_path.is_file() or not key_path.is_relative_to(Path("/tmp"))
                or key_path.stat().st_mode & 0o077):
            raise ValueError("teacher identity key must be private under /tmp")
        key = key_path.read_text(encoding="ascii").strip()
        if (len(key) != 64
                or any(character not in "0123456789abcdef" for character in key)):
            raise ValueError("teacher identity key is invalid")
        self.teacher_hmac_key = key
        self.dispatch_count: int | None = None

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
            path = self.run_root / name
            path.write_text(value + "\n", encoding="ascii")
            path.chmod(0o600)

    def verify_runtime(self) -> None:
        install = json.loads(self.runtime_install.read_text(encoding="utf-8"))
        install_revision = install.get("source_revision")
        pairs = (
            (self.root / "qwen3_6_scripts/paged_attn.py",
             self.runtime_site / "vllm/attention/ops/paged_attn.py"),
            (self.root / "qwen3_6_scripts/qwen3_5.py",
             self.runtime_site / "vllm/model_executor/models/qwen3_5.py"),
        )
        equal = all(left.is_file() and right.is_file()
                    and left.read_bytes() == right.read_bytes()
                    for left, right in pairs)
        if (not isinstance(install_revision, str) or not install_revision
                or not equal):
            raise RuntimeError("runtime overlay does not match required sources")
        self.runtime_identity = (
            f"overlay-install-{install_revision[:12]}-paged-qwen-byte-equal")
        probe = subprocess.run([
            sys.executable, "-c",
            "import json,torch,vllm,transformers;print(json.dumps({"
            "'python':__import__('sys').version.split()[0],"
            "'torch':torch.__version__,'vllm':vllm.__version__,"
            "'transformers':transformers.__version__}))",
        ], check=True, capture_output=True, text=True,
            env=self.base_environment())
        versions = json.loads(probe.stdout.strip().splitlines()[-1])
        _atomic_json(self.run_root / "runtime_identity.json", {
            "schema": "bi100-lightweight-runtime-identity-v2",
            "version": 2,
            "qualified": True,
            "source_revision": self.source_revision,
            "runtime_install_source_revision": install_revision,
            "required_files_byte_equal": equal,
            "full_tree_hash_used": False,
            "runtime_identity": self.runtime_identity,
            "versions": versions,
        })

    def service_environment(self) -> dict[str, str]:
        environment = self.base_environment()
        environment.update({
            "BI100_RUNTIME_SITE_PACKAGES": str(self.runtime_site),
            "BI100_RUNTIME_INSTALL_REPORT": str(self.runtime_install),
            "BI100_RUNTIME_WORKDIR": str(self.run_root / "runtime-workdir"),
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
            "BI100_CACHE_TRACE": "1",
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
                "BI100_ATTN_COREX_FUSED_PREFILL_EXTENSION_SHA256": (
                    self.candidate_extension_sha256),
            })
        else:
            environment.pop("BI100_ATTN_COREX_FUSED_PREFILL_EXTENSION", None)
            environment.pop(
                "BI100_ATTN_COREX_FUSED_PREFILL_EXTENSION_SHA256", None)
        return environment

    def _write_runtime_manifest(self) -> Path:
        semantic_environment = {
            name: self.service_environment()[name]
            for name in (
                "ENABLE_CUSTOM_IPC", "VLLM_ENGINE_ITERATION_TIMEOUT_S",
                "BI100_MOE_COREX_DIRECT_ROUTED", "BI100_GDN_COREX_PACKED_DECODE",
                "BI100_GDN_COMBINED_QK_NORM", "BI100_GDN_CACHE_POLICY",
                "BI100_GDN_RESTORE_MODE", "BI100_HYBRID_KV_ACCOUNTING",
                "BI100_CPU_KV_OFFLOAD", "BI100_BLOCK_MAJOR_CPU_KV",
                "BI100_CACHE_TRACE", "BI100_ATTN_COREX_FUSED_PREFILL",
                "BI100_ATTN_COREX_FUSED_PREFILL_SHADOW",
                "BI100_ATTN_CAPTURE_REPLAY", "BI100_GDN_ALLOW_NAN_ZERO",
                "BI100_GDN_FINITE_CHECK",
            )
        }
        manifest = {
            "schema": "bi100-quality-runtime-manifest-v2",
            "version": 2,
            "source_revision": self.source_revision,
            "runtime_identity": self.runtime_identity,
            "instance": self.args.instance,
            "model_path": str(self.model_path.resolve()),
            "tokenizer_path": str(self.model_path.resolve()),
            "gpu_count": 4,
            "tensor_parallel_size": 4,
            "max_model_len": 262144,
            "served_model_name": "llm",
            "command": [
                "python3", "-m", "vllm.entrypoints.openai.api_server",
                "--host", "0.0.0.0", "--port", "8000",
                "--model", str(self.model_path.resolve()),
                "--served-model-name", "llm",
                "--max-model-len", "262144",
                "--gpu-memory-utilization", "0.9", "--trust-remote-code",
                "--tensor-parallel-size", "4", "--max-num-seqs", "1",
                "--disable-log-requests", "--disable-frontend-multiprocessing",
                "--max-num-batched-tokens", "8192", "--enable-chunked-prefill",
                "--max-seq-len-to-capture", "32768", "--enable-auto-tool-choice",
                "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3",
                "--enable-prefix-caching",
            ],
            "environment": semantic_environment,
        }
        path = self.run_root / "runtime_manifest_v2.json"
        _atomic_json(path, manifest)
        return path

    def run_requests(self) -> None:
        runtime_manifest = self._write_runtime_manifest()
        rc = _run_to_files([
            sys.executable, str(self.root / "tests/short_tp4_v2_service.py"),
            "--base", "http://127.0.0.1:8000",
            "--model-path", str(self.model_path.resolve()),
            "--timeout-s", "1800", "--run-id", self.run_id,
            "--prompt-set-id", self.args.pair_id,
            "--selector", self.args.selector,
            "--out", str(self.run_root / "measurement_private.json"),
        ], self.run_root / "measurement.stdout",
            self.run_root / "measurement.stderr",
            environment=self.base_environment(), timeout_s=14400)
        if rc:
            raise RuntimeError("short TP4 request matrix failed")
        teacher_environment = self.base_environment()
        teacher_environment["BI100_TEACHER_FORCED_HMAC_KEY"] = (
            self.teacher_hmac_key)
        rc = _run_to_files([
            sys.executable, str(self.root / "tests/teacher_forced_topk_api.py"),
            "--base", "http://127.0.0.1:8000",
            "--model-path", str(self.model_path.resolve()),
            "--runtime-manifest-v2", str(runtime_manifest),
            "--runtime-identity", self.runtime_identity,
            "--source-revision", self.source_revision,
            "--instance", self.args.instance,
            "--mode", ("candidate" if self.args.selector == "candidate"
                       else "control"),
            "--targets", ",".join(map(str, TARGETS)),
            "--timeout-s", "3600",
            "--out", str(self.run_root / "teacher_forced_private.json"),
        ], self.run_root / "teacher_forced.stdout",
            self.run_root / "teacher_forced.stderr",
            environment=teacher_environment, timeout_s=14400)
        teacher_environment.pop("BI100_TEACHER_FORCED_HMAC_KEY", None)
        if rc:
            raise RuntimeError("teacher-forced population failed")

    def verify_dispatch(self) -> None:
        log = (self.run_root / "server.log").read_text(
            encoding="utf-8", errors="replace")
        count = log.count("path=corex_split4")
        if self.args.selector == "candidate" and count < 2:
            raise RuntimeError("candidate dispatch marker is absent")
        if self.args.selector != "candidate" and count:
            raise RuntimeError("control unexpectedly used candidate dispatch")
        self.dispatch_count = count
        (self.run_root / "dispatch_count.txt").write_text(
            f"{count}\n", encoding="ascii")

    def write_status(self, returncode: int) -> None:
        timeline = summarize(self.timeline, expected_run_id=self.run_id)
        _atomic_json(self.run_root / "timeline_report.json", timeline)
        artifacts = {
            name: (self.run_root / name).is_file()
            for name in (
                "runtime_identity.json", "runtime_manifest_v2.json",
                "measurement_private.json", "teacher_forced_private.json",
                "dispatch_count.txt", "fatal_scan.json",
                "postflight_after.json", "preflight_comparison.json",
                "timeline_report.json",
            )
        }
        _atomic_json(self.run_root / "runner_status.json", {
            "schema": "bi100-m1-176-short-tp4-arm-runner-v2",
            "version": 2,
            "qualified": returncode == 0,
            "result_status": "pass" if returncode == 0 else "invalid",
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
            "model_path": str(self.model_path.resolve()),
            "gpu_count": 4,
            "tensor_parallel_size": 4,
            "service_startups": 1,
            "targets": list(TARGETS),
            "cache_states": ["cold", "partial-prefix", "full-warm"],
            "repetitions": 3,
            "request_population": {
                "service_expected": 72,
                "teacher_forced_expected": 4,
                "total_expected": 76,
                "total_completed": 76 if returncode == 0 else None,
            },
            "projected_gain": self.args.projected_gain,
            "gates": self.gates,
            "artifacts_present": artifacts,
            "candidate_artifact": {
                "sha256": self.candidate_extension_sha256,
                "size_bytes": self.candidate_extension.stat().st_size,
                "active": self.args.selector == "candidate",
            },
            "dispatch_count": self.dispatch_count,
            "l2_authorization": self.l2_authorization,
            "timing": {
                "wall_span_s": timeline["wall_span_s"],
                "summed_stage_s": timeline["summed_stage_s"],
            },
            "privacy": {
                "prompts_recorded": False,
                "model_outputs_recorded": False,
                "token_ids_recorded": False,
                "credentials_recorded": False,
                "private_observations_must_remain_tmp": True,
            },
            "authorization": {
                "long_context_authorized": False,
                "full_capability_authorized": False,
                "formal_score_authorized": False,
                "main_or_yaml_change_authorized": False,
            },
        })

    def run(self) -> int:
        self.validate()
        self.prepare()
        primary_error: BaseException | None = None
        try:
            with self.stage("postflight_before"):
                self.run_postflight("postflight_before")
            with self.stage("preflight_before"):
                self.run_preflight("preflight_before")
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
        self.current_stage = "complete" if not returncode else (
            self.failed_stage or self.current_stage)
        self.write_status(returncode)
        return returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("instance")
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--selector", choices=("control_a", "control_b", "candidate"),
        required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--candidate-extension", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--l2-replay-status", type=Path, required=True)
    parser.add_argument("--l2-capture-status", type=Path, required=True)
    parser.add_argument("--projected-gain", type=float, required=True)
    parser.add_argument("--teacher-hmac-key-file", type=Path, required=True)
    args = parser.parse_args()
    args.profile = "short-v2"
    args.targets = ",".join(map(str, TARGETS))
    args.contexts = ""
    args.ordinals = ""
    runner = ShortTp4V2Runner(args)

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
        print(f"short TP4 v2 runner failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
