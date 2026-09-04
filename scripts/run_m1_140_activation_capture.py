#!/usr/bin/env python3
"""Capture a reusable private real-activation bank with one TP4 startup."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Iterator
import urllib.request

from record_experiment_timeline import append_event, summarize


RUNNER_SCHEMA = "bi100-m1-140-activation-capture-runner-v1"
TERM_GRACE_S = 60.0
KILL_GRACE_S = 20.0
SYSTEM_PYTHONPATH = (
    "/usr/local/corex/lib64/python3/dist-packages:"
    "/usr/local/corex/lib/python3/dist-packages"
)
COREX_LD_LIBRARY_PATH = (
    "/usr/local/corex/lib:/usr/local/corex/lib64:"
    "/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:"
    "/usr/local/openmpi/lib"
)
COREX_PATH = (
    "/usr/local/corex/bin:/usr/local/corex-3.2.3/bin:"
    "/usr/local/openmpi/bin:/usr/local/sbin:/usr/local/bin:"
    "/usr/sbin:/usr/bin:/sbin:/bin"
)
FATAL_PATTERNS = {
    "cuda_error": re.compile(
        r"CUDA error|illegal memory access|device-side assert", re.I),
    "segfault": re.compile(r"SIGSEGV|Fatal Python error", re.I),
    "oom": re.compile(r"out of memory", re.I),
    "distributed": re.compile(
        r"Gloo.*(?:failed|reset|error)|NCCL.*(?:failed|abort|error)|"
        r"Connection reset by peer",
        re.I,
    ),
    "worker_loss": re.compile(
        r"worker.*(?:died|lost|exited unexpectedly)", re.I),
    "timeout": re.compile(
        r"Timeout(?:Error|Expired)|engine iteration timed out|"
        r"watchdog.*tim(?:e|ed) out",
        re.I,
    ),
    "state_error": re.compile(
        r"scheduler requested a missing GDN prefix state|"
        r"non-finite GatedDeltaNet",
        re.I,
    ),
}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="ascii") as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_to_files(
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
    *,
    environment: dict[str, str] | None = None,
    timeout_s: int,
) -> int:
    wrapped = [
        "timeout",
        "--foreground",
        "--signal=TERM",
        "--kill-after=90s",
        f"{timeout_s}s",
        *command,
    ]
    with (
        stdout_path.open("wb") as stdout,
        stderr_path.open("wb") as stderr,
    ):
        blocked = {signal.SIGTERM, signal.SIGINT}
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
        process: subprocess.Popen[bytes] | None = None
        starttime: int | None = None
        try:
            process = subprocess.Popen(
                wrapped,
                stdout=stdout,
                stderr=stderr,
                env=environment,
                start_new_session=True,
            )
            starttime = _read_starttime(process.pid)
        except BaseException:
            if process is not None:
                _stop_process_group(process, starttime)
            raise
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        try:
            return process.wait()
        except BaseException:
            _stop_process_group(process, starttime)
            raise
        finally:
            if process.poll() is not None:
                process.wait()


def _health() -> bool:
    try:
        with urllib.request.urlopen(
                "http://127.0.0.1:8000/health", timeout=5) as response:
            response.read()
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def _port_free() -> bool:
    import socket

    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 8000))
        return True
    except OSError:
        return False


def _read_starttime(pid: int) -> int:
    value = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    fields = value[value.rfind(")") + 2:].split()
    return int(fields[19])


def _stop_process_group(
    process: subprocess.Popen[Any],
    expected_starttime: int | None,
) -> bool:
    if process.poll() is not None:
        process.wait()
        return True
    if expected_starttime is not None:
        try:
            if _read_starttime(process.pid) != expected_starttime:
                return False
        except (FileNotFoundError, ProcessLookupError):
            process.poll()
            return True
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.poll()
        return True
    try:
        process.wait(timeout=TERM_GRACE_S)
        return True
    except subprocess.TimeoutExpired:
        pass
    if expected_starttime is not None:
        try:
            if _read_starttime(process.pid) != expected_starttime:
                return False
        except (FileNotFoundError, ProcessLookupError):
            process.poll()
            return True
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        process.poll()
        return True
    try:
        process.wait(timeout=KILL_GRACE_S)
    except subprocess.TimeoutExpired:
        return False
    return True


class CaptureRunner:

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = Path(__file__).resolve().parents[1]
        self.run_root = args.run_root.resolve()
        self.timeline = self.run_root / "timeline.jsonl"
        self.process: subprocess.Popen[bytes] | None = None
        self.process_starttime: int | None = None
        self.server_log = None
        self.current_stage = "initialization"
        self.failed_stage: str | None = None
        self.gates: dict[str, int | None] = {}
        self.error_type: str | None = None
        self.source_revision = ""
        self.source_branch = ""
        self.runtime_identity = ""
        self.runtime_site = Path(
            os.environ.get("BI100_RUNTIME_SITE_PACKAGES", ""))
        self.runtime_install = Path(
            os.environ.get("BI100_RUNTIME_INSTALL_REPORT", ""))
        self.model_path = Path(os.environ.get(
            "MODEL_PATH",
            "/root/public-storage/models/Qwen/Qwen3.6-35B-A3B",
        ))
        self.run_id = ""

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        self.current_stage = name
        append_event(
            self.timeline, run_id=self.run_id, stage=name, event="start")
        try:
            yield
        except BaseException:
            self.gates[name] = 1
            if self.failed_stage is None:
                self.failed_stage = name
            append_event(
                self.timeline,
                run_id=self.run_id,
                stage=name,
                event="end",
                status="fail",
            )
            raise
        else:
            self.gates[name] = 0
            append_event(
                self.timeline,
                run_id=self.run_id,
                stage=name,
                event="end",
                status="pass",
            )

    def validate(self) -> None:
        if (
            self.run_root == Path("/tmp")
            or not self.run_root.is_relative_to(Path("/tmp"))
            or self.run_root.exists()
        ):
            raise ValueError("run root must be a new private path under /tmp")
        source_status = _git(
            self.root,
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            ".",
            ":(exclude)bench_runs/**",
        )
        if source_status:
            raise RuntimeError("capture runner requires a clean source tree")
        self.source_revision = _git(self.root, "rev-parse", "HEAD")
        self.source_branch = _git(self.root, "branch", "--show-current")
        self.run_id = f"m1-140-{self.source_revision[:12]}"
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
        if not _port_free():
            raise RuntimeError("API port 8000 is already occupied")
        if self.args.profile == "qualification":
            if (
                self.args.targets != "32768,65536,131072"
                or self.args.contexts != "24576,57344,122880"
                or self.args.ordinals != "0,4,9"
            ):
                raise ValueError(
                    "qualification profile requires the frozen matrix")

    def prepare(self) -> None:
        self.run_root.mkdir(mode=0o700, parents=True)
        (self.run_root / "runtime-workdir").mkdir(mode=0o700)
        (self.run_root / "activation-bank").mkdir(mode=0o700)
        for name, value in {
            "source_revision.txt": self.source_revision,
            "source_branch.txt": self.source_branch,
            "instance.txt": self.args.instance,
            "model_path.txt": str(self.model_path.resolve()),
            "runtime_site_packages.txt": str(self.runtime_site.resolve()),
        }.items():
            (self.run_root / name).write_text(
                value + "\n", encoding="ascii")

    def base_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update({
            "PYTHONPATH": (
                f"{self.root / 'tests'}:{self.runtime_site}:"
                f"{SYSTEM_PYTHONPATH}"),
            "LD_LIBRARY_PATH": COREX_LD_LIBRARY_PATH,
            "PATH": COREX_PATH,
            "PYTHONFAULTHANDLER": "1",
            "PYTHONUNBUFFERED": "1",
        })
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        return environment

    def run_postflight(self, label: str) -> None:
        rc = _run_to_files(
            [
                sys.executable,
                str(self.root / "tests" / "service_postflight_gate.py"),
                "--gpus", "0,1,2,3",
                "--settle-timeout-s", "90",
                "--clean-samples", "3",
                "--sample-interval-s", "2",
                "--out", str(self.run_root / f"{label}.json"),
            ],
            self.run_root / f"{label}.stdout",
            self.run_root / f"{label}.stderr",
            environment=self.base_environment(),
            timeout_s=300,
        )
        if rc:
            raise RuntimeError(f"{label} failed")

    def run_preflight(self, label: str) -> None:
        rc = _run_to_files(
            [
                sys.executable,
                str(self.root / "tests" / "bi100_parallel_preflight.py"),
                "--gpus", "0,1,2,3",
                "--timeout-s", "25",
                "--matmul-size", "1024",
                "--json-out", str(self.run_root / f"{label}.json"),
                "--work-dir",
                str(self.run_root / f"{label}-parallel"),
            ],
            self.run_root / f"{label}.stdout",
            self.run_root / f"{label}.stderr",
            environment=self.base_environment(),
            timeout_s=480,
        )
        if rc:
            raise RuntimeError(f"{label} failed")

    def verify_runtime(self) -> None:
        output = self.run_root / "runtime_identity.json"
        rc = _run_to_files(
            [
                sys.executable,
                str(self.root / "tests"
                    / "verify_bare_host_runtime_identity.py"),
                "--source-root", str(self.root),
                "--runtime-site-packages", str(self.runtime_site),
                "--runtime-install", str(self.runtime_install),
                "--out", str(output),
            ],
            self.run_root / "runtime_identity.stdout",
            self.run_root / "runtime_identity.stderr",
            environment=self.base_environment(),
            timeout_s=600,
        )
        if rc:
            raise RuntimeError("runtime identity verification failed")
        value = json.loads(output.read_text(encoding="ascii"))
        tree_sha = value.get("runtime_tree_sha256")
        if (
            value.get("qualified") is not True
            or not isinstance(tree_sha, str)
            or len(tree_sha) != 64
        ):
            raise RuntimeError("runtime identity is not qualified")
        self.runtime_identity = f"bare-host-overlay-v1:{tree_sha[:20]}"

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
            "BI100_ATTN_COREX_FUSED_PREFILL": "0",
            "BI100_ATTN_COREX_FUSED_PREFILL_SHADOW": "0",
            "BI100_ATTN_CAPTURE_REPLAY": "1",
            "BI100_ATTN_CAPTURE_REPLAY_DIR": str(
                self.run_root / "activation-bank"),
            "BI100_ATTN_CAPTURE_REPLAY_RUN_ID": self.run_id,
            "BI100_ATTN_CAPTURE_REPLAY_CONTEXTS": self.args.contexts,
            "BI100_ATTN_CAPTURE_REPLAY_CALL_ORDINALS": self.args.ordinals,
            "BI100_ATTN_CAPTURE_REPLAY_SOURCE_REVISION": (
                self.source_revision),
            "BI100_ATTN_CAPTURE_REPLAY_RUNTIME_IDENTITY": (
                self.runtime_identity),
            "BI100_ATTN_CAPTURE_REPLAY_SYNTHETIC_ATTESTATION": (
                "synthetic-exact-prompt-v1"),
            "BI100_PROFILE": "0",
            "BI100_PROFILE_INCLUDE_STARTUP": "0",
            "BI100_PAGED_ATTN_DIAGNOSTICS": "0",
            "BI100_GDN_ALLOW_NAN_ZERO": "0",
            "BI100_GDN_FINITE_CHECK": "0",
        })
        return environment

    def start_service(self) -> None:
        identity = self.run_root / "service_identity.json"
        self.server_log = (self.run_root / "server.log").open("wb")
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(self.root / "scripts" / "exec_bi100_session.py"),
                str(identity),
                "--",
                str(self.root / "launch_service"),
            ],
            stdout=self.server_log,
            stderr=subprocess.STDOUT,
            cwd=self.run_root / "runtime-workdir",
            env=self.service_environment(),
        )
        self.process_starttime = _read_starttime(self.process.pid)
        deadline = time.monotonic() + 3600
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("service exited before health readiness")
            if identity.is_file() and _health():
                return
            time.sleep(10)
        raise TimeoutError("service startup timed out")

    def run_capture_requests(self) -> None:
        rc = _run_to_files(
            [
                sys.executable,
                str(self.root / "tests"
                    / "capture_fused_prefill_activation_service.py"),
                "--base", "http://127.0.0.1:8000",
                "--model-path", str(self.model_path.resolve()),
                "--targets", self.args.targets,
                "--max-tokens", "2",
                "--timeout-s", "1800",
                "--run-id", self.run_id,
                "--out", str(self.run_root / "requests.json"),
            ],
            self.run_root / "requests.stdout",
            self.run_root / "requests.stderr",
            environment=self.base_environment(),
            timeout_s=7200,
        )
        if rc:
            raise RuntimeError("activation capture requests failed")

    def qualify_bank(self) -> None:
        manifests = [
            self.run_root / "activation-bank" / f"rank-{rank}.manifest.json"
            for rank in range(4)
        ]
        command = [
            sys.executable,
            str(self.root / "tests"
                / "qualify_fused_prefill_activation_bank.py"),
        ]
        for manifest in manifests:
            command.extend(["--manifest", str(manifest)])
        command.extend([
            "--contract",
            str(self.root / "quality" / "experiment_funnel.v1.json"),
            "--profile", self.args.profile,
            "--run-id", self.run_id,
            "--source-revision", self.source_revision,
            "--runtime-identity", self.runtime_identity,
            "--out", str(self.run_root / "bank_qualification.json"),
        ])
        rc = _run_to_files(
            command,
            self.run_root / "bank_qualification.stdout",
            self.run_root / "bank_qualification.stderr",
            environment=self.base_environment(),
            timeout_s=900,
        )
        if rc:
            raise RuntimeError("activation bank qualification failed")

    def cleanup_service(self) -> None:
        identity = self.run_root / "service_identity.json"
        if identity.is_file():
            rc = _run_to_files(
                [
                    sys.executable,
                    str(self.root / "scripts"
                        / "cleanup_recorded_bi100_sessions.py"),
                    "--identity", str(identity),
                    "--out", str(self.run_root / "scoped_cleanup.json"),
                ],
                self.run_root / "scoped_cleanup.stdout",
                self.run_root / "scoped_cleanup.stderr",
                environment=self.base_environment(),
                timeout_s=600,
            )
            if rc:
                raise RuntimeError("scoped service cleanup failed")
        elif self.process is not None and self.process.poll() is None:
            if _read_starttime(self.process.pid) != self.process_starttime:
                raise RuntimeError("unattested service leader identity changed")
            self.process.terminate()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                if _read_starttime(self.process.pid) != self.process_starttime:
                    raise RuntimeError(
                        "unattested service leader identity changed")
                self.process.kill()
                self.process.wait(timeout=20)
        if self.process is not None:
            try:
                self.process.wait(timeout=90)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    "service parent did not reap descendants") from exc
        if self.server_log is not None:
            self.server_log.close()
            self.server_log = None

    def scan_fatal(self) -> dict[str, int]:
        counts = {name: 0 for name in FATAL_PATTERNS}
        log = self.run_root / "server.log"
        if not log.is_file():
            return {"missing_log": 1, **counts}
        with log.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                for name, pattern in FATAL_PATTERNS.items():
                    if pattern.search(line):
                        counts[name] += 1
        _atomic_json(
            self.run_root / "fatal_scan.json",
            {
                "schema": "bi100-privacy-safe-fatal-scan-v1",
                "qualified": not any(counts.values()),
                "category_counts": counts,
                "raw_messages_recorded": False,
            },
        )
        return counts

    def compare_preflights(self) -> None:
        rc = _run_to_files(
            [
                sys.executable,
                str(self.root / "tests" / "compare_bi100_preflights.py"),
                "--preflight",
                f"before={self.run_root / 'preflight_before.json'}",
                "--preflight",
                f"after={self.run_root / 'preflight_after.json'}",
                "--expected-gpus", "0,1,2,3",
                "--max-free-memory-drop-bytes", "1073741824",
                "--out", str(self.run_root / "preflight_comparison.json"),
            ],
            self.run_root / "preflight_comparison.stdout",
            self.run_root / "preflight_comparison.stderr",
            environment=self.base_environment(),
            timeout_s=300,
        )
        if rc:
            raise RuntimeError("preflight comparison failed")

    def source_unchanged(self) -> None:
        if _git(self.root, "rev-parse", "HEAD") != self.source_revision:
            raise RuntimeError("source revision changed during capture")
        if _git(
            self.root,
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            ".",
            ":(exclude)bench_runs/**",
        ):
            raise RuntimeError("source tree changed during capture")

    def write_status(self, returncode: int) -> None:
        timeline_report = summarize(
            self.timeline, expected_run_id=self.run_id)
        _atomic_json(self.run_root / "timeline_report.json", timeline_report)
        artifacts = {}
        for name in (
            "runtime_identity.json",
            "requests.json",
            "bank_qualification.json",
            "scoped_cleanup.json",
            "postflight_after.json",
            "preflight_comparison.json",
            "timeline_report.json",
        ):
            artifacts[name] = _sha256(self.run_root / name)
        status = {
            "schema": RUNNER_SCHEMA,
            "version": 1,
            "qualified": returncode == 0,
            "returncode": returncode,
            "terminal_stage": self.current_stage,
            "error_type": self.error_type,
            "source_revision": self.source_revision,
            "source_branch": self.source_branch,
            "instance": self.args.instance,
            "profile": self.args.profile,
            "gpu_count": 4,
            "tensor_parallel_size": 4,
            "service_startups": 1,
            "targets": [
                int(value) for value in self.args.targets.split(",")],
            "gates": self.gates,
            "artifact_sha256": artifacts,
            "timing": {
                "wall_span_s": timeline_report["wall_span_s"],
                "summed_stage_s": timeline_report["summed_stage_s"],
            },
            "authorization": {
                "activation_replay_authorized": returncode == 0,
                "short_tp4_authorized": False,
                "long_context_authorized": False,
                "main_or_yaml_change_authorized": False,
            },
            "privacy": {
                "raw_activation_bank_private_tmp_only": True,
                "raw_activation_bank_may_be_committed": False,
                "contains_credentials": False,
            },
        }
        _atomic_json(self.run_root / "runner_status.json", status)

    def retain_error(
        self,
        primary_error: BaseException | None,
        exc: BaseException,
    ) -> BaseException:
        if primary_error is None:
            self.error_type = type(exc).__name__
            return exc
        return primary_error

    def run_postconditions(
        self,
        primary_error: BaseException | None,
    ) -> BaseException | None:
        def run_postcondition(
            name: str,
            action: Callable[[], None],
        ) -> bool:
            nonlocal primary_error
            try:
                with self.stage(name):
                    action()
            except BaseException as exc:
                primary_error = self.retain_error(primary_error, exc)
                return False
            return True

        run_postcondition("scoped_cleanup", self.cleanup_service)
        postflight_ok = run_postcondition(
            "postflight_after",
            lambda: self.run_postflight("postflight_after"),
        )
        preflight_ok = False
        if postflight_ok:
            preflight_ok = run_postcondition(
                "preflight_after",
                lambda: self.run_preflight("preflight_after"),
            )
        else:
            self.gates["preflight_after"] = None
        if preflight_ok:
            run_postcondition(
                "preflight_comparison", self.compare_preflights)
        else:
            self.gates["preflight_comparison"] = None

        def require_clean_fatal_scan() -> None:
            counts = self.scan_fatal()
            if any(counts.values()):
                raise RuntimeError("fatal log categories were observed")

        run_postcondition("fatal_scan", require_clean_fatal_scan)
        run_postcondition("source_unchanged", self.source_unchanged)
        return primary_error

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
            with self.stage("capture_requests"):
                self.run_capture_requests()
            with self.stage("bank_qualification"):
                self.qualify_bank()
            with self.stage("health_after_capture"):
                if not _health():
                    raise RuntimeError("service health failed after capture")
        except BaseException as exc:
            primary_error = self.retain_error(primary_error, exc)

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
        "--profile", choices=("smoke", "qualification"),
        default="qualification")
    parser.add_argument("--targets", default="32768,65536,131072")
    parser.add_argument("--contexts", default="24576,57344,122880")
    parser.add_argument("--ordinals", default="0,4,9")
    args = parser.parse_args()
    runner = CaptureRunner(args)

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
            f"M1-140 capture runner failed: {type(exc).__name__}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
