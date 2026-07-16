#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
WHEEL = Path("qwen3_6_scripts/wheels/transformers-4.55.3-py3-none-any.whl")
WHEEL_SIZE = 11_269_669
WHEEL_SHA256 = "c85e7feace634541e23b3e34d28aa9492d67974b733237ade9eba7c57c0fd1bd"

EXPECTED_COMMAND = [
    "python3",
    "-m",
    "vllm.entrypoints.openai.api_server",
    "--model",
    "/model",
    "--served-model-name",
    "llm",
    "--max-model-len",
    "262144",
    "--gpu-memory-utilization",
    "0.9",
    "--trust-remote-code",
    "-tp",
    "4",
    "--max-num-seqs",
    "1",
    "--disable-log-requests",
    "--disable-frontend-multiprocessing",
    "--max-num-batched-tokens",
    "8192",
    "--enable-chunked-prefill",
    "--max-seq-len-to-capture",
    "32768",
    "--enable-auto-tool-choice",
    "--tool-call-parser",
    "qwen3_coder",
    "--reasoning-parser",
    "qwen3",
    "--enable-prefix-caching",
]
EXPECTED_ENV = {
    "VLLM_ENGINE_ITERATION_TIMEOUT_S": "3600",
    "BI100_MOE_COREX_DIRECT_ROUTED": "1",
    "BI100_GDN_COREX_PACKED_DECODE": "1",
}
DIAGNOSTIC_ENV = {
    "BI100_CACHE_TRACE",
    "BI100_PROFILE",
    "BI100_PROFILE_INCLUDE_STARTUP",
    "BI100_PAGED_ATTN_DIAGNOSTICS",
    "BI100_GDN_ALLOW_NAN_ZERO",
    "NUM_GPU_BLOCKS_OVERRIDE",
    "BI100_MOE_COREX_THREE_BUCKET",
}


def _scalar(value: str) -> str:
    value = value.strip()
    if value[:1] in {"'", '"'}:
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, (str, int, float)):
            raise ValueError(f"unsupported YAML scalar: {value}")
        return str(parsed)
    return value


def parse_run_config(text: str) -> tuple[int, list[str], dict[str, str]]:
    concurrency = None
    command: list[str] = []
    environment: dict[str, str] = {}
    section = None
    pending_name = None

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("concurrency:"):
            concurrency = int(_scalar(stripped.split(":", 1)[1]))
            continue
        if stripped == "command:":
            section = "command"
            continue
        if stripped == "env:":
            section = "env"
            continue
        if section == "command" and stripped.startswith("- "):
            command.append(_scalar(stripped[2:]))
            continue
        if section == "env" and stripped.startswith("- name:"):
            pending_name = _scalar(stripped.split(":", 1)[1])
            if pending_name in environment:
                raise ValueError(f"duplicate environment variable: {pending_name}")
            continue
        if section == "env" and stripped.startswith("value:"):
            if pending_name is None:
                raise ValueError("environment value without name")
            environment[pending_name] = _scalar(stripped.split(":", 1)[1])
            pending_name = None
            continue
        raise ValueError(f"unsupported run-config line: {raw_line!r}")

    if concurrency is None:
        raise ValueError("missing concurrency")
    if pending_name is not None:
        raise ValueError(f"environment variable without value: {pending_name}")
    return concurrency, command, environment


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _critical_text_files(root: Path) -> list[Path]:
    return [
        root / "Dockerfile",
        root / "computility-run.yaml",
        root / "launch_service",
        *sorted((root / "qwen3_6_scripts").glob("*.sh")),
    ]


def run_checks(root: Path = ROOT) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []

    def check(name: str, function: Callable[[], str]) -> None:
        try:
            detail = function()
        except Exception as exc:
            results.append({"name": name, "ok": False, "detail": str(exc)})
        else:
            results.append({"name": name, "ok": True, "detail": detail})

    def root_manifest() -> str:
        required = ["Dockerfile", "computility-run.yaml", "qwen3_6_scripts"]
        missing = [name for name in required if not (root / name).exists()]
        if missing:
            raise ValueError(f"missing required paths: {missing}")
        return ", ".join(required)

    def run_contract() -> str:
        text = (root / "computility-run.yaml").read_text(encoding="utf-8")
        concurrency, command, environment = parse_run_config(text)
        if concurrency != 1:
            raise ValueError(f"concurrency changed: {concurrency}")
        if command != EXPECTED_COMMAND:
            raise ValueError("command does not match the qualified fixed contract")
        if environment != EXPECTED_ENV:
            raise ValueError(f"submission environment changed: {environment}")
        leaked = sorted(DIAGNOSTIC_ENV.intersection(environment))
        if leaked:
            raise ValueError(f"diagnostic environment leaked: {leaked}")
        return f"{len(command)} argv entries, {len(environment)} env entries"

    def docker_contract() -> str:
        text = (root / "Dockerfile").read_text(encoding="utf-8")
        expected_base = (
            "FROM git.modelhub.org.cn:9443/enginex-iluvatar/"
            "bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3"
        )
        if text.splitlines()[0] != expected_base:
            raise ValueError("base image changed")
        for fragment in [
            "COPY ./qwen3_6_scripts /workspace/qwen3_6_scripts",
            "RUN cd ./qwen3_6_scripts && bash ./patch_ops.sh",
            "ENABLE_CUSTOM_IPC=1",
            "BI100_EXECUTOR_STARTUP_DEBUG=1",
        ]:
            if fragment not in text:
                raise ValueError(f"Dockerfile missing: {fragment}")
        return "base image and offline patch entrypoint match"

    def wheel_asset() -> str:
        path = root / WHEEL
        if not path.is_file():
            raise ValueError(f"missing offline wheel: {WHEEL}")
        size = path.stat().st_size
        digest = _sha256(path)
        if size != WHEEL_SIZE or digest != WHEEL_SHA256:
            raise ValueError(f"wheel mismatch: size={size} sha256={digest}")
        return f"size={size} sha256={digest}"

    def line_endings() -> str:
        offenders = []
        for path in _critical_text_files(root):
            data = path.read_bytes()
            if b"\r" in data or not data.endswith(b"\n"):
                offenders.append(str(path.relative_to(root)))
        if offenders:
            raise ValueError(f"non-LF or missing final newline: {offenders}")
        return f"{len(_critical_text_files(root))} critical files are LF-only"

    def shell_syntax() -> str:
        scripts = [root / "launch_service",
                   *sorted((root / "qwen3_6_scripts").glob("*.sh"))]
        for path in scripts:
            subprocess.run(
                ["bash", "-n", str(path)], check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return f"bash -n passed for {len(scripts)} scripts"

    def python_syntax() -> str:
        paths = [
            *sorted((root / "qwen3_6_scripts").rglob("*.py")),
            *sorted((root / "tests").glob("*.py")),
        ]
        for path in paths:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
        return f"compiled {len(paths)} Python sources"

    check("root_manifest", root_manifest)
    check("run_contract", run_contract)
    check("docker_contract", docker_contract)
    check("offline_transformers_wheel", wheel_asset)
    check("line_endings", line_endings)
    check("shell_syntax", shell_syntax)
    check("python_syntax", python_syntax)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the submission RC")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    results = run_checks(args.root.resolve())
    report = {
        "ok": all(item["ok"] for item in results),
        "root": str(args.root.resolve()),
        "checks": results,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.out:
        args.out.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
