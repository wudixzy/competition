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
PREBUILT_COREX_DIR = Path(
    "qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10")
PREBUILT_COREX_SHA256 = {
    "corex_attn_head_rms_norm.so":
        "534019b3c2ad2d2c65492b01a975874ee440026eda2e8666bc3c1dc8a0a0a6f6",
    "corex_block_major_kv_transfer.so":
        "47c10acfbb3ec7d190c566d73b7616beea1fccc9ac89f336218144211f6fd1a5",
    "corex_gdn_beta_decay.so":
        "1856c86e3100415061aa698a48bdeff3fe785994b45b4e72a42cd9158552a7d8",
    "corex_gdn_causal_conv.so":
        "957c7518f5831299fc73f19a4ca2aa3c8231afe9ea7c979127b4f426cd9d6906",
    "corex_gdn_gated_norm.so":
        "ec2d11fa82d9d0816a6da53e62605e962786fa20ecd5f62e50f9d43087fc4d67",
    "corex_gdn_packed_decode.so":
        "27b7ae2ce4fe173336355d72a2678d043df4bd1ed85e9231a99bfb81885a6ce3",
    "corex_gdn_qk_map.so":
        "015b61046ad73d8f12d754f7a87d4f6cba33070af1c079879e15b71a94571670",
    "corex_moe_direct_routed.so":
        "0eb120e89608bb5b64ca4356a5d3d362121806d081ccc1ccf346dac472a819ec",
    "corex_moe_exact_reduce.so":
        "d26f2fa39c3921a95793786601e90cf6ebadd06f1d752af541bf82c21acbc1c9",
    "corex_moe_weight_gather.so":
        "50b0b44c1da779bb2c03419ed549aee9bb922d1f9bab8b7f11a3d91cca0d21c3",
    "corex_paged_kv_gather.so":
        "e944ec0528ed9b6cb74518de3c57e3730543a7bdebc872f993bfdc8424f13e6b",
}

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
    "BI100_KV_EVICTION_POLICY",
    "BI100_CPU_KV_OFFLOAD",
    "BI100_CPU_KV_TRANSFER_LAYOUT",
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
        *sorted((root / "scripts").glob("*.sh")),
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
            "FROM harbor.4pd.io/modelhubxc/enginex-iluvatar/"
            "bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3"
        )
        if text.splitlines()[0] != expected_base:
            raise ValueError(
                "base image must use the current ModelHub Harbor endpoint")
        for fragment in [
            "COPY ./qwen3_6_scripts /workspace/qwen3_6_scripts",
            "RUN cd ./qwen3_6_scripts && bash ./patch_ops.sh",
            "ENABLE_CUSTOM_IPC=1",
            "BI100_EXECUTOR_STARTUP_DEBUG=1",
        ]:
            if fragment not in text:
                raise ValueError(f"Dockerfile missing: {fragment}")
        registry_patcher = (
            root / "qwen3_6_scripts" / "patch_vllm_qwen3_5.py"
        ).read_text(encoding="utf-8")
        for forbidden in ("exec_module", "import torch", "load_library"):
            if forbidden in registry_patcher:
                raise ValueError(
                    "model registry patch executes runtime code during image "
                    f"construction: {forbidden}")
        if "ast.parse" not in registry_patcher:
            raise ValueError("model registry patch lacks static AST verification")
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

    def prebuilt_corex_assets() -> str:
        directory = root / PREBUILT_COREX_DIR
        actual_names = {
            path.name for path in directory.glob("corex_*.so")
            if path.is_file()
        }
        expected_names = set(PREBUILT_COREX_SHA256)
        if actual_names != expected_names:
            raise ValueError(
                "prebuilt CoreX artifact set changed: "
                f"missing={sorted(expected_names - actual_names)} "
                f"extra={sorted(actual_names - expected_names)}")
        total_size = 0
        for name, expected_digest in PREBUILT_COREX_SHA256.items():
            path = directory / name
            size = path.stat().st_size
            digest = _sha256(path)
            if size < 64 * 1024 or digest != expected_digest:
                raise ValueError(
                    f"prebuilt CoreX mismatch: {name} size={size} "
                    f"sha256={digest}")
            total_size += size
        manifest = (directory / "SHA256SUMS").read_text(
            encoding="utf-8").splitlines()
        expected_manifest = [
            f"{digest}  {name}"
            for name, digest in PREBUILT_COREX_SHA256.items()
        ]
        if manifest != expected_manifest:
            raise ValueError("prebuilt CoreX SHA256SUMS changed")
        return f"{len(expected_names)} artifacts, {total_size} bytes"

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
                   *sorted((root / "qwen3_6_scripts").glob("*.sh")),
                   *sorted((root / "scripts").glob("*.sh"))]
        for path in scripts:
            subprocess.run(
                ["bash", "-n", str(path)], check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return f"bash -n passed for {len(scripts)} scripts"

    def python_syntax() -> str:
        paths = [
            *sorted((root / "qwen3_6_scripts").rglob("*.py")),
            *sorted((root / "scripts").glob("*.py")),
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
    check("prebuilt_corex_extensions", prebuilt_corex_assets)
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
