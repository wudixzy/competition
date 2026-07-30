#!/usr/bin/env python3
"""Run the bounded M1-160 ixinfer FMHA capability matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any


SCHEMA = "bi100-m1-160-ixinfer-fmha-capability-matrix-v1"
STATUS_PATTERN = re.compile(r"CUINFER_STATUS_[A-Z_]+")
CASES = (
    {
        "name": "bshd_d128_mha",
        "query_length": 128,
        "key_length": 128,
        "query_heads": 4,
        "kv_heads": 4,
        "head_size": 128,
        "layout": "bshd",
        "causal": False,
    },
    {
        "name": "bshd_d256_mha",
        "query_length": 128,
        "key_length": 128,
        "query_heads": 4,
        "kv_heads": 4,
        "head_size": 256,
        "layout": "bshd",
        "causal": False,
    },
    {
        "name": "bhsd_d128_mha",
        "query_length": 128,
        "key_length": 128,
        "query_heads": 4,
        "kv_heads": 4,
        "head_size": 128,
        "layout": "bhsd",
        "causal": False,
    },
    {
        "name": "bshd_d256_gqa_causal",
        "query_length": 16,
        "key_length": 32,
        "query_heads": 4,
        "kv_heads": 1,
        "head_size": 256,
        "layout": "bshd",
        "causal": True,
    },
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _run_case(
    args: argparse.Namespace,
    case: dict[str, Any],
    physical_gpu: int,
    temporary: Path,
) -> dict[str, Any]:
    output = temporary / f"{case['name']}.json"
    command = [
        args.python,
        str(args.probe.resolve(strict=True)),
        "--extension",
        str(args.extension.resolve(strict=True)),
        "--expected-sha256",
        args.expected_sha256,
        "--source-revision",
        args.source_revision,
        "--runtime-identity",
        args.runtime_identity,
        "--instance",
        args.instance,
        "--visible-physical-gpu",
        str(physical_gpu),
        "--query-length",
        str(case["query_length"]),
        "--key-length",
        str(case["key_length"]),
        "--query-heads",
        str(case["query_heads"]),
        "--kv-heads",
        str(case["kv_heads"]),
        "--head-size",
        str(case["head_size"]),
        "--layout",
        str(case["layout"]),
        "--causal" if case["causal"] else "--no-causal",
        "--trials",
        "3",
        "--output",
        str(output),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    environment["PYTHONUNBUFFERED"] = "1"
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=args.timeout_s,
            env=environment,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as error:
        timed_out = True
        returncode = None
        stdout = error.stdout or b""
        stderr = error.stderr or b""
    decoded = stderr.decode("utf-8", "replace")
    statuses = sorted(set(STATUS_PATTERN.findall(decoded)))
    result = {
        "name": case["name"],
        "shape": {
            key: case[key]
            for key in (
                "query_length",
                "key_length",
                "query_heads",
                "kv_heads",
                "head_size",
                "layout",
                "causal",
            )
        },
        "visible_physical_gpu": physical_gpu,
        "returncode": returncode,
        "timed_out": timed_out,
        "result_written": output.is_file(),
        "stdout_sha256": _sha256_bytes(stdout),
        "stderr_sha256": _sha256_bytes(stderr),
        "cuinfer_statuses": statuses,
    }
    if output.is_file():
        result["result_sha256"] = _sha256(output)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.gpus) != len(CASES):
        raise ValueError(f"exactly {len(CASES)} GPU assignments are required")
    extension = args.extension.resolve(strict=True)
    if _sha256(extension) != args.expected_sha256:
        raise ValueError("extension SHA-256 mismatch")
    with tempfile.TemporaryDirectory(
        prefix="m1-160-ixinfer-", dir="/tmp"
    ) as name:
        temporary = Path(name)
        rows = [
            _run_case(args, case, gpu, temporary)
            for case, gpu in zip(CASES, args.gpus, strict=True)
        ]
    all_bad_param = all(
        row["returncode"] not in (None, 0)
        and row["timed_out"] is False
        and row["result_written"] is False
        and row["cuinfer_statuses"] == ["CUINFER_STATUS_BAD_PARAM"]
        for row in rows
    )
    return {
        "schema": SCHEMA,
        "version": 1,
        "source_revision": args.source_revision,
        "runtime_identity": args.runtime_identity,
        "instance": args.instance,
        "probe_sha256": _sha256(args.probe.resolve(strict=True)),
        "extension_sha256": args.expected_sha256,
        "timeout_s_per_case": args.timeout_s,
        "cases": rows,
        "conclusion": {
            "all_dispatches_rejected_bad_param": all_bad_param,
            "documented_contract_usable": not all_bad_param,
            "continue_ixinfer_parameter_guessing": False,
        },
        "authorization": {
            "runtime_overlay_authorized": False,
            "tp4_service_authorized": False,
            "main_or_yaml_change_authorized": False,
        },
        "privacy": {
            "tensor_values_recorded": False,
            "model_outputs_recorded": False,
            "credentials_recorded": False,
            "full_stderr_recorded": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="python3")
    parser.add_argument(
        "--probe",
        type=Path,
        default=Path(__file__).with_name(
            "run_corex_ixinfer_fmha_probe.py"
        ),
    )
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--runtime-identity", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument(
        "--gpus",
        type=lambda value: [int(item) for item in value.split(",")],
        default=[1, 2, 3, 1],
    )
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            result,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "conclusion": result["conclusion"],
                "statuses": [
                    row["cuinfer_statuses"] for row in result["cases"]
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if result["conclusion"][
        "all_dispatches_rejected_bad_param"
    ] else 1


if __name__ == "__main__":
    raise SystemExit(main())
