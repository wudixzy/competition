#!/usr/bin/env python3
"""Compare M1-108 and M1-131 on identical BI100 production tensors."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

from tests import bench_m1_55_production_prefill as production


SCHEMA = "bi100-m1-131-exact-sum-softmax-cell-v1"
CONTROL_MODULE_NAME = "corex_fused_paged_prefill"
CANDIDATE_MODULE_NAME = "corex_fused_paged_prefill_exact_sum"
WARMUPS = 1
TRIALS = 5
RELATIVE_L2_LIMIT = 1e-5
MAX_ABS_LIMIT = 1e-3
CONTROL_EXTENSION_SHA256 = (
    "f654eee2c0677812394ff419d316e7e8"
    "c98ed1bcc84853a7f8d2ed5755503009"
)
CASES = {
    name: production.CASES[name]
    for name in (
        "production_dense_q8176",
        "production_65k_q8176",
        "production_128k_q8176",
        "production_235k_q5616",
    )
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _load_extension(
    path: Path,
    expected_sha256: str,
    module_name: str,
) -> tuple[Any, dict[str, Any]]:
    resolved = path.resolve(strict=True)
    actual_sha256 = production.sha256_file(resolved)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{module_name} SHA-256 mismatch: expected "
            f"{expected_sha256}, got {actual_sha256}"
        )
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension spec from {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    if not callable(getattr(module, "forward", None)):
        raise RuntimeError(f"{module_name} does not expose callable forward")
    return module, {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": actual_sha256,
    }


def _measure_once(
    function: Callable[[], tuple[Any, ...]],
) -> tuple[float, float, tuple[Any, ...]]:
    import torch

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    host_start = time.perf_counter()
    start_event.record()
    result = function()
    end_event.record()
    torch.cuda.synchronize()
    return (
        float(start_event.elapsed_time(end_event)),
        (time.perf_counter() - host_start) * 1000.0,
        result,
    )


def _measure_pair(
    control: Callable[[], tuple[Any, ...]],
    candidate: Callable[[], tuple[Any, ...]],
) -> tuple[dict[str, Any], tuple[Any, ...], tuple[Any, ...]]:
    import torch

    operations = {"control": control, "candidate": candidate}
    for _ in range(WARMUPS):
        for label in ("control", "candidate"):
            result = operations[label]()
            torch.cuda.synchronize()
            del result

    cuda_trials = {"control": [], "candidate": []}
    host_trials = {"control": [], "candidate": []}
    latest: dict[str, tuple[Any, ...]] = {}
    for trial in range(TRIALS):
        order = (
            ("control", "candidate")
            if trial % 2 == 0
            else ("candidate", "control")
        )
        for label in order:
            cuda_ms, host_ms, result = _measure_once(operations[label])
            cuda_trials[label].append(cuda_ms)
            host_trials[label].append(host_ms)
            latest[label] = result

    timings: dict[str, Any] = {}
    for label in ("control", "candidate"):
        timings[label] = {
            "warmups": WARMUPS,
            "trials": TRIALS,
            "cuda_trials_ms": cuda_trials[label],
            "cuda_median_ms": statistics.median(cuda_trials[label]),
            "host_trials_ms": host_trials[label],
            "host_median_ms": statistics.median(host_trials[label]),
        }
    timings["control_over_candidate_speedup"] = (
        timings["control"]["cuda_median_ms"]
        / timings["candidate"]["cuda_median_ms"]
    )
    return timings, latest["control"], latest["candidate"]


def _finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value >= 0
    )


def evaluate_cell(result: Any) -> dict[str, Any]:
    reasons: list[str] = []
    if not isinstance(result, dict):
        return {"qualified": False, "reasons": ["result must be an object"]}
    if result.get("schema") != SCHEMA:
        reasons.append(f"schema must equal {SCHEMA}")
    case_name = result.get("case")
    if case_name not in CASES:
        reasons.append("case is not in the frozen matrix")
        return {"qualified": False, "reasons": reasons}
    context_len, query_len, kind = CASES[case_name]
    for field, expected in (
        ("context_len", context_len),
        ("query_len", query_len),
        ("total_kv_len", context_len + query_len),
        ("kind", kind),
        ("seed", production.SEED),
        ("warmups", WARMUPS),
        ("trials", TRIALS),
        ("physical_block_permutation", context_len > 0),
    ):
        if result.get(field) != expected:
            reasons.append(f"{field} must equal {expected!r}")
    if not GIT_COMMIT_RE.fullmatch(str(result.get("source_commit", ""))):
        reasons.append("source_commit must be a full lowercase Git commit")
    if (
        not isinstance(result.get("runtime_identity"), str)
        or not result["runtime_identity"]
    ):
        reasons.append("runtime_identity must be a non-empty string")
    if (
        not isinstance(result.get("instance"), str)
        or not result["instance"]
    ):
        reasons.append("instance must be a non-empty string")
    visible_gpu = result.get("visible_physical_gpu")
    if (
        not isinstance(visible_gpu, int)
        or isinstance(visible_gpu, bool)
        or visible_gpu not in range(4)
    ):
        reasons.append("visible_physical_gpu must be in [0, 3]")

    extensions = result.get("extensions")
    if not isinstance(extensions, dict):
        reasons.append("extensions must be an object")
    else:
        for label in ("control", "candidate"):
            artifact = extensions.get(label)
            if (
                not isinstance(artifact, dict)
                or not SHA256_RE.fullmatch(str(artifact.get("sha256", "")))
                or not isinstance(artifact.get("size_bytes"), int)
                or artifact["size_bytes"] <= 0
            ):
                reasons.append(f"extensions.{label} identity is invalid")
        if (
            isinstance(extensions.get("control"), dict)
            and isinstance(extensions.get("candidate"), dict)
            and extensions["control"].get("sha256")
            == extensions["candidate"].get("sha256")
        ):
            reasons.append("control and candidate extensions are identical")
        control = extensions.get("control")
        if (
            isinstance(control, dict)
            and control.get("sha256") != CONTROL_EXTENSION_SHA256
        ):
            reasons.append("control extension is not the frozen M1-108 binary")

    output_contract = result.get("output_contract")
    required_contract_fields = (
        "control_result_arity_ok",
        "candidate_result_arity_ok",
        "candidate_repeat_contract_ok",
        "control_output_shape_ok",
        "candidate_output_shape_ok",
        "control_lse_shape_ok",
        "candidate_lse_shape_ok",
        "control_output_dtype_ok",
        "candidate_output_dtype_ok",
        "control_lse_dtype_ok",
        "candidate_lse_dtype_ok",
        "control_device_ok",
        "candidate_device_ok",
        "control_contiguous",
        "candidate_contiguous",
    )
    if not isinstance(output_contract, dict):
        reasons.append("output_contract must be an object")
    else:
        for field in required_contract_fields:
            if output_contract.get(field) is not True:
                reasons.append(f"output_contract.{field} must be true")

    numerical = result.get("numerical")
    if not isinstance(numerical, dict):
        reasons.append("numerical must be an object")
    else:
        for field in (
            "control_finite",
            "candidate_finite",
            "output_exact",
            "lse_exact",
            "candidate_repeat_output_exact",
            "candidate_repeat_lse_exact",
        ):
            if numerical.get(field) is not True:
                reasons.append(f"numerical.{field} must be true")
        for field, limit in (
            ("output_relative_l2", RELATIVE_L2_LIMIT),
            ("lse_relative_l2", RELATIVE_L2_LIMIT),
            ("output_max_abs", MAX_ABS_LIMIT),
            ("lse_max_abs", MAX_ABS_LIMIT),
        ):
            value = numerical.get(field)
            if not _finite_nonnegative(value):
                reasons.append(f"numerical.{field} is invalid")
            elif value > limit + 1e-12:
                reasons.append(
                    f"numerical.{field}={value:.9g} exceeds {limit:.9g}"
                )

    timings = result.get("timings")
    if not isinstance(timings, dict):
        reasons.append("timings must be an object")
    else:
        for label in ("control", "candidate"):
            timing = timings.get(label)
            if not isinstance(timing, dict):
                reasons.append(f"timings.{label} must be an object")
                continue
            values = timing.get("cuda_trials_ms")
            median = timing.get("cuda_median_ms")
            if (
                not isinstance(values, list)
                or len(values) != TRIALS
                or not all(
                    _finite_nonnegative(value) and value > 0
                    for value in values
                )
            ):
                reasons.append(
                    f"timings.{label}.cuda_trials_ms must contain "
                    f"{TRIALS} positive values"
                )
            elif (
                not _finite_nonnegative(median)
                or not math.isclose(
                    float(median),
                    statistics.median(values),
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            ):
                reasons.append(
                    f"timings.{label}.cuda_median_ms does not match trials"
                )
        speedup = timings.get("control_over_candidate_speedup")
        if not _finite_nonnegative(speedup):
            reasons.append("timings.control_over_candidate_speedup is invalid")
        else:
            control_timing = timings.get("control")
            candidate_timing = timings.get("candidate")
            control_median = (
                control_timing.get("cuda_median_ms")
                if isinstance(control_timing, dict)
                else None
            )
            candidate_median = (
                candidate_timing.get("cuda_median_ms")
                if isinstance(candidate_timing, dict)
                else None
            )
            if (
                _finite_nonnegative(control_median)
                and _finite_nonnegative(candidate_median)
                and candidate_median > 0
                and not math.isclose(
                    float(speedup),
                    float(control_median) / float(candidate_median),
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            ):
                reasons.append(
                    "timings.control_over_candidate_speedup "
                    "does not match medians"
                )

    if result.get("authorization") != {
        "tp4_service_authorized": False,
        "main_or_yaml_change_authorized": False,
        "official_score_claim_authorized": False,
    }:
        reasons.append("authorization must fail closed")
    return {"qualified": not reasons, "reasons": reasons}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/CoreX device is not available")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "benchmark requires exactly one visible device; set "
            "CUDA_VISIBLE_DEVICES to one healthy physical GPU"
        )

    context_len, query_len, kind = CASES[args.case]
    if args.expected_control_sha256 != CONTROL_EXTENSION_SHA256:
        raise RuntimeError("control extension is not the frozen M1-108 binary")
    control, control_artifact = _load_extension(
        args.control_extension,
        args.expected_control_sha256,
        CONTROL_MODULE_NAME,
    )
    candidate, candidate_artifact = _load_extension(
        args.candidate_extension,
        args.expected_candidate_sha256,
        CANDIDATE_MODULE_NAME,
    )
    inputs = production._make_inputs(context_len, query_len)
    scale = production.HEAD_DIM ** -0.5
    def invoke(module: Any) -> tuple[Any, ...]:
        value = module.forward(*inputs, context_len, scale)
        if isinstance(value, (list, tuple)):
            return tuple(value)
        return (value,)

    control_call = lambda: invoke(control)
    candidate_call = lambda: invoke(candidate)
    timings, control_result, candidate_result = _measure_pair(
        control_call, candidate_call
    )
    candidate_repeat = candidate_call()
    torch.cuda.synchronize()

    expected_output_shape = (
        query_len,
        production.NUM_QUERY_HEADS,
        production.HEAD_DIM,
    )
    expected_lse_shape = (query_len, production.NUM_QUERY_HEADS)
    query_device = inputs[0].device
    control_arity_ok = len(control_result) == 2
    candidate_arity_ok = len(candidate_result) == 2
    repeat_arity_ok = len(candidate_repeat) == 2
    control_output = control_result[0] if control_arity_ok else None
    control_lse = control_result[1] if control_arity_ok else None
    candidate_output = candidate_result[0] if candidate_arity_ok else None
    candidate_lse = candidate_result[1] if candidate_arity_ok else None
    repeat_output = candidate_repeat[0] if repeat_arity_ok else None
    repeat_lse = candidate_repeat[1] if repeat_arity_ok else None

    def tensor_matches(
        value: Any,
        shape: tuple[int, ...],
        dtype: Any,
    ) -> bool:
        return bool(
            isinstance(value, torch.Tensor)
            and tuple(value.shape) == shape
            and value.dtype == dtype
            and value.device == query_device
            and value.is_contiguous()
        )

    control_output_ok = tensor_matches(
        control_output, expected_output_shape, torch.float16
    )
    control_lse_ok = tensor_matches(
        control_lse, expected_lse_shape, torch.float32
    )
    candidate_output_ok = tensor_matches(
        candidate_output, expected_output_shape, torch.float16
    )
    candidate_lse_ok = tensor_matches(
        candidate_lse, expected_lse_shape, torch.float32
    )
    repeat_output_ok = tensor_matches(
        repeat_output, expected_output_shape, torch.float16
    )
    repeat_lse_ok = tensor_matches(
        repeat_lse, expected_lse_shape, torch.float32
    )
    output_contract = {
        "control_result_arity_ok": control_arity_ok,
        "candidate_result_arity_ok": candidate_arity_ok,
        "candidate_repeat_contract_ok": (
            repeat_arity_ok and repeat_output_ok and repeat_lse_ok
        ),
        "control_output_shape_ok": (
            isinstance(control_output, torch.Tensor)
            and tuple(control_output.shape) == expected_output_shape
        ),
        "candidate_output_shape_ok": (
            isinstance(candidate_output, torch.Tensor)
            and tuple(candidate_output.shape) == expected_output_shape
        ),
        "control_lse_shape_ok": (
            isinstance(control_lse, torch.Tensor)
            and tuple(control_lse.shape) == expected_lse_shape
        ),
        "candidate_lse_shape_ok": (
            isinstance(candidate_lse, torch.Tensor)
            and tuple(candidate_lse.shape) == expected_lse_shape
        ),
        "control_output_dtype_ok": (
            isinstance(control_output, torch.Tensor)
            and control_output.dtype == torch.float16
        ),
        "candidate_output_dtype_ok": (
            isinstance(candidate_output, torch.Tensor)
            and candidate_output.dtype == torch.float16
        ),
        "control_lse_dtype_ok": (
            isinstance(control_lse, torch.Tensor)
            and control_lse.dtype == torch.float32
        ),
        "candidate_lse_dtype_ok": (
            isinstance(candidate_lse, torch.Tensor)
            and candidate_lse.dtype == torch.float32
        ),
        "control_device_ok": (
            isinstance(control_output, torch.Tensor)
            and isinstance(control_lse, torch.Tensor)
            and control_output.device == query_device
            and control_lse.device == query_device
        ),
        "candidate_device_ok": (
            isinstance(candidate_output, torch.Tensor)
            and isinstance(candidate_lse, torch.Tensor)
            and candidate_output.device == query_device
            and candidate_lse.device == query_device
        ),
        "control_contiguous": bool(
            isinstance(control_output, torch.Tensor)
            and isinstance(control_lse, torch.Tensor)
            and control_output.is_contiguous()
            and control_lse.is_contiguous()
        ),
        "candidate_contiguous": bool(
            isinstance(candidate_output, torch.Tensor)
            and isinstance(candidate_lse, torch.Tensor)
            and candidate_output.is_contiguous()
            and candidate_lse.is_contiguous()
        ),
    }
    contract_qualified = all(output_contract.values())
    if contract_qualified:
        numerical = {
            "control_finite": bool(
                torch.isfinite(control_output).all().item()
                and torch.isfinite(control_lse).all().item()
            ),
            "candidate_finite": bool(
                torch.isfinite(candidate_output).all().item()
                and torch.isfinite(candidate_lse).all().item()
            ),
            "output_exact": bool(
                torch.equal(candidate_output, control_output)
            ),
            "lse_exact": bool(torch.equal(candidate_lse, control_lse)),
            "candidate_repeat_output_exact": bool(
                torch.equal(repeat_output, candidate_output)
            ),
            "candidate_repeat_lse_exact": bool(
                torch.equal(repeat_lse, candidate_lse)
            ),
            "output_relative_l2": production._relative_l2(
                candidate_output, control_output
            ),
            "lse_relative_l2": production._relative_l2(
                candidate_lse, control_lse
            ),
            "output_max_abs": float(
                (candidate_output.float() - control_output.float())
                .abs()
                .max()
                .item()
            ),
            "lse_max_abs": float(
                (candidate_lse - control_lse).abs().max().item()
            ),
        }
    else:
        numerical = {
            "control_finite": False,
            "candidate_finite": False,
            "output_exact": False,
            "lse_exact": False,
            "candidate_repeat_output_exact": False,
            "candidate_repeat_lse_exact": False,
            "output_relative_l2": None,
            "lse_relative_l2": None,
            "output_max_abs": None,
            "lse_max_abs": None,
        }
    result = {
        "schema": SCHEMA,
        "source_commit": args.source_commit,
        "runtime_identity": args.runtime_identity,
        "instance": args.instance,
        "visible_physical_gpu": args.visible_physical_gpu,
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "case": args.case,
        "kind": kind,
        "context_len": context_len,
        "query_len": query_len,
        "total_kv_len": context_len + query_len,
        "seed": production.SEED,
        "warmups": WARMUPS,
        "trials": TRIALS,
        "physical_block_permutation": context_len > 0,
        "extensions": {
            "control": control_artifact,
            "candidate": candidate_artifact,
        },
        "output_contract": output_contract,
        "timings": timings,
        "numerical": numerical,
        "authorization": {
            "tp4_service_authorized": False,
            "main_or_yaml_change_authorized": False,
            "official_score_claim_authorized": False,
        },
    }
    result["evaluation"] = evaluate_cell(result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--control-extension", type=Path, required=True)
    parser.add_argument("--candidate-extension", type=Path, required=True)
    parser.add_argument("--expected-control-sha256", required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--runtime-identity", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--visible-physical-gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "case": result["case"],
                "qualified": result["evaluation"]["qualified"],
                "speedup": result["timings"][
                    "control_over_candidate_speedup"
                ],
                "output_exact": result["numerical"]["output_exact"],
                "lse_exact": result["numerical"]["lse_exact"],
                "reasons": result["evaluation"]["reasons"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
