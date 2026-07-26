#!/usr/bin/env python3
"""Qualify the single-GPU/TP2 Qwen3.6 diagnostic service pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SERVICE_SCHEMA = "qwen36-diagnostic-service-gate-v1"
API_SCHEMA = "qwen36-diagnostic-api-gate-v1"
CHECKPOINT_SCHEMA = "qwen36-diagnostic-checkpoint-v1"
VERIFY_SCHEMA = "qwen36-diagnostic-checkpoint-verification-v1"
RUNTIME_SCHEMA = "bi100-bare-host-runtime-install-v2"
MAX_MODEL_LEN = 262144
LAYER_COUNT = 4
PARTIAL_CACHED_TOKENS = 8176
WARM_CACHED_TOKENS = 11600


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_status(
    report: dict[str, Any],
    tp_size: int,
    reasons: list[str],
) -> dict[str, Any]:
    prefix = f"TP{tp_size}"
    if report.get("schema") != SERVICE_SCHEMA:
        reasons.append(f"{prefix} service schema differs")
    if report.get("qualified") is not True:
        reasons.append(f"{prefix} service did not qualify")
    if report.get("semantic_quality_evaluated") is not False:
        reasons.append(f"{prefix} semantic scope marker differs")
    if report.get("production_promotion_authorized") is not False:
        reasons.append(f"{prefix} promotion marker differs")
    runtime = report.get("runtime_identity")
    if not isinstance(runtime, dict):
        reasons.append(f"{prefix} runtime identity is missing")
        runtime = {}
    if runtime.get("tensor_parallel_size") != tp_size:
        reasons.append(f"{prefix} tensor parallel size differs")
    if runtime.get("max_model_len") != MAX_MODEL_LEN:
        reasons.append(f"{prefix} max model length differs")
    if runtime.get("semantic_quality_evaluated") is not False:
        reasons.append(f"{prefix} runtime semantic scope marker differs")
    if runtime.get("production_promotion_authorized") is not False:
        reasons.append(f"{prefix} runtime promotion marker differs")
    gates = report.get("gates")
    if not isinstance(gates, dict) or not gates or any(
        value != 0 for value in gates.values()
    ):
        reasons.append(f"{prefix} service contains a failed or missing gate")
    if report.get("layer_trace_count", 0) < tp_size * LAYER_COUNT:
        reasons.append(f"{prefix} completed layer trace count is too small")
    api_summary = report.get("api_summary")
    if not isinstance(api_summary, dict) or (
        api_summary.get("qualified") is not True
        or api_summary.get("case_count") != 7
    ):
        reasons.append(f"{prefix} API summary differs")
    prefix_summary = report.get("prefix_summary")
    if not isinstance(prefix_summary, dict) or (
        prefix_summary.get("partial_cached_tokens")
        != PARTIAL_CACHED_TOKENS
        or prefix_summary.get("warm_cached_tokens")
        != WARM_CACHED_TOKENS
    ):
        reasons.append(f"{prefix} cached-token summary differs")
    return {
        "source_revision": runtime.get("source_revision"),
        "runtime_install_sha256": runtime.get("runtime_install_sha256"),
        "config_sha256": runtime.get("diagnostic_config_sha256"),
        "index_sha256": runtime.get("diagnostic_index_sha256"),
        "layer_trace_count": report.get("layer_trace_count"),
        "max_model_len": runtime.get("max_model_len"),
    }


def _check_preflight(
    before: dict[str, Any],
    after: dict[str, Any],
    expected_gpus: list[int],
    name: str,
    reasons: list[str],
) -> dict[str, Any]:
    if before.get("ok") is not True or after.get("ok") is not True:
        reasons.append(f"{name} GPU preflight did not pass")
    if before.get("gpus") != expected_gpus \
            or after.get("gpus") != expected_gpus:
        reasons.append(f"{name} physical GPU set differs")

    def rows(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
        values = report.get("results")
        if not isinstance(values, list):
            return {}
        return {
            row["gpu"]: row
            for row in values
            if isinstance(row, dict) and isinstance(row.get("gpu"), int)
        }

    before_rows = rows(before)
    after_rows = rows(after)
    if set(before_rows) != set(expected_gpus) \
            or set(after_rows) != set(expected_gpus):
        reasons.append(f"{name} GPU preflight rows differ")
    free_bytes: dict[str, dict[str, int | None]] = {}
    for gpu in expected_gpus:
        first = before_rows.get(gpu, {})
        last = after_rows.get(gpu, {})
        if first.get("ok") is not True or last.get("ok") is not True:
            reasons.append(f"{name} GPU{gpu} health differs")
        if first.get("free") != last.get("free") \
                or first.get("total") != last.get("total"):
            reasons.append(f"{name} GPU{gpu} memory was not fully restored")
        free_bytes[str(gpu)] = {
            "before": first.get("free"),
            "after": last.get("free"),
        }
    return {"physical_gpus": expected_gpus, "free_bytes": free_bytes}


def _checkpoint_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    output = manifest.get("output")
    if not isinstance(output, dict):
        output = {}
    shards = output.get("shards")
    if not isinstance(shards, list):
        shards = []
    return {
        "diagnostic": manifest.get("diagnostic"),
        "tensor_contract": manifest.get("tensor_contract"),
        "source": {
            key: manifest.get("source", {}).get(key)
            for key in ("architecture", "config_sha256", "index_sha256",
                        "model_type")
        },
        "config_sha256": output.get("config_sha256"),
        "index_sha256": output.get("index_sha256"),
        "shard_count": output.get("shard_count"),
        "shards": [
            {
                "file": row.get("file"),
                "bytes": row.get("bytes"),
                "sha256": row.get("sha256"),
            }
            for row in shards if isinstance(row, dict)
        ],
        "weight_count": output.get("weight_count"),
        "weight_payload_bytes": output.get("weight_payload_bytes"),
    }


def _normalize_api(report: dict[str, Any]) -> list[dict[str, Any]]:
    cases = report.get("cases")
    if not isinstance(cases, list):
        return []
    return [
        {
            "name": row.get("name"),
            "ok": row.get("ok"),
            "evidence": row.get("evidence"),
        }
        for row in cases if isinstance(row, dict)
    ]


def _normalize_prefix(report: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: value
        for key, value in report.items()
        if key not in ("primer", "partial_cache", "warm_cache")
    }
    for name in ("primer", "partial_cache", "warm_cache"):
        row = report.get(name)
        if isinstance(row, dict):
            result[name] = {
                key: value for key, value in row.items()
                if key != "elapsed_s"
            }
    return result


def _check_api_pair(
    tp1: dict[str, Any],
    tp2: dict[str, Any],
    reasons: list[str],
) -> dict[str, Any]:
    for name, report in (("TP1", tp1), ("TP2", tp2)):
        if report.get("schema") != API_SCHEMA \
                or report.get("qualified") is not True \
                or report.get("case_count") != 7:
            reasons.append(f"{name} API gate differs")
        if report.get("semantic_quality_evaluated") is not False \
                or report.get("production_promotion_authorized") is not False:
            reasons.append(f"{name} API scope marker differs")
    first = _normalize_api(tp1)
    second = _normalize_api(tp2)
    if first != second:
        reasons.append("TP1/TP2 API response structures or digests differ")
    names = [row.get("name") for row in first]
    expected_names = {
        "models_262144_contract",
        "deterministic_replay",
        "tool_message_surface",
        "reasoning_surface",
        "structured_output_surface",
        "multimodal_surface",
        "invalid_empty_messages_400",
    }
    if set(names) != expected_names:
        reasons.append("API case coverage differs")
    return {
        "case_count": len(first),
        "case_names": names,
        "response_evidence_exact_across_tp": first == second,
    }


def _check_prefix_pair(
    tp1: dict[str, Any],
    tp2: dict[str, Any],
    reasons: list[str],
) -> dict[str, Any]:
    first = _normalize_prefix(tp1)
    second = _normalize_prefix(tp2)
    if first != second:
        reasons.append("TP1/TP2 prefix-cache structures or digests differ")
    hashes: dict[str, Any] = {}
    for name, report in (("TP1", tp1), ("TP2", tp2)):
        rows = [
            report.get(key) for key in
            ("primer", "partial_cache", "warm_cache")
        ]
        digests = [
            row.get("message_sha256")
            for row in rows if isinstance(row, dict)
        ]
        if len(digests) != 3 or len(set(digests)) != 1:
            reasons.append(f"{name} cold/partial/warm output differs")
        hashes[name.lower()] = digests[0] if digests else None
    return {
        "exact_across_tp": first == second,
        "cold_partial_warm_sha256": hashes,
        "partial_cached_tokens": (
            tp1.get("partial_cache", {}).get("cached_tokens")),
        "warm_cached_tokens": (
            tp1.get("warm_cache", {}).get("cached_tokens")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-verify", type=Path, required=True)
    for prefix in ("tp1", "tp2"):
        parser.add_argument(f"--{prefix}-status", type=Path, required=True)
        parser.add_argument(f"--{prefix}-api", type=Path, required=True)
        parser.add_argument(f"--{prefix}-prefix", type=Path, required=True)
        parser.add_argument(f"--{prefix}-runtime", type=Path, required=True)
        parser.add_argument(f"--{prefix}-manifest", type=Path, required=True)
        parser.add_argument(
            f"--{prefix}-preflight-before", type=Path, required=True)
        parser.add_argument(
            f"--{prefix}-preflight-after", type=Path, required=True)
    parser.add_argument("--tp2-nccl", type=Path, required=True)
    parser.add_argument("--gdn-broadcast", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        key: value for key, value in vars(args).items()
        if isinstance(value, Path) and key != "out"
    }
    reasons: list[str] = []
    report: dict[str, Any] = {
        "schema": "qwen36-diagnostic-service-pair-v1",
        "version": 1,
        "scope": "real-weight four-layer diagnostic model, TP1 and TP2",
        "semantic_quality_evaluated": False,
        "full_model_tp4_evaluated": False,
        "official_performance_evaluated": False,
        "production_promotion_authorized": False,
    }
    try:
        values = {name: _load(path) for name, path in paths.items()}
        verify = values["checkpoint_verify"]
        if verify.get("schema") != VERIFY_SCHEMA \
                or verify.get("qualified") is not True \
                or verify.get("full_hash_checked") is not True \
                or verify.get("source_payload_bytes_compared") is not True \
                or verify.get("tensor_contract_preserved") is not True \
                or verify.get("layer_count") != LAYER_COUNT:
            reasons.append("checkpoint source-byte verification differs")

        tp1_status = _check_status(values["tp1_status"], 1, reasons)
        tp2_status = _check_status(values["tp2_status"], 2, reasons)
        report["services"] = {"tp1": tp1_status, "tp2": tp2_status}

        tp1_runtime = values["tp1_runtime"]
        tp2_runtime = values["tp2_runtime"]
        for name, runtime, status in (
            ("TP1", tp1_runtime, tp1_status),
            ("TP2", tp2_runtime, tp2_status),
        ):
            if runtime.get("schema") != RUNTIME_SCHEMA \
                    or runtime.get("qualified") is not True:
                reasons.append(f"{name} immutable runtime did not qualify")
            if runtime.get("source_revision") != status["source_revision"]:
                reasons.append(f"{name} runtime source revision differs")
        tree1 = tp1_runtime.get("runtime_tree_sha256")
        tree2 = tp2_runtime.get("runtime_tree_sha256")
        if not isinstance(tree1, str) or tree1 != tree2:
            reasons.append("TP1/TP2 runtime trees differ")
        report["runtime"] = {
            "tree_sha256": tree1,
            "identical_across_tp": tree1 == tree2,
            "source_revisions": [
                tp1_runtime.get("source_revision"),
                tp2_runtime.get("source_revision"),
            ],
        }

        for name, manifest in (
            ("TP1", values["tp1_manifest"]),
            ("TP2", values["tp2_manifest"]),
        ):
            if manifest.get("schema") != CHECKPOINT_SCHEMA:
                reasons.append(f"{name} checkpoint manifest schema differs")
        checkpoint1 = _checkpoint_identity(values["tp1_manifest"])
        checkpoint2 = _checkpoint_identity(values["tp2_manifest"])
        if checkpoint1 != checkpoint2:
            reasons.append("TP1/TP2 checkpoint tensor identities differ")
        if tp1_status["config_sha256"] != checkpoint1["config_sha256"] \
                or tp2_status["config_sha256"] != checkpoint2["config_sha256"]:
            reasons.append("service checkpoint config identity differs")
        if tp1_status["index_sha256"] != checkpoint1["index_sha256"] \
                or tp2_status["index_sha256"] != checkpoint2["index_sha256"]:
            reasons.append("service checkpoint index identity differs")
        report["checkpoint"] = {
            "identical_across_tp": checkpoint1 == checkpoint2,
            "config_sha256": checkpoint1["config_sha256"],
            "index_sha256": checkpoint1["index_sha256"],
            "shard_count": checkpoint1["shard_count"],
            "shard_sha256": [
                row["sha256"] for row in checkpoint1["shards"]
            ],
            "weight_payload_bytes": checkpoint1["weight_payload_bytes"],
            "source_payload_bytes_compared": (
                verify.get("source_payload_bytes_compared")),
            "tensor_contract": checkpoint1["tensor_contract"],
        }

        report["api"] = _check_api_pair(
            values["tp1_api"], values["tp2_api"], reasons)
        report["prefix_cache"] = _check_prefix_pair(
            values["tp1_prefix"], values["tp2_prefix"], reasons)
        report["gpu_health"] = {
            "tp1": _check_preflight(
                values["tp1_preflight_before"],
                values["tp1_preflight_after"],
                [3], "TP1", reasons),
            "tp2": _check_preflight(
                values["tp2_preflight_before"],
                values["tp2_preflight_after"],
                [1, 2], "TP2", reasons),
        }
        nccl = values["tp2_nccl"]
        if nccl.get("ok") is not True \
                or nccl.get("timed_out_ranks") != [] \
                or len(nccl.get("results", [])) != 2:
            reasons.append("TP2 NCCL preflight did not pass")
        broadcast = values["gdn_broadcast"]
        if broadcast.get("qualified") is not True \
                or broadcast.get(
                    "missing_restore_fail_fast_source_attested") is not True:
            reasons.append("GDN action broadcast/fail-fast gate did not pass")
        report["distributed"] = {
            "tp2_nccl_qualified": nccl.get("ok"),
            "gdn_action_broadcast_qualified": broadcast.get("qualified"),
            "missing_restore_fail_fast_source_attested": broadcast.get(
                "missing_restore_fail_fast_source_attested"),
        }
        report["evidence_sha256"] = {
            name: _sha256(path) for name, path in paths.items()
        }
    except Exception as exc:
        reasons.append(f"qualification error {type(exc).__name__}: {exc}")

    report["qualified"] = not reasons
    report["reasons"] = reasons
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "out": str(args.out),
        "qualified": report["qualified"],
        "reasons": reasons,
    }, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
