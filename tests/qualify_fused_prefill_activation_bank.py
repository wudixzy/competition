#!/usr/bin/env python3
"""Validate private activation-bank manifests without loading raw tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


MANIFEST_SCHEMA = "bi100-fused-prefill-activation-bank-v1"
RESULT_SCHEMA = "bi100-fused-prefill-activation-bank-qualification-v1"
CONTRACT_SCHEMA = "bi100-experiment-funnel-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            json.dump(value, stream, ensure_ascii=True, indent=2,
                      sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def qualify(
    manifests: list[tuple[Path, Any]],
    contract: Any,
    *,
    profile: str,
    run_id: str,
    source_revision: str,
    runtime_identity: str,
) -> dict[str, Any]:
    invalid_reasons = []
    coverage_reasons = []
    if (
        not isinstance(contract, dict)
        or contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("version") != 1
    ):
        invalid_reasons.append("experiment funnel contract is invalid")
        stages = []
    else:
        stages = contract.get("stages") or []
    l2 = next(
        (
            stage for stage in stages
            if isinstance(stage, dict) and stage.get("id") == "L2"
        ),
        {},
    )
    capture = l2.get("capture") or {}
    required_ranks = capture.get("required_tp_ranks")
    required_buckets = capture.get("required_context_buckets")
    required_ordinals = capture.get("required_full_attention_call_ordinals")
    if (
        required_ranks != [0, 1, 2, 3]
        or required_buckets != [24576, 57344, 122880]
        or required_ordinals != [0, 4, 9]
        or capture.get("raw_bank_location") != "private_tmp_only"
        or capture.get("raw_bank_may_be_committed") is not False
    ):
        invalid_reasons.append("L2 activation capture contract differs")

    ranks = set()
    observed = set()
    total_bytes = 0
    case_count = 0
    manifest_summaries = []
    for path, manifest in manifests:
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema") != MANIFEST_SCHEMA
            or manifest.get("version") != 1
            or manifest.get("run_id") != run_id
            or manifest.get("source_revision") != source_revision
            or manifest.get("runtime_identity") != runtime_identity
            or manifest.get("producer") != "baseline-pytorch-fallback"
            or manifest.get("synthetic_prompt_attestation")
            != "synthetic-exact-prompt-v1"
        ):
            invalid_reasons.append(f"{path.name}: manifest identity differs")
            continue
        rank = manifest.get("rank")
        records = manifest.get("records")
        privacy = manifest.get("privacy") or {}
        if (
            rank not in {0, 1, 2, 3}
            or rank in ranks
            or not isinstance(records, list)
            or not records
            or manifest.get("record_count") != len(records)
            or privacy.get("raw_activation_files_may_be_committed")
            is not False
        ):
            invalid_reasons.append(f"{path.name}: manifest structure differs")
            continue
        ranks.add(rank)
        for record in records:
            filename = record.get("file") if isinstance(record, dict) else None
            case = path.parent / filename if isinstance(filename, str) else None
            if (
                case is None
                or Path(filename).name != filename
                or not case.is_file()
                or case.stat().st_size != record.get("size_bytes")
                or _sha256(case) != record.get("sha256")
            ):
                invalid_reasons.append(
                    f"{path.name}: activation case identity differs")
                continue
            bucket = record.get("bucket_min_context_tokens")
            ordinal = record.get("call_ordinal")
            observed.add((rank, bucket, ordinal))
            total_bytes += case.stat().st_size
            case_count += 1
        manifest_summaries.append({
            "rank": rank,
            "manifest_sha256": _sha256(path),
            "record_count": len(records),
        })

    if ranks != {0, 1, 2, 3}:
        coverage_reasons.append("activation bank does not cover four TP ranks")
    if profile == "qualification":
        expected = {
            (rank, bucket, ordinal)
            for rank in required_ranks
            for bucket in required_buckets
            for ordinal in required_ordinals
        }
        missing = expected - observed
        extra = observed - expected
        if missing:
            coverage_reasons.append(
                f"qualification bank is missing {len(missing)} cases")
        if extra:
            coverage_reasons.append(
                f"qualification bank has {len(extra)} unexpected cases")
    elif profile == "smoke":
        for rank in required_ranks:
            if not any(row[0] == rank for row in observed):
                coverage_reasons.append(
                    f"smoke bank has no case for rank {rank}")
    else:
        invalid_reasons.append("unknown activation bank profile")

    qualified = not invalid_reasons and not coverage_reasons
    return {
        "schema": RESULT_SCHEMA,
        "version": 1,
        "profile": profile,
        "qualified": qualified,
        "invalid_reasons": invalid_reasons,
        "coverage_reasons": coverage_reasons,
        "run_id": run_id,
        "source_revision": source_revision,
        "runtime_identity": runtime_identity,
        "ranks": sorted(ranks),
        "case_count": case_count,
        "total_bytes": total_bytes,
        "manifests": sorted(
            manifest_summaries, key=lambda row: row["rank"]),
        "authorization": {
            "smoke_replay_authorized": (
                qualified and profile == "smoke"),
            "qualification_replay_authorized": (
                qualified and profile == "qualification"),
            "short_tp4_authorized": False,
            "main_or_yaml_change_authorized": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=("smoke", "qualification"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--runtime-identity", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifests = [
        (path.resolve(strict=True),
         json.loads(path.read_text(encoding="ascii")))
        for path in args.manifest
    ]
    contract = json.loads(args.contract.read_text(encoding="ascii"))
    result = qualify(
        manifests,
        contract,
        profile=args.profile,
        run_id=args.run_id,
        source_revision=args.source_revision,
        runtime_identity=args.runtime_identity,
    )
    result["contract_sha256"] = _sha256(args.contract)
    _atomic_write(args.out, result)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
