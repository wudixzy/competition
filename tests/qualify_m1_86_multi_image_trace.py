#!/usr/bin/env python3
"""Qualify M1-86 multimodal cache isolation from privacy-safe v4 traces."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


Json = dict[str, Any]
SCHEMA = "bi100-m1-86-multi-image-trace-v1"
VERSION = 1
HTTP_SCHEMA = "qwen36-diagnostic-multi-image-http-gate-v1"
TRACE_MARKER = "[BI100_CACHE_TRACE] "
CASE_NAMES = (
    "models_262144_contract",
    "stream_one_image_cold",
    "stream_two_images_cold",
    "stream_two_images_warm",
    "stream_two_images_reversed",
    "stream_two_images_reversed_warm",
    "post_request_health",
)
CANDIDATE_RECORD_CASES = (
    "stream_one_image_cold",
    "stream_two_images_cold",
    "stream_two_images_warm",
    "stream_two_images_reversed",
    "stream_two_images_reversed_warm",
)
CONTROL_RECORD_CASES = ("stream_one_image_cold",)


def _load(path: Path) -> Json:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _case_map(report: Json, reasons: list[str]) -> dict[str, Json]:
    cases = report.get("cases")
    if not isinstance(cases, list):
        reasons.append("HTTP report cases are missing")
        return {}
    result: dict[str, Json] = {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("name"), str):
            reasons.append("HTTP report contains a malformed case")
            continue
        if case["name"] in result:
            reasons.append(f"HTTP report duplicates case {case['name']}")
        result[case["name"]] = case
    if tuple(result) != CASE_NAMES:
        reasons.append("HTTP report case order or identity differs")
    return result


def _evidence(
    cases: dict[str, Json],
    name: str,
    reasons: list[str],
) -> Json:
    case = cases.get(name)
    if not isinstance(case, dict) or case.get("ok") is not True:
        reasons.append(f"HTTP case {name} did not pass")
        return {}
    evidence = case.get("evidence")
    if not isinstance(evidence, dict):
        reasons.append(f"HTTP case {name} has no evidence")
        return {}
    return evidence


def _trace_records(path: Path, reasons: list[str]) -> list[Json]:
    records: list[Json] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            if TRACE_MARKER not in line:
                continue
            payload = line.split(TRACE_MARKER, 1)[1].strip()
            try:
                value = json.loads(payload)
            except json.JSONDecodeError:
                reasons.append(
                    f"cache trace line {line_number} is malformed")
                continue
            if not isinstance(value, dict):
                reasons.append(
                    f"cache trace line {line_number} is not an object")
                continue
            records.append(value)
    return records


def _integer(value: Any, *, minimum: int = 0) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= minimum
    )


def _prompt_hashes(
    record: Json,
    label: str,
    reasons: list[str],
) -> tuple[bytes, ...]:
    block_size = record.get("block_size")
    prompt_tokens = record.get("prompt_tokens")
    full_blocks = record.get("full_blocks")
    if (
        not _integer(block_size, minimum=1)
        or not _integer(prompt_tokens)
        or not _integer(full_blocks)
        or record.get("version") != 4
        or record.get("hash_encoding") != "sha256_base64"
    ):
        reasons.append(f"{label} trace shape differs")
        return ()
    encoded = record.get("block_hashes")
    if not isinstance(encoded, str):
        reasons.append(f"{label} block hashes are missing")
        return ()
    try:
        packed = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError, TypeError):
        reasons.append(f"{label} block hashes are malformed")
        return ()
    if len(packed) != full_blocks * 32:
        reasons.append(f"{label} block hash count differs")
        return ()
    prompt_full_blocks = prompt_tokens // block_size
    if prompt_full_blocks > full_blocks:
        reasons.append(f"{label} prompt exceeds finalized full blocks")
        return ()
    return tuple(
        packed[offset:offset + 32]
        for offset in range(0, prompt_full_blocks * 32, 32)
    )


def _common_prefix(left: tuple[bytes, ...], right: tuple[bytes, ...]) -> int:
    count = 0
    for left_hash, right_hash in zip(left, right):
        if left_hash != right_hash:
            break
        count += 1
    return count


def _trace_summary(
    record: Json,
    prompt_hashes: tuple[bytes, ...],
) -> Json:
    return {
        "ordinal": record.get("ordinal"),
        "prompt_tokens": record.get("prompt_tokens"),
        "prompt_full_blocks": len(prompt_hashes),
        "prompt_chain_sha256": hashlib.sha256(
            b"".join(prompt_hashes)).hexdigest(),
        "initial_raw_kv_contiguous_hit_blocks":
            record.get("initial_raw_kv_contiguous_hit_blocks"),
        "raw_kv_contiguous_hit_blocks":
            record.get("raw_kv_contiguous_hit_blocks"),
        "effective_gdn_hit_blocks":
            record.get("effective_gdn_hit_blocks"),
        "observed_effective_cached_tokens":
            record.get("observed_effective_cached_tokens"),
    }


def qualify(log_path: Path, report: Json, mode: str) -> Json:
    if mode not in ("control", "candidate"):
        raise ValueError("mode must be control or candidate")
    reasons: list[str] = []
    if (
        report.get("schema") != HTTP_SCHEMA
        or report.get("version") != 1
        or report.get("qualified") is not True
    ):
        reasons.append("HTTP report is not qualified")
    cases = _case_map(report, reasons)
    case_names = (
        CONTROL_RECORD_CASES if mode == "control"
        else CANDIDATE_RECORD_CASES
    )
    evidence = {
        name: _evidence(cases, name, reasons) for name in case_names
    }

    records = _trace_records(log_path, reasons)
    records.sort(key=lambda item: (
        item.get("ordinal")
        if _integer(item.get("ordinal"), minimum=1)
        else -1
    ))
    if len(records) != len(case_names):
        reasons.append(
            f"{mode} trace count is {len(records)}, "
            f"expected {len(case_names)}")
    expected_ordinals = list(range(1, len(records) + 1))
    if [record.get("ordinal") for record in records] != expected_ordinals:
        reasons.append("cache trace ordinals are not contiguous")
    request_ids = [record.get("request_id_sha256") for record in records]
    if (
        not all(isinstance(value, str)
                and len(value) == 16
                and all(character in "0123456789abcdef"
                        for character in value)
                for value in request_ids)
        or len(set(request_ids)) != len(request_ids)
    ):
        reasons.append("cache trace request identities differ")
    session_values = [
        record.get("trace_session_sha256") for record in records
    ]
    if (
        not session_values
        or not all(isinstance(value, str) and len(value) == 16
                   for value in session_values)
        or len(set(session_values)) != 1
    ):
        reasons.append("cache trace session identity differs")

    prompt_hashes: list[tuple[bytes, ...]] = []
    summaries: dict[str, Json] = {}
    for index, record in enumerate(records):
        label = case_names[index] if index < len(case_names) else (
            f"unexpected_{index + 1}")
        hashes = _prompt_hashes(record, label, reasons)
        prompt_hashes.append(hashes)
        raw_hit = record.get("raw_kv_contiguous_hit_blocks")
        initial_raw_hit = record.get(
            "initial_raw_kv_contiguous_hit_blocks")
        effective_hit = record.get("effective_gdn_hit_blocks")
        observed_tokens = record.get("observed_effective_cached_tokens")
        block_size = record.get("block_size")
        if (
            record.get("gdn_policy") != "fine32"
            or block_size != 16
            or not _integer(raw_hit)
            or not _integer(initial_raw_hit)
            or not _integer(effective_hit)
            or not _integer(observed_tokens)
            or not _integer(block_size, minimum=1)
            or initial_raw_hit > raw_hit
            or effective_hit > initial_raw_hit
            or observed_tokens != effective_hit * block_size
        ):
            reasons.append(f"{label} effective cache accounting differs")
        restore_digest = record.get("gdn_restore_digest_base64")
        if effective_hit == 0:
            if restore_digest is not None:
                reasons.append(f"{label} unexpected GDN restore digest")
        elif _integer(effective_hit, minimum=1):
            try:
                decoded_restore = base64.b64decode(
                    restore_digest, validate=True)
            except (binascii.Error, TypeError, ValueError):
                decoded_restore = b""
            if (
                len(decoded_restore) != 32
                or effective_hit > len(hashes)
                or decoded_restore != hashes[effective_hit - 1]
            ):
                reasons.append(f"{label} GDN restore digest differs")
        case_evidence = evidence.get(label, {})
        if (
            case_evidence.get("prompt_tokens") != record.get("prompt_tokens")
            or case_evidence.get("cached_tokens") != observed_tokens
        ):
            reasons.append(f"{label} HTTP and trace accounting differ")
        summaries[label] = _trace_summary(record, hashes)

    isolation: Json = {}
    if mode == "control" and records:
        if records[0].get("observed_effective_cached_tokens") != 0:
            reasons.append("control one-image request was not cold")
    elif mode == "candidate" and len(records) == len(case_names):
        one, normal, normal_warm, reversed_images, reversed_warm = (
            prompt_hashes
        )
        if normal != normal_warm:
            reasons.append("normal two-image prompt hash chain differs")
        if reversed_images != reversed_warm:
            reasons.append("reversed two-image prompt hash chain differs")
        if normal == reversed_images:
            reasons.append("normal and reversed images share one hash chain")

        normal_prior_common = _common_prefix(one, normal)
        reversed_prior_common = max(
            _common_prefix(reversed_images, prior)
            for prior in (one, normal, normal_warm)
        )
        isolation = {
            "normal_initial_prior_common_blocks": normal_prior_common,
            "reversed_initial_prior_common_blocks": reversed_prior_common,
            "normal_reversed_common_blocks":
                _common_prefix(normal, reversed_images),
        }
        for index, common, label in (
            (1, normal_prior_common, "stream_two_images_cold"),
            (3, reversed_prior_common, "stream_two_images_reversed"),
        ):
            record = records[index]
            raw_hit = record.get(
                "initial_raw_kv_contiguous_hit_blocks")
            effective_hit = record.get("effective_gdn_hit_blocks")
            observed_tokens = record.get("observed_effective_cached_tokens")
            block_size = record.get("block_size")
            if (
                not _integer(raw_hit)
                or not _integer(effective_hit)
                or not _integer(observed_tokens)
                or not _integer(block_size, minimum=1)
                or raw_hit > common
                or effective_hit > common
                or observed_tokens > common * block_size
            ):
                reasons.append(
                    f"{label} crossed the content-hash prefix boundary")
        for index, label in (
            (2, "stream_two_images_warm"),
            (4, "stream_two_images_reversed_warm"),
        ):
            effective_hit = records[index].get("effective_gdn_hit_blocks")
            if not _integer(effective_hit, minimum=1):
                reasons.append(f"{label} has no effective GDN restore")

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "qualified": not reasons,
        "reasons": reasons,
        "mode": mode,
        "trace_version": 4,
        "trace_count": len(records),
        "request_summaries": summaries,
        "content_isolation": isolation,
        "privacy": {
            "contains_raw_tokens": False,
            "contains_raw_images": False,
            "contains_raw_prompt_or_output": False,
            "contains_request_id": False,
            "contains_credentials": False,
        },
        "semantic_quality_evaluated": False,
        "production_promotion_authorized": False,
    }


def _atomic_write(path: Path, value: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("server_log", type=Path)
    parser.add_argument("http_report", type=Path)
    parser.add_argument("--mode", choices=("control", "candidate"),
                        required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    value = qualify(args.server_log, _load(args.http_report), args.mode)
    _atomic_write(args.out, value)
    print(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if value["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
