#!/usr/bin/env python3
"""Attest GDN scheduler actions across installed model-input broadcasts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
from typing import Any


SCHEMA = "bi100-gdn-action-broadcast-gate-v1"
VERSION = 1
RANK_COUNT = 4
RESTORE_KEY = (64, hashlib.sha256(b"restore").digest())
CAPTURE_POINTS = [
    (16, (65, hashlib.sha256(b"capture-1").digest())),
    (32, (66, hashlib.sha256(b"capture-2").digest())),
]
EVICT_KEYS = [(32, hashlib.sha256(b"evict").digest())]
SEGMENT_OFFSETS = [48, 64, 8176, 8192]
ACTION_FIELDS = (
    "gdn_restore_key", "gdn_capture_points", "gdn_evict_keys",
    "gdn_segment_offsets",
)


def atomic_write(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                     dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _actions(value: Any) -> tuple[Any, Any, Any]:
    return tuple(getattr(value, field) for field in ACTION_FIELDS)


def evaluate_class(name: str, model_input_cls: type) -> dict[str, Any]:
    source = model_input_cls(
        gdn_restore_key=RESTORE_KEY,
        gdn_capture_points=copy.deepcopy(CAPTURE_POINTS),
        gdn_evict_keys=copy.deepcopy(EVICT_KEYS),
        gdn_segment_offsets=copy.deepcopy(SEGMENT_OFFSETS),
    )
    payload = source.as_broadcastable_tensor_dict()
    fields_present = all(payload.get(field) == getattr(source, field)
                         for field in ACTION_FIELDS)
    reconstructed = [
        model_input_cls.from_broadcasted_tensor_dict(copy.deepcopy(payload))
        for _ in range(RANK_COUNT)
    ]
    actions_match = all(_actions(rank_input) == _actions(source)
                        for rank_input in reconstructed)
    capture_containers_independent = len({
        id(rank_input.gdn_capture_points) for rank_input in reconstructed
    }) == RANK_COUNT
    evict_containers_independent = len({
        id(rank_input.gdn_evict_keys) for rank_input in reconstructed
    }) == RANK_COUNT
    segment_containers_independent = len({
        id(rank_input.gdn_segment_offsets) for rank_input in reconstructed
    }) == RANK_COUNT
    return {
        "class": name,
        "fields_present": fields_present,
        "rank_reconstruction_count": len(reconstructed),
        "actions_match": actions_match,
        "capture_containers_independent": capture_containers_independent,
        "evict_containers_independent": evict_containers_independent,
        "segment_containers_independent": segment_containers_independent,
        "restore_blocks": RESTORE_KEY[0],
        "restore_digest_sha256": RESTORE_KEY[1].hex(),
        "capture_count": len(CAPTURE_POINTS),
        "eviction_count": len(EVICT_KEYS),
        "segment_count": len(SEGMENT_OFFSETS),
    }


def build_report(base_cls: type, sampling_cls: type,
                 model_source: Path) -> dict[str, Any]:
    reasons: list[str] = []
    cases: list[dict[str, Any]] = []
    for name, model_input_cls in (
            ("ModelInputForGPU", base_cls),
            ("ModelInputForGPUWithSamplingMetadata", sampling_cls)):
        try:
            case = evaluate_class(name, model_input_cls)
            cases.append(case)
            for field in (
                    "fields_present", "actions_match",
                    "capture_containers_independent",
                    "evict_containers_independent",
                    "segment_containers_independent"):
                if case[field] is not True:
                    reasons.append(f"{name}: {field} is not true")
            if case["rank_reconstruction_count"] != RANK_COUNT:
                reasons.append(f"{name}: rank reconstruction count differs")
        except Exception as error:
            reasons.append(f"{name}: {type(error).__name__}: {error}")

    fail_fast_attested = False
    model_source_sha256 = None
    try:
        source = model_source.read_text(encoding="utf-8")
        model_source_sha256 = hashlib.sha256(
            model_source.read_bytes()).hexdigest()
        fail_fast_attested = all(marker in source for marker in (
            "saved_state = self._gdn_prefix_cache.get(restore_key)",
            "if saved_state is None:",
            "scheduler requested a missing GDN prefix state",
        ))
        if not fail_fast_attested:
            reasons.append("model source lacks missing-restore fail-fast")
    except Exception as error:
        reasons.append(f"model-source: {type(error).__name__}: {error}")

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "rank_count": RANK_COUNT,
        "cases": cases,
        "missing_restore_fail_fast_source_attested": fail_fast_attested,
        "model_source_sha256": model_source_sha256,
        "qualified": not reasons and len(cases) == 2,
        "reasons": reasons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gate installed GDN action broadcast metadata")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from vllm.worker.model_runner import (
        ModelInputForGPU, ModelInputForGPUWithSamplingMetadata,
    )

    model_runner_source = Path(inspect.getfile(ModelInputForGPU)).resolve()
    model_source = (
        model_runner_source.parents[1]
        / "model_executor" / "models" / "qwen3_5.py"
    )
    report = build_report(
        ModelInputForGPU, ModelInputForGPUWithSamplingMetadata, model_source)
    atomic_write(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
