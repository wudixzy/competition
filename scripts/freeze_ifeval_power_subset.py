#!/usr/bin/env python3
"""Freeze the predeclared 149-pair Google IFEval capability subset."""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = ROOT / "quality/external/google_ifeval"
BASE_MANIFEST = EXTERNAL_ROOT / "manifest.v1.json"
BASE_MANIFEST_SHA256 = (
    "07ec4efb5fe7afaacb55723c1d53be4c2f58c840bbd6a54bf944e15cfbca1855"
)
SOURCE_PATH = EXTERNAL_ROOT / "source/ifeval_input_data.jsonl"
DEFAULT_SUBSET = EXTERNAL_ROOT / "subset.power149.v2.jsonl"
DEFAULT_MANIFEST = EXTERNAL_ROOT / "manifest.power149.v2.json"
SEED = "bi100-ifeval-power149-v2-seed-20260730"
SUBSET_SIZE = 149
MIN_INSTRUCTION_COVERAGE = 10
CONFIDENCE = 0.95
NONINFERIORITY_MARGIN = 0.02
BOOTSTRAP_SAMPLES = 20000
BOOTSTRAP_SEED = 20260729
Json = dict[str, Any]


def _load_base_module():
    path = ROOT / "scripts/freeze_ifeval_subset.py"
    specification = importlib.util.spec_from_file_location(
        "freeze_ifeval_subset_base", path)
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load the pinned IFEval source validator")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


BASE = _load_base_module()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selection_rank(row: Json) -> str:
    row_sha = hashlib.sha256(BASE.canonical_bytes(row)).hexdigest()
    return hashlib.sha256(f"{SEED}\0{row_sha}".encode("ascii")).hexdigest()


def select_rows(rows: list[Json]) -> list[Json]:
    counts: collections.Counter[str] = collections.Counter()
    remaining = list(enumerate(rows))
    selected: list[tuple[int, Json]] = []

    while any(
        counts[item] < MIN_INSTRUCTION_COVERAGE
        for item in BASE.EXPECTED_INSTRUCTION_IDS
    ):
        def candidate_rank(candidate: tuple[int, Json]) -> tuple[Any, ...]:
            index, row = candidate
            unique_ids = set(row["instruction_id_list"])
            gain = sum(
                counts[item] < MIN_INSTRUCTION_COVERAGE
                for item in unique_ids)
            deficit = sum(
                max(MIN_INSTRUCTION_COVERAGE - counts[item], 0)
                for item in unique_ids)
            return (-gain, -deficit, selection_rank(row), index)

        winner = min(remaining, key=candidate_rank)
        if candidate_rank(winner)[0] == 0:
            raise ValueError("unable to satisfy IFEval instruction coverage")
        remaining.remove(winner)
        selected.append(winner)
        counts.update(winner[1]["instruction_id_list"])
        if len(selected) > SUBSET_SIZE:
            raise ValueError("IFEval coverage exceeds power subset size")

    for candidate in sorted(
        remaining, key=lambda item: (selection_rank(item[1]), item[0])
    ):
        if len(selected) == SUBSET_SIZE:
            break
        selected.append(candidate)
        counts.update(candidate[1]["instruction_id_list"])

    if (
        len(selected) != SUBSET_SIZE
        or set(counts) != BASE.EXPECTED_INSTRUCTION_IDS
        or min(counts.values()) < MIN_INSTRUCTION_COVERAGE
    ):
        raise ValueError("IFEval power subset coverage is invalid")
    return [row for _, row in sorted(selected)]


def write_subset(path: Path, rows: list[Json]) -> None:
    payload = b"".join(BASE.canonical_bytes(row) + b"\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def build_manifest(subset: Path, rows: list[Json]) -> Json:
    if sha256(BASE_MANIFEST) != BASE_MANIFEST_SHA256:
        raise ValueError("base IFEval manifest identity differs")
    base = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    counts = collections.Counter(
        item for row in rows for item in row["instruction_id_list"])
    shape_counts = collections.Counter(
        len(row["instruction_id_list"]) for row in rows)
    manifest = copy.deepcopy(base)
    manifest.update({
        "schema": "bi100-ifeval-manifest-v2",
        "version": 2,
        "name": "google-ifeval-bi100-power149-v2",
        "derived_from": {
            "manifest_path": str(BASE_MANIFEST.relative_to(ROOT)),
            "manifest_sha256": BASE_MANIFEST_SHA256,
            "source_snapshot_unchanged": True,
            "evaluator_snapshot_unchanged": True,
        },
    })
    manifest["selection"] = {
        "algorithm": (
            "greedy maximum uncovered instruction IDs, then maximum "
            "remaining deficit, then ascending SHA-256(seed,row); fill "
            "by ascending SHA-256 and emit in source order"
        ),
        "seed": SEED,
        "size": SUBSET_SIZE,
        "minimum_per_instruction_id": MIN_INSTRUCTION_COVERAGE,
        "instruction_id_count": len(counts),
        "instruction_counts": dict(sorted(counts.items())),
        "instruction_arity_counts": {
            str(key): value for key, value in sorted(shape_counts.items())
        },
        "selected_keys_in_request_order": [row["key"] for row in rows],
        "frozen_before_candidate_observation": True,
    }
    manifest["subset"] = {
        "repository_path": str(subset.relative_to(ROOT)),
        "bytes": subset.stat().st_size,
        "sha256": sha256(subset),
        "rows": len(rows),
        "stable_order": True,
    }
    manifest["evaluator"]["dataset_difference_from_evaluator_repo"][
        "selected"
    ] = any(row["key"] == 2785 for row in rows)
    manifest["scoring"] = {
        "response_field": "choices[0].message.content",
        "strict": "official test_instruction_following_strict",
        "loose": "official test_instruction_following_loose",
        "comparison_unit": "paired prompt",
        "candidate_rule": (
            "one-sided 95% paired noninferiority for strict and loose "
            "prompt pass rates at a predeclared 0.02 margin"
        ),
        "exact_output_role": "diagnostic_only_across_arms",
        "overall_promotion_authorized": False,
    }
    manifest["statistical_contract"] = {
        "confidence": CONFIDENCE,
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "minimum_zero_regression_pairs": SUBSET_SIZE,
        "sample_count": SUBSET_SIZE,
        "sample_selection_after_results_forbidden": True,
        "margin_change_after_results_forbidden": True,
    }
    script = Path(__file__).resolve()
    manifest["generator"] = {
        "path": str(script.relative_to(ROOT)),
        "sha256": sha256(script),
    }
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.out.exists() or args.manifest.exists():
        raise FileExistsError("IFEval power snapshot already exists")
    rows = select_rows(BASE.load_source(args.source))
    write_subset(args.out, rows)
    manifest = build_manifest(args.out, rows)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "qualified": True,
        "subset": str(args.out),
        "subset_sha256": sha256(args.out),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
