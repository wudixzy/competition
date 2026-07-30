#!/usr/bin/env python3
"""Parse candidate-only ixprof summaries into phase-attribution evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import profile_m1_155_fused_prefill_phase as workload


SCHEMA = "bi100-m1-155-ixprof-phase-attribution-v1"
ROW = re.compile(
    r"^\s*(?:GPU activities:)?\s*"
    r"(?P<percent>[0-9]+(?:\.[0-9]+)?)%\s+"
    r"(?P<time>[0-9]+(?:\.[0-9]+)?)"
    r"(?P<unit>ns|us|ms|s)\s+"
    r"(?P<calls>[0-9]+)\s+"
    r"\S+\s+\S+\s+\S+\s+"
    r"(?P<name>.+?)\s*$"
)


def _milliseconds(value: str, unit: str) -> float:
    scales = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1000.0}
    return float(value) * scales[unit]


def parse_gpu_rows(lines: Iterable[str]) -> list[dict[str, Any]]:
    rows = []
    in_gpu_summary = False
    for line in lines:
        if "GPU activities:" in line:
            in_gpu_summary = True
        if in_gpu_summary and "API Calls:" in line:
            break
        if not in_gpu_summary:
            continue
        match = ROW.match(line)
        if match is None:
            continue
        rows.append({
            "percent": float(match.group("percent")),
            "time_ms": _milliseconds(
                match.group("time"), match.group("unit")),
            "calls": int(match.group("calls")),
            "name": match.group("name"),
        })
    return rows


def classify(name: str) -> str | None:
    if "convert_query_kernel" in name:
        return "convert_query"
    if "gather_kv_group_kernel" in name:
        return "gather"
    if "mask_group_scores_kernel" in name:
        return "mask"
    if "normalize_split_scores_kernel" in name:
        return "normalize"
    if "merge_split_output_kernel" in name:
        return "merge"
    if "Gemm_tcu_bi_kernel" in name and "true, false>" in name:
        return "qk"
    if "Gemm_tcu_bi_kernel" in name and "false, false>" in name:
        return "pv"
    return None


def qualify(
    cell: Any,
    profile_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    reasons: list[str] = []
    if not isinstance(cell, dict):
        return {"qualified": False, "reasons": ["cell must be an object"]}
    cell_evaluation = workload.evaluate(cell)
    if not cell_evaluation["qualified"]:
        reasons.append("profile workload cell did not qualify")
    if not profile_rows:
        reasons.append("ixprof GPU summary is missing")

    phase_names = (
        "convert_query", "gather", "qk", "mask", "normalize", "pv",
        "merge",
    )
    phases = {
        phase: {"time_ms": 0.0, "calls": 0, "percent": 0.0}
        for phase in phase_names
    }
    phase_time_scale = cell.get("phase_time_scale")
    if (
        not isinstance(phase_time_scale, (int, float))
        or isinstance(phase_time_scale, bool)
        or not math.isfinite(float(phase_time_scale))
        or not 0.0 < float(phase_time_scale) <= 1.0
    ):
        reasons.append("phase time scale is invalid")
        phase_time_scale = 1.0
    process_total_ms = 0.0
    unclassified_process_ms = 0.0
    for row in profile_rows:
        phase = classify(str(row["name"]))
        row_ms = float(row["time_ms"])
        process_total_ms += row_ms
        if phase is None:
            unclassified_process_ms += row_ms
            continue
        phases[phase]["time_ms"] += (
            float(row["time_ms"]) * float(phase_time_scale))
        phases[phase]["calls"] += int(row["calls"])
    attributed_ms = sum(value["time_ms"] for value in phases.values())

    expected = cell.get("expected_launches")
    if isinstance(expected, dict):
        for phase in (
            "convert_query", "gather", "qk", "mask", "normalize", "pv",
            "merge",
        ):
            if phases[phase]["calls"] != expected.get(phase):
                reasons.append(f"{phase} launch count differs")
    else:
        reasons.append("expected launch counts are missing")

    event_ms = cell.get("profile_cuda_ms")
    attributed_ratio = (
        attributed_ms / float(event_ms)
        if (
            isinstance(event_ms, (int, float))
            and not isinstance(event_ms, bool)
            and math.isfinite(float(event_ms))
            and float(event_ms) > 0.0
        )
        else None
    )
    if (
        attributed_ratio is None
        or attributed_ratio < 0.75
        or attributed_ratio > 1.05
    ):
        reasons.append("candidate phase attribution coverage differs")
    candidate_unattributed_ms = (
        max(0.0, float(event_ms) - attributed_ms)
        if attributed_ratio is not None else None
    )
    if attributed_ratio is not None:
        for value in phases.values():
            value["percent"] = (
                value["time_ms"] / float(event_ms) * 100.0)
    phases["candidate_unattributed"] = {
        "time_ms": candidate_unattributed_ms,
        "calls": None,
        "percent": (
            candidate_unattributed_ms / float(event_ms) * 100.0
            if candidate_unattributed_ms is not None else None
        ),
    }

    qualified = not reasons
    return {
        "schema": SCHEMA,
        "version": 1,
        "qualified": qualified,
        "reasons": reasons,
        "case": cell.get("case"),
        "source_revision": cell.get("source_revision"),
        "instance": cell.get("instance"),
        "visible_physical_gpu": cell.get("visible_physical_gpu"),
        "extension_sha256": (
            (cell.get("extension") or {}).get("sha256")
            if isinstance(cell.get("extension"), dict) else None
        ),
        "profile_trials": cell.get("profile_trials"),
        "warmup_trials": cell.get("warmup_trials"),
        "ixprof_candidate_trials": cell.get("ixprof_candidate_trials"),
        "phase_time_scale": cell.get("phase_time_scale"),
        "profile_event_time_ms": event_ms,
        "attributed_candidate_time_ms": attributed_ms,
        "attributed_candidate_ratio": attributed_ratio,
        "process_gpu_activity_time_ms": process_total_ms,
        "unclassified_process_gpu_activity_time_ms": (
            unclassified_process_ms),
        "phases": phases,
        "raw_kernel_rows": len(profile_rows),
        "authorization": {
            "implementation_direction_authorized": qualified,
            "tp4_service_authorized": False,
            "main_or_yaml_change_authorized": False,
            "official_score_claim_authorized": False,
        },
        "privacy": {
            "raw_tensors_recorded": False,
            "model_outputs_recorded": False,
            "prompts_recorded": False,
            "credentials_recorded": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True, type=Path)
    parser.add_argument(
        "--profile-log", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cell = json.loads(args.cell.read_text(encoding="ascii"))
    lines = []
    for path in args.profile_log:
        lines.extend(path.read_text(
            encoding="utf-8", errors="replace").splitlines())
    result = qualify(cell, parse_gpu_rows(lines))
    args.output.write_text(
        json.dumps(
            result,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="ascii",
    )
    print(json.dumps({
        "case": result["case"],
        "qualified": result["qualified"],
        "phases": {
            name: round(value["percent"], 3)
            for name, value in result["phases"].items()
        },
        "reasons": result["reasons"],
    }, sort_keys=True))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
