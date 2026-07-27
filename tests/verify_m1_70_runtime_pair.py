#!/usr/bin/env python3
"""Verify immutable baseline/candidate overlays for the M1-70 HTTP A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from verify_bare_host_runtime_identity import (
    DIRECT_SOURCE_FILES,
    INSTALL_SCHEMA,
    REQUIRED_FILES,
    runtime_tree_sha256,
)


Json = dict[str, Any]
SCHEMA = "bi100-m1-70-runtime-pair-v1"
VERSION = 1
ALLOWED_RUNTIME_FILE_DELTA = {"api_server", "protocol"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_overlay(
    label: str,
    site: Path,
    install: Json,
    expected_revision: str,
    reasons: list[str],
) -> dict[str, Any]:
    site = site.resolve()
    if (install.get("schema") != INSTALL_SCHEMA
            or install.get("version") != 2
            or install.get("qualified") is not True
            or install.get("source_tree_clean") is not True
            or install.get("system_site_packages_modified") is not False):
        reasons.append(f"{label} install report is not qualified")
    if install.get("source_revision") != expected_revision:
        reasons.append(f"{label} source revision mismatch")
    reported_site = install.get("site_packages")
    if (not isinstance(reported_site, str)
            or Path(reported_site).resolve() != site):
        reasons.append(f"{label} active site differs from install report")
    actual_tree = runtime_tree_sha256(site) if site.is_dir() else None
    if install.get("runtime_tree_sha256") != actual_tree:
        reasons.append(f"{label} runtime tree identity mismatch")

    files = install.get("files")
    if not isinstance(files, dict):
        files = {}
        reasons.append(f"{label} file identities are missing")
    if not REQUIRED_FILES.issubset(files):
        reasons.append(f"{label} required runtime files are missing")

    file_sha256: dict[str, str | None] = {}
    for name in sorted(REQUIRED_FILES):
        row = files.get(name)
        if not isinstance(row, dict):
            file_sha256[name] = None
            continue
        installed_path = row.get("installed_path")
        path = Path(installed_path).resolve() \
            if isinstance(installed_path, str) else None
        actual = (
            _sha256(path)
            if path is not None
            and path.is_relative_to(site)
            and path.is_file()
            else None
        )
        reported = row.get("installed_sha256")
        source = row.get("source_sha256")
        if (row.get("same") is not True
                or actual is None
                or actual != reported
                or source != reported):
            reasons.append(f"{label} runtime file mismatch: {name}")
        file_sha256[name] = actual
    return {
        "source_revision": install.get("source_revision"),
        "site_packages": str(site),
        "runtime_tree_sha256": actual_tree,
        "file_sha256": file_sha256,
    }


def verify(
    source_root: Path,
    control_site: Path,
    control_install: Json,
    control_revision: str,
    candidate_site: Path,
    candidate_install: Json,
    candidate_revision: str,
) -> Json:
    reasons: list[str] = []
    control = _validate_overlay(
        "control",
        control_site,
        control_install,
        control_revision,
        reasons,
    )
    candidate = _validate_overlay(
        "candidate",
        candidate_site,
        candidate_install,
        candidate_revision,
        reasons,
    )

    control_files = control["file_sha256"]
    candidate_files = candidate["file_sha256"]
    observed_delta = {
        name
        for name in REQUIRED_FILES
        if control_files.get(name) != candidate_files.get(name)
    }
    if observed_delta != ALLOWED_RUNTIME_FILE_DELTA:
        reasons.append(
            "runtime file delta differs from api_server/protocol: "
            + ",".join(sorted(observed_delta)))

    source_root = source_root.resolve()
    current_candidate_match: dict[str, bool] = {}
    for name, relative in DIRECT_SOURCE_FILES.items():
        source_path = source_root / relative
        current_sha = _sha256(source_path) if source_path.is_file() else None
        same = current_sha == candidate_files.get(name)
        current_candidate_match[name] = same
        if not same:
            reasons.append(
                f"current source differs from candidate runtime: {name}")

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "qualified": not reasons,
        "reasons": reasons,
        "control": control,
        "candidate": candidate,
        "observed_runtime_file_delta": sorted(observed_delta),
        "allowed_runtime_file_delta": sorted(ALLOWED_RUNTIME_FILE_DELTA),
        "current_candidate_match": current_candidate_match,
        "privacy": {
            "contains_environment": False,
            "contains_credentials": False,
        },
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--control-site", type=Path, required=True)
    parser.add_argument("--control-install", type=Path, required=True)
    parser.add_argument("--control-revision", required=True)
    parser.add_argument("--candidate-site", type=Path, required=True)
    parser.add_argument("--candidate-install", type=Path, required=True)
    parser.add_argument("--candidate-revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = verify(
        args.source_root,
        args.control_site,
        json.loads(args.control_install.read_text(encoding="utf-8")),
        args.control_revision,
        args.candidate_site,
        json.loads(args.candidate_install.read_text(encoding="utf-8")),
        args.candidate_revision,
    )
    _atomic_write(args.out, report)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
