#!/usr/bin/env python3
"""Make pip's local-wheel provenance metadata reproducible."""

from __future__ import annotations

import argparse
import base64
import csv
from email.parser import BytesParser
from email.policy import default
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.parse import quote


def _canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_hash(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
    return "sha256=" + encoded.rstrip(b"=").decode("ascii")


def _distribution_identity(dist_info: Path) -> tuple[str, str]:
    metadata_path = dist_info / "METADATA"
    if not metadata_path.is_file():
        raise ValueError(f"distribution metadata is missing: {metadata_path}")
    metadata = BytesParser(policy=default).parsebytes(
        metadata_path.read_bytes())
    name = metadata.get("Name")
    version = metadata.get("Version")
    if not name or not version:
        raise ValueError(
            f"distribution Name or Version is missing: {metadata_path}")
    return _canonical_name(str(name)), str(version)


def _find_distribution(
    site_packages: Path,
    distribution: str,
    version: str,
) -> Path:
    expected_name = _canonical_name(distribution)
    matches = []
    for candidate in sorted(site_packages.glob("*.dist-info")):
        if _distribution_identity(candidate) == (expected_name, version):
            matches.append(candidate)
    if len(matches) != 1:
        rendered = ", ".join(path.name for path in matches) or "none"
        raise ValueError(
            "expected exactly one installed distribution "
            f"{distribution}=={version}; found {len(matches)}: {rendered}")
    return matches[0]


def normalize_distribution(
    *,
    site_packages: Path,
    distribution: str,
    version: str,
    wheel: Path,
) -> dict[str, str]:
    site_packages = site_packages.resolve()
    wheel = wheel.resolve()
    if not site_packages.is_dir():
        raise ValueError(
            f"site-packages directory does not exist: {site_packages}")
    if not wheel.is_file():
        raise ValueError(f"offline wheel does not exist: {wheel}")

    dist_info = _find_distribution(site_packages, distribution, version)
    direct_url_path = dist_info / "direct_url.json"
    record_path = dist_info / "RECORD"
    if not direct_url_path.is_file():
        raise ValueError(f"direct URL metadata is missing: {direct_url_path}")
    if not record_path.is_file():
        raise ValueError(f"distribution RECORD is missing: {record_path}")

    wheel_sha256 = _sha256(wheel)
    direct_url = {
        "archive_info": {
            "hash": f"sha256={wheel_sha256}",
            "hashes": {"sha256": wheel_sha256},
        },
        "url": f"file:///offline/{quote(wheel.name)}",
    }
    direct_url_bytes = (
        json.dumps(direct_url, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    record_rows = list(csv.reader(
        io.StringIO(record_path.read_text(encoding="utf-8"), newline="")))
    direct_url_relative = direct_url_path.relative_to(
        site_packages).as_posix()
    matching_rows = [
        row for row in record_rows if row and row[0] == direct_url_relative
    ]
    if len(matching_rows) != 1:
        raise ValueError(
            "expected exactly one direct_url.json row in RECORD; "
            f"found {len(matching_rows)}")
    row = matching_rows[0]
    if len(row) != 3:
        raise ValueError(
            f"invalid RECORD row for {direct_url_relative}: {row!r}")
    row[1] = _record_hash(direct_url_bytes)
    row[2] = str(len(direct_url_bytes))

    rendered_record = io.StringIO(newline="")
    writer = csv.writer(rendered_record, lineterminator="\n")
    writer.writerows(record_rows)
    direct_url_path.write_bytes(direct_url_bytes)
    record_path.write_text(
        rendered_record.getvalue(), encoding="utf-8", newline="\n")

    normalized_text = direct_url_path.read_text(encoding="utf-8")
    if str(wheel.parent) in normalized_text or "/tmp/" in normalized_text:
        raise ValueError("temporary wheel staging path remains in metadata")
    return {
        "distribution": distribution,
        "version": version,
        "wheel_sha256": wheel_sha256,
        "direct_url_sha256": _sha256(direct_url_path),
        "record_sha256": _sha256(record_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-packages", required=True, type=Path)
    parser.add_argument("--distribution", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--wheel", required=True, type=Path)
    args = parser.parse_args()
    report = normalize_distribution(
        site_packages=args.site_packages,
        distribution=args.distribution,
        version=args.version,
        wheel=args.wheel,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
