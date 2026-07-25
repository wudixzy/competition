#!/usr/bin/env python3
"""Build a private offline IFEval environment from committed distributions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = ROOT / "quality/external/google_ifeval"
DEFAULT_MANIFEST = EXTERNAL_ROOT / "manifest.v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("schema") != "bi100-ifeval-manifest-v1"
            or value.get("version") != 1):
        raise ValueError("IFEval manifest is invalid")
    return value


def verify_distributions(manifest: dict) -> list[Path]:
    distributions = []
    for item in manifest["offline_environment"]["distribution_artifacts"]:
        path = EXTERNAL_ROOT / "wheelhouse" / item["path"]
        if (not path.is_file() or path.stat().st_size != item["bytes"]
                or sha256(path) != item["sha256"]):
            raise ValueError(f"IFEval distribution identity differs: {path.name}")
        distributions.append(path)
    return distributions


def extract_english_punkt(archive: Path, destination: Path) -> None:
    prefix = "punkt_tab/english/"
    copied = 0
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            if not info.filename.startswith(prefix) or info.is_dir():
                continue
            relative = Path(info.filename).relative_to("punkt_tab")
            if ".." in relative.parts:
                raise ValueError("unsafe NLTK resource path")
            target = destination / "nltk_data/tokenizers/punkt_tab" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(info) as reader, target.open("wb") as writer:
                shutil.copyfileobj(reader, writer)
            copied += 1
    if copied != 4:
        raise ValueError("English punkt_tab resource is incomplete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--punkt-tab-archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("offline IFEval environment requires CPython 3.10")
    if args.target.exists():
        raise FileExistsError(f"target already exists: {args.target}")
    manifest = load_manifest(args.manifest)
    resource = manifest["offline_environment"]["nltk_punkt_tab"]
    if (not args.punkt_tab_archive.is_file()
            or sha256(args.punkt_tab_archive) != resource["archive_sha256"]):
        raise ValueError("pinned punkt_tab archive identity differs")
    distributions = verify_distributions(manifest)

    args.target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{args.target.name}.", dir=args.target.parent))
    try:
        site = staging / "site-packages"
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--no-index",
            "--no-deps", "--no-build-isolation", "--no-cache-dir",
            "--target", str(site),
            *(str(path) for path in distributions),
        ], check=True)
        extract_english_punkt(args.punkt_tab_archive, staging)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join((
            str(site), str(EXTERNAL_ROOT),
        ))
        environment["NLTK_DATA"] = str(staging / "nltk_data")
        subprocess.run([
            sys.executable, "-c",
            "from instruction_following_eval import instructions_util as u; "
            "assert u.count_sentences('One. Two.') == 2; "
            "assert u.count_words('three fixed words') == 3",
        ], check=True, env=environment)
        report = {
            "schema": "bi100-ifeval-offline-environment-v1",
            "version": 1,
            "qualified": True,
            "manifest_sha256": sha256(args.manifest),
            "python": ".".join(map(str, sys.version_info[:3])),
            "system_site_packages_modified": False,
            "punkt_tab_archive_sha256": sha256(args.punkt_tab_archive),
            "distribution_sha256": {
                path.name: sha256(path) for path in distributions
            },
        }
        (staging / "install.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, args.target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({
        "qualified": True,
        "target": str(args.target),
        "install_sha256": sha256(args.target / "install.json"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
