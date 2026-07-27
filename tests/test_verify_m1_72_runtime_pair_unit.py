from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

SPEC = importlib.util.spec_from_file_location(
    "verify_m1_72_runtime_pair",
    TESTS / "verify_m1_72_runtime_pair.py",
)
VERIFY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VERIFY)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def build_overlay(
    root: Path,
    label: str,
    revision: str,
) -> tuple[Path, dict]:
    site = root / label / "site-packages"
    site.mkdir(parents=True)
    files = {}
    for name in sorted(VERIFY.REQUIRED_FILES):
        payload = (
            f"{name}-{label}".encode("ascii")
            if name in VERIFY.ALLOWED_RUNTIME_FILE_DELTA
            else f"{name}-common".encode("ascii")
        )
        installed = site / f"{name}.artifact"
        installed.write_bytes(payload)
        digest = sha256(payload)
        files[name] = {
            "same": True,
            "source_sha256": digest,
            "installed_sha256": digest,
            "installed_path": str(installed),
        }
    install = {
        "schema": VERIFY.INSTALL_SCHEMA,
        "version": 2,
        "qualified": True,
        "source_tree_clean": True,
        "system_site_packages_modified": False,
        "source_revision": revision,
        "site_packages": str(site),
        "runtime_tree_sha256": VERIFY.runtime_tree_sha256(site),
        "versions": {"transformers": "4.55.3"},
        "files": files,
    }
    return site, install


class VerifyM172RuntimePairTest(unittest.TestCase):

    def test_exact_three_file_delta_qualifies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            control_site, control = build_overlay(
                root, "control", "c" * 40)
            candidate_site, candidate = build_overlay(
                root, "candidate", "d" * 40)
            report = VERIFY.verify(
                control_site,
                control,
                "c" * 40,
                candidate_site,
                candidate,
                "d" * 40,
            )
            self.assertTrue(report["qualified"], report)
            self.assertEqual(
                set(report["observed_runtime_file_delta"]),
                VERIFY.ALLOWED_RUNTIME_FILE_DELTA,
            )

    def test_unexpected_runtime_delta_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            control_site, control = build_overlay(
                root, "control", "c" * 40)
            candidate_site, candidate = build_overlay(
                root, "candidate", "d" * 40)
            changed = copy.deepcopy(candidate)
            name = "scheduler"
            path = Path(changed["files"][name]["installed_path"])
            payload = b"unexpected-candidate-scheduler"
            path.write_bytes(payload)
            digest = sha256(payload)
            changed["files"][name]["source_sha256"] = digest
            changed["files"][name]["installed_sha256"] = digest
            changed["runtime_tree_sha256"] = VERIFY.runtime_tree_sha256(
                candidate_site)
            report = VERIFY.verify(
                control_site,
                control,
                "c" * 40,
                candidate_site,
                changed,
                "d" * 40,
            )
            self.assertFalse(report["qualified"])
            self.assertTrue(any(
                "runtime file delta differs" in reason
                for reason in report["reasons"]))

    def test_tampered_overlay_fails_tree_and_file_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            control_site, control = build_overlay(
                root, "control", "c" * 40)
            candidate_site, candidate = build_overlay(
                root, "candidate", "d" * 40)
            Path(candidate["files"]["protocol"]["installed_path"]).write_bytes(
                b"tampered")
            report = VERIFY.verify(
                control_site,
                control,
                "c" * 40,
                candidate_site,
                candidate,
                "d" * 40,
            )
            self.assertFalse(report["qualified"])
            self.assertTrue(any(
                "runtime tree identity mismatch" in reason
                for reason in report["reasons"]))
            self.assertTrue(any(
                "runtime file mismatch: protocol" in reason
                for reason in report["reasons"]))


if __name__ == "__main__":
    unittest.main()
