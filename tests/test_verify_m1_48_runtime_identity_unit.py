from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from tests.verify_m1_48_runtime_identity import SOURCE_FILES, verify


class VerifyM148RuntimeIdentityTest(unittest.TestCase):
    @staticmethod
    def _write_worker(runtime: Path) -> str:
        worker = runtime / "vllm" / "worker" / "worker.py"
        worker.parent.mkdir(parents=True, exist_ok=True)
        worker.write_text(
            "# Mark this synthetic pass so BI100_PROFILE can exclude\n"
            "os.environ[\"BI100_IN_STARTUP_PROFILE\"] = \"1\"\n",
            encoding="utf-8",
        )
        return hashlib.sha256(worker.read_bytes()).hexdigest()

    def test_current_install_and_runtime_hashes_must_match(self):
        with tempfile.TemporaryDirectory() as directory_text:
            root = Path(directory_text)
            runtime = root / "runtime" / "site-packages"
            files = {}
            for name, relative in SOURCE_FILES.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = f"{name}\n".encode()
                path.write_bytes(payload)
                installed_path = runtime / relative
                installed_path.parent.mkdir(parents=True, exist_ok=True)
                installed_path.write_bytes(payload)
                digest = hashlib.sha256(payload).hexdigest()
                files[name] = {
                    "same": True,
                    "source_sha256": digest,
                    "installed_sha256": digest,
                    "installed_path": str(installed_path),
                }
            install = {
                "schema": "bi100-bare-host-runtime-install-v2",
                "version": 2,
                "qualified": True,
                "site_packages": str(runtime),
                "startup_profile_guard_patch": True,
                "worker_sha256": self._write_worker(runtime),
                "files": files,
            }
            report = verify(root, runtime, install, "b" * 40)

        self.assertTrue(report["qualified"], report)
        self.assertEqual(report["source_revision"], "b" * 40)

    def test_stale_overlay_fails(self):
        with tempfile.TemporaryDirectory() as directory_text:
            root = Path(directory_text)
            runtime = root / "runtime" / "site-packages"
            files = {}
            for name, relative in SOURCE_FILES.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name, encoding="utf-8")
                installed_path = runtime / relative
                installed_path.parent.mkdir(parents=True, exist_ok=True)
                installed_path.write_text("stale", encoding="utf-8")
                files[name] = {
                    "same": True,
                    "source_sha256": "a" * 64,
                    "installed_sha256": "a" * 64,
                    "installed_path": str(installed_path),
                }
            install = {
                "schema": "bi100-bare-host-runtime-install-v2",
                "version": 2,
                "qualified": True,
                "site_packages": str(runtime),
                "startup_profile_guard_patch": True,
                "worker_sha256": self._write_worker(runtime),
                "files": files,
            }
            report = verify(root, runtime, install, "b" * 40)

        self.assertFalse(report["qualified"])
        self.assertEqual(len(report["reasons"]), len(SOURCE_FILES))

    def test_runtime_mutation_fails_even_when_report_hashes_match(self):
        with tempfile.TemporaryDirectory() as directory_text:
            root = Path(directory_text)
            runtime = root / "runtime" / "site-packages"
            files = {}
            for name, relative in SOURCE_FILES.items():
                source = root / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(name, encoding="utf-8")
                digest = hashlib.sha256(name.encode()).hexdigest()
                installed_path = runtime / relative
                installed_path.parent.mkdir(parents=True, exist_ok=True)
                installed_path.write_text(name, encoding="utf-8")
                files[name] = {
                    "same": True,
                    "source_sha256": digest,
                    "installed_sha256": digest,
                    "installed_path": str(installed_path),
                }
            first_installed = Path(next(iter(files.values()))["installed_path"])
            first_installed.write_text("mutated", encoding="utf-8")
            install = {
                "schema": "bi100-bare-host-runtime-install-v2",
                "version": 2,
                "qualified": True,
                "site_packages": str(runtime),
                "startup_profile_guard_patch": True,
                "worker_sha256": self._write_worker(runtime),
                "files": files,
            }
            report = verify(root, runtime, install, "b" * 40)

        self.assertFalse(report["qualified"])
        self.assertEqual(len(report["reasons"]), 1)


if __name__ == "__main__":
    unittest.main()
