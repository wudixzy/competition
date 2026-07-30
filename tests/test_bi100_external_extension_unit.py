from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "qwen3_6_scripts" / "bi100_external_extension.py"
SPEC = importlib.util.spec_from_file_location(
    "bi100_external_extension_unit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

PATH_ENV = "BI100_TEST_EXTERNAL_EXTENSION"
SHA_ENV = "BI100_TEST_EXTERNAL_EXTENSION_SHA256"


class Bi100ExternalExtensionTest(unittest.TestCase):

    def _load(self):
        return MODULE.load_hashed_private_extension(
            "bi100_test_external_extension",
            path_environment=PATH_ENV,
            sha256_environment=SHA_ENV,
            required_callable="forward",
        )

    def test_absent_override_uses_default_import_path(self):
        with mock.patch.dict(
                os.environ, {PATH_ENV: "", SHA_ENV: ""}, clear=False):
            self.assertIsNone(self._load())

    def test_private_hash_bound_module_loads(self):
        with tempfile.TemporaryDirectory(
                prefix="bi100-extension-", dir="/tmp") as temporary:
            path = Path(temporary) / "candidate.py"
            path.write_text(
                "def forward():\n    return 'candidate'\n",
                encoding="ascii",
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with mock.patch.dict(os.environ, {
                PATH_ENV: str(path),
                SHA_ENV: digest,
            }, clear=False):
                module = self._load()
            self.assertIsNotNone(module)
            self.assertEqual(module.forward(), "candidate")

    def test_partial_or_mismatched_identity_fails_closed(self):
        with tempfile.TemporaryDirectory(
                prefix="bi100-extension-", dir="/tmp") as temporary:
            path = Path(temporary) / "candidate.py"
            path.write_text("def forward():\n    pass\n", encoding="ascii")
            with mock.patch.dict(os.environ, {
                PATH_ENV: str(path),
                SHA_ENV: "",
            }, clear=False):
                with self.assertRaisesRegex(RuntimeError, "set together"):
                    self._load()
            with mock.patch.dict(os.environ, {
                PATH_ENV: str(path),
                SHA_ENV: "0" * 64,
            }, clear=False):
                with self.assertRaisesRegex(RuntimeError, "mismatch"):
                    self._load()

    def test_group_writable_artifact_fails_closed(self):
        with tempfile.TemporaryDirectory(
                prefix="bi100-extension-", dir="/tmp") as temporary:
            path = Path(temporary) / "candidate.py"
            path.write_text("def forward():\n    pass\n", encoding="ascii")
            path.chmod(0o660)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with mock.patch.dict(os.environ, {
                PATH_ENV: str(path),
                SHA_ENV: digest,
            }, clear=False):
                with self.assertRaisesRegex(
                        RuntimeError, "non-writable file under /tmp"):
                    self._load()


if __name__ == "__main__":
    unittest.main()
