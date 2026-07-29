from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ensure_bi100_runtime_overlay.py"
SPEC = importlib.util.spec_from_file_location("overlay_cache", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class OverlayCacheTest(unittest.TestCase):

    def test_verifier_uses_the_active_python_interpreter(self):
        self.assertEqual(
            Path(MODULE.sys.executable).resolve(),
            Path(sys.executable).resolve(),
        )
        self.assertNotIn(
            '"/usr/bin/python3"',
            SCRIPT.read_text(encoding="ascii"),
        )

    def test_source_identity_is_exact_commit_and_clean_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(
                ["git", "-C", str(root), "config", "user.email", "a@b.c"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "config", "user.name", "test"],
                check=True,
            )
            path = root / "tracked"
            path.write_text("a\n", encoding="ascii")
            subprocess.run(
                ["git", "-C", str(root), "add", "tracked"], check=True)
            subprocess.run(
                ["git", "-C", str(root), "commit", "-qm", "initial"],
                check=True,
            )
            revision, clean = MODULE.source_identity(root)
            self.assertEqual(len(revision), 40)
            self.assertTrue(clean)
            path.write_text("b\n", encoding="ascii")
            same_revision, clean = MODULE.source_identity(root)
            self.assertEqual(same_revision, revision)
            self.assertFalse(clean)


if __name__ == "__main__":
    unittest.main()
