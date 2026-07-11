import importlib
import pathlib
import shlex
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwen3_6_scripts.patch_utils import (ensure_dir, ensure_file,
                                         package_root, replace_once,
                                         replace_one_of, shell_env_line)


class PatchUtilsUnitTest(unittest.TestCase):

    def test_package_root_finds_package_on_sys_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            package = root / "fakepkg"
            package.mkdir()
            (package / "__init__.py").write_text("")
            old_path = list(sys.path)
            sys.path.insert(0, str(root))
            importlib.invalidate_caches()
            try:
                self.assertEqual(package_root("fakepkg"), package.resolve())
            finally:
                sys.path[:] = old_path
                sys.modules.pop("fakepkg", None)

    def test_package_root_missing_package_raises_runtime_error(self):
        with self.assertRaises(RuntimeError):
            package_root("definitely_missing_bi100_package")

    def test_ensure_file_and_dir_validate_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            file_path = root / "target.py"
            dir_path = root / "pkg"
            file_path.write_text("")
            dir_path.mkdir()

            self.assertEqual(ensure_file(file_path), file_path)
            self.assertEqual(ensure_dir(dir_path), dir_path)
            with self.assertRaises(FileNotFoundError):
                ensure_file(dir_path)
            with self.assertRaises(FileNotFoundError):
                ensure_dir(file_path)

    def test_replace_once_patches_once_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "target.py"
            path.write_text("alpha\nalpha\n")
            with redirect_stdout(StringIO()):
                patched = replace_once(path, "alpha", "beta")
            self.assertTrue(patched)
            self.assertEqual(path.read_text(), "beta\nalpha\n")

            with redirect_stdout(StringIO()):
                patched = replace_once(path, "alpha", "beta")
            self.assertFalse(patched)
            self.assertEqual(path.read_text(), "beta\nalpha\n")

    def test_replace_one_of_patches_first_matching_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "target.py"
            path.write_text("left\n")
            with redirect_stdout(StringIO()):
                patched = replace_one_of(path, [
                    ("missing", "bad\n"),
                    ("left\n", "right\n"),
                ])
            self.assertTrue(patched)
            self.assertEqual(path.read_text(), "right\n")

    def test_replace_one_of_respects_already_contains(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "target.py"
            path.write_text("patched marker\n")
            with redirect_stdout(StringIO()):
                patched = replace_one_of(
                    path,
                    [("missing", "new\n")],
                    already_contains="patched marker",
                )
            self.assertFalse(patched)
            self.assertEqual(path.read_text(), "patched marker\n")

    def test_replace_one_of_required_missing_anchor_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "target.py"
            path.write_text("alpha\n")
            with self.assertRaises(RuntimeError):
                with redirect_stdout(StringIO()):
                    replace_one_of(path, [("missing", "new\n")])

    def test_replace_one_of_optional_missing_anchor_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "target.py"
            path.write_text("alpha\n")
            with redirect_stdout(StringIO()):
                patched = replace_one_of(
                    path,
                    [("missing", "new\n")],
                    required=False,
                )
            self.assertFalse(patched)
            self.assertEqual(path.read_text(), "alpha\n")

    def test_shell_env_line_is_shell_safe(self):
        line = shell_env_line("VLLM_ROOT", pathlib.Path("/tmp/corex lib/vllm"))
        self.assertEqual(shlex.split(line), ["VLLM_ROOT=/tmp/corex lib/vllm"])


if __name__ == "__main__":
    unittest.main()
