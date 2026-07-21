from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qwen3_6_scripts.patch_xformers_profile import (
    DENSE_NEW,
    DENSE_OLD,
    IMPORT_NEW,
    IMPORT_OLD,
    KV_WRITE_NEW,
    KV_WRITE_OLD,
    PAGED_NEW,
    PAGED_OLD,
    patch_file,
)


class PatchXFormersProfileTest(unittest.TestCase):
    def test_patch_is_exact_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory_text:
            path = Path(directory_text) / "xformers.py"
            path.write_text(
                "\n\n".join((IMPORT_OLD, KV_WRITE_OLD, DENSE_OLD, PAGED_OLD)),
                encoding="utf-8",
            )
            patch_file(path)
            first = path.read_text(encoding="utf-8")
            patch_file(path)
            second = path.read_text(encoding="utf-8")

        self.assertEqual(first, second)
        for expected in (IMPORT_NEW, KV_WRITE_NEW, DENSE_NEW, PAGED_NEW):
            self.assertIn(expected, first)
        for removed in (KV_WRITE_OLD, DENSE_OLD, PAGED_OLD):
            self.assertNotIn(removed, first)

    def test_patch_reconstructs_the_canonical_runtime_file(self):
        canonical_path = (Path(__file__).resolve().parents[1] / "vllm" /
                          "attention" / "backends" / "xformers.py")
        canonical = canonical_path.read_text(encoding="utf-8")
        base = canonical
        for new, old in (
            (IMPORT_NEW, IMPORT_OLD),
            (KV_WRITE_NEW, KV_WRITE_OLD),
            (DENSE_NEW, DENSE_OLD),
            (PAGED_NEW, PAGED_OLD),
        ):
            self.assertEqual(base.count(new), 1)
            base = base.replace(new, old, 1)

        with tempfile.TemporaryDirectory() as directory_text:
            path = Path(directory_text) / "xformers.py"
            path.write_text(base, encoding="utf-8")
            patch_file(path)
            rebuilt = path.read_text(encoding="utf-8")

        self.assertEqual(rebuilt, canonical)


if __name__ == "__main__":
    unittest.main()
