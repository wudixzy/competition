from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/prepare_ifeval_env.py"
SPEC = importlib.util.spec_from_file_location("prepare_ifeval_env", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PrepareIFEvalEnvironmentTest(unittest.TestCase):

    def test_manifest_and_all_distributions_validate(self):
        manifest = MODULE.load_manifest(MODULE.DEFAULT_MANIFEST)
        distributions = MODULE.verify_distributions(manifest)
        self.assertEqual(len(distributions), 10)
        self.assertTrue(all(path.is_file() for path in distributions))

    def test_power149_manifest_uses_same_pinned_distributions(self):
        manifest = MODULE.load_manifest(
            MODULE.EXTERNAL_ROOT / "manifest.power149.v2.json")
        distributions = MODULE.verify_distributions(manifest)
        self.assertEqual(manifest["selection"]["size"], 149)
        self.assertEqual(len(distributions), 10)
        self.assertTrue(all(path.is_file() for path in distributions))

    def test_extracts_only_four_english_punkt_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "punkt.zip"
            with zipfile.ZipFile(archive, "w") as output:
                for name in (
                        "collocations.tab", "sent_starters.txt",
                        "abbrev_types.txt", "ortho_context.tab"):
                    output.writestr(f"punkt_tab/english/{name}", name)
                output.writestr("punkt_tab/french/ignored.txt", "ignored")
            MODULE.extract_english_punkt(archive, root / "out")
            files = sorted(path.name for path in (
                root / "out/nltk_data/tokenizers/punkt_tab/english"
            ).iterdir())
            self.assertEqual(files, [
                "abbrev_types.txt", "collocations.tab",
                "ortho_context.tab", "sent_starters.txt",
            ])

    def test_rejects_unsafe_resource_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "punkt.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr(
                    "punkt_tab/english/../../outside.txt", "unsafe")
            with self.assertRaisesRegex(ValueError, "unsafe"):
                MODULE.extract_english_punkt(archive, root / "out")


if __name__ == "__main__":
    unittest.main()
