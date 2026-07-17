import os
import pathlib
import stat
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "qwen3_6_scripts" / "install_prebuilt_corex.sh"
EXPECTED_NAMES = set(
    __import__("tests.submission_preflight", fromlist=[
        "PREBUILT_COREX_SHA256"
    ]).PREBUILT_COREX_SHA256
)


class PrebuiltCorexInstallerTest(unittest.TestCase):

    def test_installer_verifies_and_installs_all_extensions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            vllm_root = root / "vllm"
            vllm_root.mkdir()
            completed = subprocess.run(
                ["bash", str(INSTALLER), str(vllm_root)],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ.copy(),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            installed = {path.name for path in vllm_root.glob("corex_*.so")}
            self.assertEqual(installed, EXPECTED_NAMES)
            for name in EXPECTED_NAMES:
                mode = (vllm_root / name).stat().st_mode
                self.assertTrue(mode & stat.S_IXUSR)
            self.assertEqual(completed.stdout.count("[ok] installed"), 10)


if __name__ == "__main__":
    unittest.main()
