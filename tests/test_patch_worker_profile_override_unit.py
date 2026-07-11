import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "qwen3_6_scripts"
PATCH_SCRIPT = SCRIPTS / "patch_worker_profile_override.py"

CLEAN_BLOCK = """\
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()
"""

GUARDED_BLOCK = """\
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model. Mark this synthetic pass so BI100_PROFILE can skip
        # timing it by default; profiling real requests is the useful signal.
        _bi100_prev_startup_profile = os.environ.get("BI100_IN_STARTUP_PROFILE")
        os.environ["BI100_IN_STARTUP_PROFILE"] = "1"
        try:
            self.model_runner.profile_run()
        finally:
            if _bi100_prev_startup_profile is None:
                os.environ.pop("BI100_IN_STARTUP_PROFILE", None)
            else:
                os.environ["BI100_IN_STARTUP_PROFILE"] = _bi100_prev_startup_profile
"""


def _make_fake_vllm(root: pathlib.Path, worker_text: str) -> pathlib.Path:
    package = root / "vllm"
    worker = package / "worker"
    worker.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (worker / "__init__.py").write_text("")
    (worker / "worker.py").write_text(worker_text)
    return worker / "worker.py"


def _run_patch(fake_root: pathlib.Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(fake_root), str(SCRIPTS), env.get("PYTHONPATH", "")])
    return subprocess.run(
        [sys.executable, str(PATCH_SCRIPT)],
        cwd=SCRIPTS,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


class WorkerProfileOverridePatchUnitTest(unittest.TestCase):

    def test_patch_accepts_clean_and_guarded_vendor_blocks(self):
        for name, block in [("clean", CLEAN_BLOCK), ("guarded", GUARDED_BLOCK)]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                worker = _make_fake_vllm(root, "class Worker:\n" + block)

                first = _run_patch(root)
                self.assertEqual(first.returncode, 0, first.stderr)
                patched = worker.read_text()
                self.assertEqual(
                    patched.count("[BI100] skipping worker.profile_run"), 1)
                self.assertIn("BI100_IN_STARTUP_PROFILE", patched)
                self.assertIn("num_gpu_blocks_override is not None", patched)

                second = _run_patch(root)
                self.assertEqual(second.returncode, 0, second.stderr)
                self.assertIn("[skip] already patched", second.stdout)
                self.assertEqual(patched, worker.read_text())

    def test_patch_fails_fast_for_unknown_worker_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _make_fake_vllm(root, "class Worker:\n    pass\n")

            result = _run_patch(root)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("anchor not found", result.stderr)


if __name__ == "__main__":
    unittest.main()
