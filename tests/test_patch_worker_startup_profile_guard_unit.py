from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "qwen3_6_scripts"
PATCH_SCRIPT = SCRIPTS / "patch_worker_startup_profile_guard.py"
CLEAN_BLOCK = """\
class Worker:
        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()
"""


def _run_patch(root: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(root), str(SCRIPTS), environment.get("PYTHONPATH", "")))
    return subprocess.run(
        [sys.executable, str(PATCH_SCRIPT)],
        cwd=SCRIPTS,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


class WorkerStartupProfileGuardPatchTest(unittest.TestCase):
    def test_patch_is_idempotent_and_does_not_add_capacity_override(self):
        with tempfile.TemporaryDirectory() as directory_text:
            root = Path(directory_text)
            worker_dir = root / "vllm" / "worker"
            worker_dir.mkdir(parents=True)
            (root / "vllm" / "__init__.py").write_text("", encoding="utf-8")
            (worker_dir / "__init__.py").write_text("", encoding="utf-8")
            worker = worker_dir / "worker.py"
            worker.write_text(CLEAN_BLOCK, encoding="utf-8")

            first = _run_patch(root)
            self.assertEqual(first.returncode, 0, first.stderr)
            patched = worker.read_text(encoding="utf-8")
            self.assertIn("BI100_IN_STARTUP_PROFILE", patched)
            self.assertNotIn("num_gpu_blocks_override", patched)

            second = _run_patch(root)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("[skip] already patched", second.stdout)
            self.assertEqual(worker.read_text(encoding="utf-8"), patched)

    def test_unknown_worker_layout_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory_text:
            root = Path(directory_text)
            worker_dir = root / "vllm" / "worker"
            worker_dir.mkdir(parents=True)
            (root / "vllm" / "__init__.py").write_text("", encoding="utf-8")
            (worker_dir / "__init__.py").write_text("", encoding="utf-8")
            (worker_dir / "worker.py").write_text(
                "class Worker:\n    pass\n", encoding="utf-8")

            result = _run_patch(root)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("anchor not found", result.stderr)


if __name__ == "__main__":
    unittest.main()
