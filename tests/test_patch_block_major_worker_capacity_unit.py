import os
import pathlib
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "qwen3_6_scripts"
PATCH_SCRIPT = SCRIPTS / "patch_block_major_worker_capacity.py"

WORKER_SOURCE = """\
from vllm.logger import init_logger


class Worker:

    def determine_num_available_blocks(self):
        num_gpu_blocks = 67512
        num_cpu_blocks = 26212
        cache_block_size = 163840
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        return num_gpu_blocks, num_cpu_blocks
"""


def make_fake_worker(root: pathlib.Path, source: str) -> pathlib.Path:
    package = root / "vllm"
    worker = package / "worker"
    worker.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (worker / "__init__.py").write_text("", encoding="utf-8")
    target = worker / "worker.py"
    target.write_text(source, encoding="utf-8")
    return target


def run_patch(fake_root: pathlib.Path) -> subprocess.CompletedProcess:
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


class BlockMajorWorkerCapacityPatchTest(unittest.TestCase):

    def test_patch_reserves_staging_capacity_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            target = make_fake_worker(root, WORKER_SOURCE)
            first = run_patch(root)
            self.assertEqual(first.returncode, 0, first.stderr)
            patched = target.read_text(encoding="utf-8")
            self.assertIn(
                "from vllm.block_major_kv_cache import "
                "reserve_block_major_gpu_blocks",
                patched,
            )
            self.assertIn(
                "num_gpu_blocks = reserve_block_major_gpu_blocks(",
                patched,
            )
            self.assertLess(
                patched.index("reserve_block_major_gpu_blocks("),
                patched.index("num_gpu_blocks = max("),
            )

            second = run_patch(root)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("[skip] already patched", second.stdout)
            self.assertEqual(target.read_text(encoding="utf-8"), patched)

    def test_unknown_worker_layout_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            make_fake_worker(root, "class Worker:\n    pass\n")
            result = run_patch(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("anchor not found", result.stderr)


if __name__ == "__main__":
    unittest.main()
