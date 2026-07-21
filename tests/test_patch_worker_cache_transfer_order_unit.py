import os
import pathlib
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "qwen3_6_scripts"
PATCH_SCRIPT = SCRIPTS / "patch_worker_cache_transfer_order.py"

CLEAN_BLOCK = """\
        if (worker_input.blocks_to_swap_in is not None
                and worker_input.blocks_to_swap_in.numel() > 0):
            self.cache_engine[virtual_engine].swap_in(
                worker_input.blocks_to_swap_in)
        if (worker_input.blocks_to_swap_out is not None
                and worker_input.blocks_to_swap_out.numel() > 0):
            self.cache_engine[virtual_engine].swap_out(
                worker_input.blocks_to_swap_out)
"""


def make_fake_worker(root: pathlib.Path, body: str) -> pathlib.Path:
    package = root / "vllm"
    worker = package / "worker"
    worker.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (worker / "__init__.py").write_text("", encoding="utf-8")
    target = worker / "worker.py"
    target.write_text("class Worker:\n    def execute(self):\n" + body,
                      encoding="utf-8")
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


class WorkerCacheTransferOrderPatchTest(unittest.TestCase):
    def test_patch_orders_d2h_before_h2d_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            target = make_fake_worker(root, CLEAN_BLOCK)
            first = run_patch(root)
            self.assertEqual(first.returncode, 0, first.stderr)
            patched = target.read_text(encoding="utf-8")
            self.assertIn("Complete every D2H before any H2D", patched)
            self.assertLess(
                patched.index(".swap_out("), patched.index(".swap_in("))

            second = run_patch(root)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("[skip] already patched", second.stdout)
            self.assertEqual(target.read_text(encoding="utf-8"), patched)

    def test_unknown_worker_layout_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            make_fake_worker(root, "        pass\n")
            result = run_patch(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("anchor not found", result.stderr)


if __name__ == "__main__":
    unittest.main()
