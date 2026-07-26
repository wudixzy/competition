import os
import pathlib
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "qwen3_6_scripts"
PATCH_SCRIPT = SCRIPTS / "patch_block_major_cache_engine.py"

FAKE_CACHE_ENGINE = """\
import torch

from vllm.logger import init_logger
from vllm.utils import is_pin_memory_available


class CacheEngine:
    def __init__(self):
        self.gpu_cache = self._allocate_kv_cache(
            self.num_gpu_blocks, self.device_config.device_type)
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                          src_to_dst)

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)
"""


def make_fake_package(root: pathlib.Path, source: str) -> pathlib.Path:
    package = root / "vllm"
    worker = package / "worker"
    worker.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (worker / "__init__.py").write_text("", encoding="utf-8")
    target = worker / "cache_engine.py"
    target.write_text(source, encoding="utf-8")
    return target


def run_patch(root: pathlib.Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([
        str(root),
        str(SCRIPTS),
        env.get("PYTHONPATH", ""),
    ])
    return subprocess.run(
        [sys.executable, str(PATCH_SCRIPT)],
        cwd=SCRIPTS,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


class BlockMajorCacheEnginePatchTest(unittest.TestCase):

    def test_patch_is_complete_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            target = make_fake_package(root, FAKE_CACHE_ENGINE)
            first = run_patch(root)
            self.assertEqual(first.returncode, 0, first.stderr)
            patched = target.read_text(encoding="utf-8")
            self.assertIn(
                "from vllm.block_major_kv_cache import", patched)
            self.assertIn(
                "self._bi100_block_major_cpu_kv = None", patched)
            self.assertIn(
                "self._bi100_block_major_cpu_kv.swap_in", patched)
            self.assertIn(
                "self._bi100_block_major_cpu_kv.swap_out", patched)
            self.assertIn(
                "self.cpu_cache = self._allocate_kv_cache", patched)

            second = run_patch(root)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(target.read_text(encoding="utf-8"), patched)
            self.assertEqual(
                second.stdout.count("[skip] already patched"), 3)

    def test_unknown_vendor_layout_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            make_fake_package(
                root,
                "from vllm.logger import init_logger\n"
                "class CacheEngine:\n"
                "    pass\n",
            )
            result = run_patch(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("anchor not found", result.stderr)

    def test_patch_ops_installs_module_and_patch(self):
        patch_ops = (SCRIPTS / "patch_ops.sh").read_text(
            encoding="utf-8")
        self.assertIn(
            'cp ./block_major_kv_cache.py "${VLLM_ROOT}/'
            'block_major_kv_cache.py"',
            patch_ops,
        )
        self.assertIn(
            "python3 ./patch_block_major_cache_engine.py", patch_ops)
        installer = (
            ROOT / "scripts" / "install_bi100_bare_host_runtime.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "corex_block_major_kv_transfer.so", installer)
        self.assertIn(
            '"block_major_cache_engine_patch": '
            "block_major_cache_engine_patch",
            installer,
        )


if __name__ == "__main__":
    unittest.main()
