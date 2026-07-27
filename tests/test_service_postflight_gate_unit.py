from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from tests.service_postflight_gate import scan


class ServicePostflightGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.proc = self.root / "proc"
        self.dev = self.root / "dev"
        self.proc.mkdir()
        self.dev.mkdir()
        for index in range(4):
            (self.dev / f"iluvatar{index}").touch()

    def tearDown(self):
        self.temporary.cleanup()

    def add_process(
        self,
        pid: int,
        *,
        command: bytes = b"python3\0",
        comm: str = "python3",
        gpu: int | None = None,
    ):
        process = self.proc / str(pid)
        fd = process / "fd"
        fd.mkdir(parents=True)
        (process / "cmdline").write_bytes(command)
        (process / "comm").write_text(comm + "\n", encoding="utf-8")
        if gpu is not None:
            os.symlink(self.dev / f"iluvatar{gpu}", fd / "7")

    def test_clean_host_passes(self):
        result = scan(self.proc, self.dev, (0, 1, 2, 3), 999)
        self.assertTrue(result["qualified"])
        self.assertEqual(result["gpu_processes"], [])

    def test_api_worker_and_gpu_holders_fail_without_raw_command(self):
        self.add_process(
            101,
            command=(
                b"python3\0-m\0vllm.entrypoints.openai.api_server\0"
                b"--api-key\0secret\0"
            ),
            gpu=2,
        )
        self.add_process(102, comm="VllmWorkerProcess")
        result = scan(self.proc, self.dev, (0, 1, 2, 3), 999)
        self.assertFalse(result["qualified"])
        self.assertEqual(result["api_server_pids"], [101])
        self.assertEqual(result["worker_pids"], [102])
        self.assertEqual(
            result["gpu_processes"],
            [{"pid": 101, "comm": "python3", "gpu_indices": [2]}],
        )
        self.assertNotIn("secret", str(result))

    def test_missing_device_fails_closed(self):
        (self.dev / "iluvatar3").unlink()
        result = scan(self.proc, self.dev, (0, 1, 2, 3), 999)
        self.assertFalse(result["qualified"])
        self.assertEqual(result["missing_devices"], [3])

    def test_scanner_ignores_itself(self):
        self.add_process(555, gpu=0)
        result = scan(self.proc, self.dev, (0, 1, 2, 3), 555)
        self.assertTrue(result["qualified"])


if __name__ == "__main__":
    unittest.main()
