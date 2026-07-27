from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from tests.service_postflight_gate import scan, scan_until_stable


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

    def test_settling_requires_consecutive_clean_samples(self):
        clock = [0.0]
        rows = iter([
            {"qualified": False, "gpu_processes": [{"pid": 10}]},
            {"qualified": True},
            {"qualified": False, "gpu_processes": [{"pid": 11}]},
            {"qualified": True},
            {"qualified": True},
            {"qualified": True},
        ])

        result = scan_until_stable(
            lambda: next(rows),
            settle_timeout_s=10.0,
            clean_samples=3,
            sample_interval_s=1.0,
            monotonic=lambda: clock[0],
            sleep=lambda seconds: clock.__setitem__(
                0, clock[0] + seconds),
        )

        self.assertTrue(result["qualified"])
        self.assertEqual(result["settling"]["attempts"], 6)
        self.assertEqual(result["settling"]["final_clean_streak"], 3)
        self.assertEqual(
            result["settling"]["observations"][0]["gpu_processes"],
            [{"pid": 10}],
        )

    def test_persistent_process_fails_after_settle_timeout(self):
        clock = [0.0]

        result = scan_until_stable(
            lambda: {
                "qualified": False,
                "api_server_pids": [101],
                "gpu_processes": [],
            },
            settle_timeout_s=2.0,
            clean_samples=2,
            sample_interval_s=1.0,
            monotonic=lambda: clock[0],
            sleep=lambda seconds: clock.__setitem__(
                0, clock[0] + seconds),
        )

        self.assertFalse(result["qualified"])
        self.assertEqual(result["settling"]["attempts"], 3)
        self.assertEqual(result["api_server_pids"], [101])

    def test_disabled_settling_is_one_snapshot(self):
        calls = [0]

        def scan_once():
            calls[0] += 1
            return {"qualified": True}

        result = scan_until_stable(
            scan_once,
            settle_timeout_s=0.0,
            clean_samples=1,
            sample_interval_s=1.0,
        )
        self.assertTrue(result["qualified"])
        self.assertEqual(calls[0], 1)


if __name__ == "__main__":
    unittest.main()
