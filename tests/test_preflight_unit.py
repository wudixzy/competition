import argparse
import os
import pathlib
import socket
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import bi100_nccl_preflight
import bi100_preflight


class Bi100PreflightUnitTest(unittest.TestCase):

    def test_cuda_parse_gpus_allows_single_gpu_and_ignores_empty_parts(self):
        self.assertEqual(bi100_preflight.parse_gpus("0, 2,,3"), [0, 2, 3])
        self.assertEqual(bi100_preflight.parse_gpus("1"), [1])
        with self.assertRaises(argparse.ArgumentTypeError):
            bi100_preflight.parse_gpus(",,")

    def test_clean_stream_normalizes_timeout_streams(self):
        self.assertEqual(bi100_preflight._clean_stream(None), "")
        self.assertEqual(bi100_preflight._clean_stream("  text\n"), "text")
        self.assertEqual(
            bi100_preflight._clean_stream(b"bad:\xff\n"),
            "bad:\ufffd",
        )

    def test_last_progress_stage_ignores_non_json_lines(self):
        output = "\n".join((
            "warning: initializing runtime",
            '{"gpu": 0, "stage": "set_device"}',
            "not-json",
            '{"gpu": 0, "stage": "synchronize"}',
        ))
        self.assertEqual(
            bi100_preflight._last_progress_stage(output), "synchronize")

    def test_cuda_timeout_reports_last_flushed_progress_stage(self):
        timeout = subprocess.TimeoutExpired(
            cmd=["python3"],
            timeout=25,
            output=(
                b'{"gpu": 0, "stage": "import_torch"}\n'
                b'{"gpu": 0, "stage": "matmul"}\n'
            ),
            stderr=b"runtime warning\n",
        )
        with patch.object(
                bi100_preflight.subprocess, "run", side_effect=timeout):
            result = bi100_preflight.probe_gpu(0, 25, 1024)

        self.assertEqual(result["stage"], "timeout")
        self.assertEqual(result["last_progress_stage"], "matmul")
        self.assertEqual(result["returncode"], 124)
        self.assertEqual(result["stderr"], "runtime warning")

    def test_cuda_corex_env_prepends_paths_without_mutating_os_environ(self):
        with patch.dict(os.environ, {
                "PATH": "/bin",
                "LD_LIBRARY_PATH": "/lib",
                "PYTHONPATH": "/py",
        }, clear=True):
            env = bi100_preflight.corex_env()
            self.assertEqual(os.environ["PATH"], "/bin")

        self.assertTrue(env["PATH"].startswith("/usr/local/corex/bin:"))
        self.assertTrue(
            env["LD_LIBRARY_PATH"].startswith("/usr/local/corex/lib:"))
        self.assertTrue(
            env["PYTHONPATH"].startswith(
                "/usr/local/corex/lib64/python3/dist-packages:"))


class Bi100NcclPreflightUnitTest(unittest.TestCase):

    def test_nccl_parse_gpus_requires_at_least_two_gpus(self):
        self.assertEqual(bi100_nccl_preflight.parse_gpus("0,1, 3"), [0, 1, 3])
        with self.assertRaises(argparse.ArgumentTypeError):
            bi100_nccl_preflight.parse_gpus("0")
        with self.assertRaises(argparse.ArgumentTypeError):
            bi100_nccl_preflight.parse_gpus("")

    def test_nccl_setup_corex_env_prepends_paths(self):
        with patch.object(sys, "path", ["/existing"]), patch.dict(os.environ, {
                "PATH": "/bin",
                "LD_LIBRARY_PATH": "/lib",
                "PYTHONPATH": "/py",
        }, clear=True):
            bi100_nccl_preflight.setup_corex_env()
            self.assertTrue(os.environ["PATH"].startswith(
                "/usr/local/corex/bin:"))
            self.assertTrue(os.environ["LD_LIBRARY_PATH"].startswith(
                "/usr/local/corex/lib:"))
            self.assertTrue(os.environ["PYTHONPATH"].startswith(
                "/usr/local/corex/lib64/python3/dist-packages:"))
            self.assertEqual(
                sys.path[:len(bi100_nccl_preflight.COREX_PYTHON_PATHS)],
                bi100_nccl_preflight.COREX_PYTHON_PATHS)

    def test_nccl_free_port_returns_bindable_local_port(self):
        port = bi100_nccl_preflight.free_port()
        self.assertIsInstance(port, int)
        self.assertGreater(port, 0)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", port))


if __name__ == "__main__":
    unittest.main()
