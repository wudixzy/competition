from __future__ import annotations

import json
from pathlib import Path
import socket
from types import SimpleNamespace
import tempfile
import unittest

import run_attention_operator_tp4_arm as runner


class AttentionOperatorRunnerTests(unittest.TestCase):

    def test_long_workload_configuration_is_validated(self) -> None:
        self.assertEqual(
            runner._workload_config("131072,235000", 2),
            ((131072, 235000), 2))
        with self.assertRaises(ValueError):
            runner._workload_config("235000,131072", 2)
        with self.assertRaises(ValueError):
            runner._workload_config("262140", 2)

    def test_port_probe_checks_listener_not_bindability(self) -> None:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            self.assertTrue(runner._api_listener_absent(port))
            listener.listen()
            self.assertFalse(runner._api_listener_absent(port))

    def test_compiler_probe_uses_fixed_corex_toolchain(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertIn('"/usr/local/corex-3.2.3/bin/clang++"', source)
        self.assertIn('"runtime_probe.stderr"', source)
        self.assertIn("if probe.returncode:", source)
        self.assertIn('cwd=self.run_root / "runtime-workdir"', source)

    def test_reusable_session_preflight_is_fail_closed(self) -> None:
        value = {
            "schema": "bi100-session-preflight-v1", "version": 1,
            "qualified": True, "instance": "i-1",
            "session_preflight_id": "p-1", "gpu_indices": [0, 1, 2, 3],
            "gpu_health_qualified": True, "fp16_matmul_qualified": True,
            "collective_qualified": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preflight.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            self.assertEqual(runner._load_session_preflight(
                path, "i-1")["session_preflight_id"], "p-1")
            value["collective_qualified"] = False
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ValueError):
                runner._load_session_preflight(path, "i-1")

    def test_service_environment_differs_only_by_selector(self) -> None:
        root = Path(__file__).resolve().parents[1]
        values = []
        for selector in ("control", "candidate"):
            value = runner.AttentionOperatorTp4Runner.__new__(
                runner.AttentionOperatorTp4Runner)
            value.root = root
            value.runtime_site = root
            value.runtime_install = root / "README.md"
            value.run_root = Path("/tmp/attention-runner-unit")
            value.model_path = runner.EXPECTED_MODEL_PATH
            value.args = SimpleNamespace(selector=selector)
            values.append(value.service_environment())
        control, candidate = values
        self.assertEqual(control.pop("BI100_ATTN_COREX_FUSED_PREFILL"), "0")
        self.assertEqual(candidate.pop("BI100_ATTN_COREX_FUSED_PREFILL"), "1")
        self.assertEqual(control, candidate)
        self.assertNotIn(
            "BI100_ATTN_COREX_FUSED_PREFILL_EXTENSION_SHA256", values[0])


if __name__ == "__main__":
    unittest.main()
