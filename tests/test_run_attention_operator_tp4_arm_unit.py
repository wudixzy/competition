from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import run_attention_operator_tp4_arm as runner


class AttentionOperatorRunnerTests(unittest.TestCase):

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
