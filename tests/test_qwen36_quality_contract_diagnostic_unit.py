from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))
SCRIPT = ROOT / "tests/qwen36_quality_contract_diagnostic.py"
SPEC = importlib.util.spec_from_file_location(
    "qwen36_quality_contract_diagnostic", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeClient:

    def __init__(self, healthy: bool = True) -> None:
        self.healthy = healthy
        self.health_checks = 0

    def models(self, expected_model: str = "llm"):
        self.health_checks += 1
        if not self.healthy:
            raise MODULE.quality.CaseFailure("model-list endpoint unavailable")
        return {"data": [{"id": expected_model}]}


def observation(case_id: str) -> dict:
    status = 200 if case_id in {"top_p_0", "n_2"} else 400
    return {
        "status_codes": [status],
        "finish_reasons": ["stop"] if status == 200 else [],
        "prompt_tokens": [4] if status == 200 else [],
        "cached_tokens": [0] if status == 200 else [],
        "completion_tokens": [1] if status == 200 else [],
        "semantic_output_sha256": "a" * 64,
        "facts": {"contract_checked": True},
    }


class DiagnosticQualityContractTest(unittest.TestCase):

    def test_frozen_cases_run_in_manifest_order_without_raw_payloads(self):
        calls = []

        def handler(case_id):
            def run(client, config):
                calls.append(case_id)
                self.assertEqual(config.model, "llm")
                self.assertEqual(config.max_model_len, 262144)
                self.assertEqual(config.endpoint_mode, "direct")
                return observation(case_id)
            return run

        handlers = {
            case_id: handler(case_id) for case_id in MODULE.CASE_IDS
        }
        client = FakeClient()
        report = MODULE.run_gate(
            "http://private-endpoint:8000",
            client=client,
            handlers=handlers,
        )
        self.assertTrue(report["qualified"])
        self.assertEqual(report["case_count"], 9)
        self.assertEqual(report["passed"], 9)
        self.assertEqual(calls, list(MODULE.CASE_IDS))
        self.assertEqual(client.health_checks, 1)
        serialized = json.dumps(report)
        self.assertNotIn("private-endpoint", serialized)
        self.assertFalse(report["scope"]["production_promotion_authorized"])

    def test_case_failure_and_final_health_fail_closed(self):
        handlers = {
            case_id: (lambda client, config, name=case_id: observation(name))
            for case_id in MODULE.CASE_IDS
        }

        def fail(client, config):
            raise MODULE.quality.CaseFailure("fixed contract failure")

        handlers["empty_messages"] = fail
        report = MODULE.run_gate(
            "http://127.0.0.1:8000",
            client=FakeClient(),
            handlers=handlers,
        )
        self.assertFalse(report["qualified"])
        self.assertEqual(report["failed"], 1)
        failed = next(case for case in report["cases"] if not case["ok"])
        self.assertEqual(failed["id"], "empty_messages")
        self.assertEqual(failed["error_code"], "fixed contract failure")
        self.assertIsNone(failed["observation"])

        unhealthy = MODULE.run_gate(
            "http://127.0.0.1:8000",
            client=FakeClient(healthy=False),
            handlers={
                case_id: (
                    lambda client, config, name=case_id: observation(name))
                for case_id in MODULE.CASE_IDS
            },
        )
        self.assertFalse(unhealthy["qualified"])
        self.assertFalse(unhealthy["final_health"])


if __name__ == "__main__":
    unittest.main()
