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


def observation(
    case_id: str,
    *,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    choice_digest: str = "b" * 64,
) -> dict:
    status = 200 if case_id in {"top_p_0", "n_1", "n_2"} else 400
    if prompt_tokens is None:
        prompt_tokens = 4
    if completion_tokens is None:
        completion_tokens = 2 if case_id == "n_2" else 1
    facts = {"contract_checked": True}
    if case_id in {"n_1", "n_2"}:
        facts.update({
            "n": 1 if case_id == "n_1" else 2,
            "choice_indices_exact": True,
            "usage_accounted": True,
            "deterministic_choices_exact": True,
            "choice_output_sha256": choice_digest,
        })
    return {
        "status_codes": [status],
        "finish_reasons": ["stop"] if status == 200 else [],
        "prompt_tokens": [prompt_tokens] if status == 200 else [],
        "cached_tokens": [0] if status == 200 else [],
        "completion_tokens": [completion_tokens] if status == 200 else [],
        "semantic_output_sha256": "a" * 64,
        "facts": facts,
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
        self.assertEqual(report["schema"],
                         "qwen36-diagnostic-quality-contract-v2")
        self.assertEqual(report["version"], 2)
        self.assertEqual(report["case_count"], 10)
        self.assertEqual(report["passed"], 10)
        self.assertTrue(report["n_cross_case_contract"]["qualified"])
        self.assertTrue(all(
            report["n_cross_case_contract"]["checks"].values()))
        self.assertEqual(report["n_cross_case_contract"]["reasons"], [])
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

    def test_n_cross_case_contract_fails_closed_on_accounting_or_output(self):
        scenarios = {
            "prompt": {
                "n_2": {"prompt_tokens": 8},
                "failed_check": "prompt_counted_once",
            },
            "completion": {
                "n_2": {"completion_tokens": 3},
                "failed_check": "completion_summed",
            },
            "output": {
                "n_2": {"choice_digest": "c" * 64},
                "failed_check": "choice_output_exact",
            },
        }
        for name, scenario in scenarios.items():
            with self.subTest(name=name):
                overrides = scenario.get("n_2", {})
                handlers = {
                    case_id: (
                        lambda client, config, case=case_id:
                        observation(
                            case,
                            **(overrides if case == "n_2" else {}),
                        )
                    )
                    for case_id in MODULE.CASE_IDS
                }
                report = MODULE.run_gate(
                    "http://127.0.0.1:8000",
                    client=FakeClient(),
                    handlers=handlers,
                )
                self.assertEqual(report["failed"], 0)
                self.assertFalse(report["qualified"])
                contract = report["n_cross_case_contract"]
                self.assertFalse(contract["qualified"])
                self.assertFalse(
                    contract["checks"][scenario["failed_check"]])

    def test_n_cross_case_contract_rejects_noncanonical_digest(self):
        handlers = {
            case_id: (
                lambda client, config, case=case_id:
                observation(
                    case,
                    choice_digest=("B" * 64 if case == "n_2" else "b" * 64),
                )
            )
            for case_id in MODULE.CASE_IDS
        }
        report = MODULE.run_gate(
            "http://127.0.0.1:8000",
            client=FakeClient(),
            handlers=handlers,
        )
        self.assertFalse(report["qualified"])
        self.assertFalse(report["n_cross_case_contract"]["checks"][
            "choice_output_exact"])


if __name__ == "__main__":
    unittest.main()
