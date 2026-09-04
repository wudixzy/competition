from __future__ import annotations

import copy
import unittest

import attention_operator_tp4_service as service


def _response(target: int, ttft: float = 1.0) -> dict:
    return {
        "ok": True, "http_status": 200, "sse_complete": True,
        "usage_complete": True, "elapsed_s": ttft + 0.7,
        "ttft_s": ttft, "last_output_s": ttft + 0.7,
        "decode_window_s": 0.7, "tpot_s": 0.1, "output_tps": 10.0,
        "prompt_tokens": target, "completion_tokens": service.MAX_TOKENS,
        "finish_reason": "length",
    }


def report(selector: str = "control", ttft: float = 1.0) -> dict:
    cases = [
        {"target_prompt_tokens": target, "repetition": repetition,
         "response": _response(target, ttft)}
        for target in service.TARGETS
        for repetition in range(service.REPETITIONS)
    ]
    return {
        "schema": service.SCHEMA, "version": 1,
        "change_scope": "attention_operator", "selector": selector,
        "run_id": f"run-{selector}", "workload_id": "pair-1",
        "targets": list(service.TARGETS),
        "repetitions": service.REPETITIONS,
        "max_tokens": service.MAX_TOKENS, "seed": service.SEED,
        "workload_order": service.WORKLOAD_ORDER,
        "expected_requests": 9, "attempted_requests": 9,
        "completed_requests": 9, "failed_requests": 0,
        "cases": cases,
    }


class AttentionOperatorServiceTests(unittest.TestCase):

    def test_complete_cold_population_passes(self) -> None:
        value = report()
        self.assertTrue(service.evaluate(value)["qualified"])

    def test_missing_request_is_rejected(self) -> None:
        value = report()
        value["cases"].pop()
        self.assertFalse(service.evaluate(value)["qualified"])

    def test_protocol_and_nonfinite_timing_are_rejected(self) -> None:
        value = report()
        value["cases"][0]["response"]["sse_complete"] = False
        value["cases"][1]["response"]["ttft_s"] = float("nan")
        result = service.evaluate(value)
        self.assertFalse(result["qualified"])
        self.assertTrue(any("HTTP/SSE/usage" in item for item in result["reasons"]))
        self.assertTrue(any("ttft_s" in item for item in result["reasons"]))

    def test_summary_defines_first_to_last_token_metrics(self) -> None:
        raw = {
            "ok": True, "elapsed_s": 3.0, "ttft_s": 2.0,
            "last_output_s": 2.7, "completion_tokens": 8,
            "prompt_tokens": 16, "finish_reason": "length",
        }
        value = service.summarize_response(copy.deepcopy(raw))
        self.assertEqual(value["ttft_s"], 2.0)
        self.assertAlmostEqual(value["tpot_s"], 0.1)
        self.assertAlmostEqual(value["output_tps"], 10.0)


if __name__ == "__main__":
    unittest.main()
