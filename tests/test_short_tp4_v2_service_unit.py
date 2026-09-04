from __future__ import annotations

import short_tp4_v2_service as service
import unittest


def _response(prompt: int, cached: int, completion: int = 8) -> dict:
    return {
        "ok": True,
        "elapsed_s": 1.1,
        "ttft_s": 1.0,
        "tpot_s": 0.01 if completion > 1 else 0.0,
        "itl_s": 0.01 if completion > 1 else 0.0,
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "completion_tokens": completion,
        "finish_reason": "length",
        "input_tps": float(prompt),
        "output_tps": float(completion),
        "cache_tps": float(cached),
        "request_throughput_rps": 1.0 / 1.1,
        "first_output_identity": "a" * 64,
        "output_identity": "b" * 64,
    }


def _report() -> dict:
    cold_cases = []
    partial_cases = []
    for target in service.TARGETS:
        for repetition in range(service.REPETITIONS):
            cold_cases.append({
                "target_prompt_tokens": target,
                "repetition": repetition,
                "cold": _response(target, 0),
                "warm": _response(target, target - 16),
                "output_exact": True,
            })
            context = target - service.PARTIAL_RESIDUAL_TOKENS
            partial_cases.append({
                "target_prompt_tokens": target,
                "block_context_tokens": context,
                "partial_residual_tokens": service.PARTIAL_RESIDUAL_TOKENS,
                "shared_tokens_before_block_rounding": context,
                "repetition": repetition,
                "primer": _response(context, 0, 1),
                "first_sibling": _response(target, 0, 1),
                "partial": _response(target, context - 16),
                "warm": _response(target, target - 16),
                "output_exact": True,
            })
    return {
        "schema": service.SCHEMA,
        "version": 2,
        "targets": list(service.TARGETS),
        "partial_residual_tokens": service.PARTIAL_RESIDUAL_TOKENS,
        "block_size": service.BLOCK_SIZE,
        "max_tokens": service.MAX_TOKENS,
        "repetitions": service.REPETITIONS,
        "seed": service.SEED,
        "cold_cases": cold_cases,
        "partial_cases": partial_cases,
    }


class ShortTp4V2ServiceTests(unittest.TestCase):

    def test_ttft_tpot_and_output_tps_definitions(self) -> None:
        raw = {
            "ok": True, "elapsed_s": 3.0, "ttft_s": 2.0,
            "last_output_s": 2.7, "decode_window_s": 0.7,
            "output_tps": 999.0, "prompt_tokens": 16,
            "cached_tokens": 0, "completion_tokens": 8,
            "finish_reason": "length", "first_token_sha256": "a" * 64,
            "output_sha256": "b" * 64,
        }
        value = service._summarize_response(raw)
        self.assertEqual(value["ttft_s"], 2.0)
        self.assertAlmostEqual(value["tpot_s"], 0.1)
        self.assertAlmostEqual(value["output_tps"], 10.0)

    def test_single_output_token_has_no_decode_interval(self) -> None:
        raw = {
            "ok": True, "elapsed_s": 2.1, "ttft_s": 2.0,
            "last_output_s": 2.0, "decode_window_s": 0.0,
            "output_tps": 999.0, "prompt_tokens": 16,
            "cached_tokens": 0, "completion_tokens": 1,
            "finish_reason": "length", "first_token_sha256": "a" * 64,
            "output_sha256": "b" * 64,
        }
        value = service._summarize_response(raw)
        self.assertEqual(value["tpot_s"], 0.0)
        self.assertEqual(value["output_tps"], 0.0)

    def test_fixed_population_passes(self) -> None:
        result = service.evaluate(_report())
        self.assertTrue(result["qualified"], result)

    def test_incomplete_population_is_invalid(self) -> None:
        report = _report()
        report["cold_cases"].pop()
        result = service.evaluate(report)
        self.assertFalse(result["qualified"])

    def test_wrong_cache_accounting_fails(self) -> None:
        report = _report()
        report["partial_cases"][0]["partial"]["cached_tokens"] = 0
        result = service.evaluate(report)
        self.assertFalse(result["qualified"])
        self.assertTrue(any("partial cache" in reason
                            for reason in result["reasons"]))

    def test_output_identity_mismatch_fails(self) -> None:
        report = _report()
        report["cold_cases"][0]["warm"]["output_identity"] = "c" * 64
        result = service.evaluate(report)
        self.assertFalse(result["qualified"])


if __name__ == "__main__":
    unittest.main()
