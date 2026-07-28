from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import unittest


TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))


def _load(name: str):
    path = TESTS / f"{name}.py"
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification and specification.loader
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


diagnostic = _load("diagnose_m1_116_fused_prefill_output")
comparison = _load("compare_m1_116_fused_prefill_output")
runtime_contract = _load("quality_runtime_contract")


def _digest(value: int) -> str:
    return f"{value:064x}"


def _request(
    max_tokens: int,
    *,
    cached: bool,
    output_digest: int,
    first_digest: int = 1,
    target_prompt_tokens: int = diagnostic.TARGET_PROMPT_TOKENS,
) -> dict:
    return {
        "status": 200,
        "elapsed_s": 1.25,
        "ttft_s": 0.75,
        "decode_window_s": 0.25,
        "output_tps": float(max_tokens),
        "prompt_tokens": target_prompt_tokens,
        "cached_tokens": (
            target_prompt_tokens - 32 if cached else 0),
        "completion_tokens": max_tokens,
        "finish_reason": "length",
        "first_token_hmac_sha256": _digest(first_digest),
        "output_hmac_sha256": _digest(output_digest),
        "request_contract_sha256": _digest(100 + max_tokens),
    }


def _contract(mode: str) -> dict:
    source_revision = "a" * 40
    runtime_identity = "b" * 64
    model_path = "/model"
    environment = runtime_contract.service_environment(
        "/runtime/site-packages",
        gdn_cache_policy="admission64",
        gdn_restore_mode="hybrid64",
        fused_prefill="0" if mode == "control" else "1",
        kv_eviction_policy="lru",
        kernel_profile="submission",
    )
    return {
        "schema": "bi100-quality-runtime-contract-v1",
        "version": 1,
        "source_revision": source_revision,
        "runtime_identity": runtime_identity,
        "runtime_overlay_sha256": "c" * 64,
        "instance": "test-instance",
        "gpu_count": 4,
        "tensor_parallel_size": 4,
        "max_model_len": 262144,
        "model_path": model_path,
        "tokenizer_path": model_path,
        "served_model_name": "llm",
        "base_image": runtime_contract.BASE_IMAGE,
        "command": runtime_contract.service_command(model_path),
        "environment": environment,
        "cache_trace_enabled": True,
        "optimization_label": f"m1-116-{mode}",
    }


def _report(mode: str) -> dict:
    contract = _contract(mode)
    cold = _request(32, cached=False, output_digest=32)
    warm = _request(32, cached=True, output_digest=32)
    ladder = []
    for max_tokens in diagnostic.MAX_TOKENS_LADDER:
        ladder.append({
            "max_tokens": max_tokens,
            "warm_1": _request(
                max_tokens, cached=True, output_digest=max_tokens),
            "warm_2": _request(
                max_tokens, cached=True, output_digest=max_tokens),
        })
    return {
        "schema": diagnostic.SCHEMA,
        "version": diagnostic.VERSION,
        "mode": mode,
        "qualified_diagnostic": True,
        "strict_quality_non_regression_authorized": False,
        "production_promotion_authorized": False,
        "reasons": [],
        "source_revision": contract["source_revision"],
        "runtime_identity": contract["runtime_identity"],
        "instance": contract["instance"],
        "model_path": contract["model_path"],
        "target_prompt_tokens": diagnostic.TARGET_PROMPT_TOKENS,
        "reproduction_max_tokens": diagnostic.REPRODUCTION_MAX_TOKENS,
        "max_tokens_ladder": list(diagnostic.MAX_TOKENS_LADDER),
        "seed": diagnostic.SEED,
        "run_id_sha256": "d" * 64,
        "runtime_contract": {
            "sha256": runtime_contract.sha256_json(contract),
            "contract": contract,
        },
        "reproduction": {
            "cold": cold,
            "warm": warm,
            "cold_warm_exact": True,
        },
        "secondary_reproduction": {
            "target_prompt_tokens":
                diagnostic.SECONDARY_TARGET_PROMPT_TOKENS,
            "max_tokens": diagnostic.REPRODUCTION_MAX_TOKENS,
            "run_id_sha256": "e" * 64,
            "cold": _request(
                32,
                cached=False,
                output_digest=320,
                target_prompt_tokens=
                    diagnostic.SECONDARY_TARGET_PROMPT_TOKENS,
            ),
            "warm": _request(
                32,
                cached=True,
                output_digest=320,
                target_prompt_tokens=
                    diagnostic.SECONDARY_TARGET_PROMPT_TOKENS,
            ),
            "cold_warm_exact": True,
        },
        "ladder": ladder,
        "privacy": {
            "contains_raw_requests": False,
            "contains_raw_model_outputs": False,
            "contains_token_ids": False,
            "contains_credentials": False,
        },
    }


class DiagnosticValidationTests(unittest.TestCase):
    def test_reported_identities_are_keyed_and_hide_raw_hashes(self) -> None:
        raw = {
            "elapsed_s": 1.0,
            "ttft_s": 0.5,
            "decode_window_s": 0.25,
            "output_tps": 4.0,
            "prompt_tokens": diagnostic.TARGET_PROMPT_TOKENS,
            "cached_tokens": 0,
            "completion_tokens": 1,
            "finish_reason": "length",
            "first_token_sha256": "1" * 64,
            "output_sha256": "2" * 64,
        }
        left = diagnostic._request_summary(raw, "3" * 64, b"a" * 32)
        same = diagnostic._request_summary(raw, "3" * 64, b"a" * 32)
        other = diagnostic._request_summary(raw, "3" * 64, b"b" * 32)
        self.assertEqual(left, same)
        self.assertNotEqual(
            left["first_token_hmac_sha256"],
            other["first_token_hmac_sha256"],
        )
        self.assertNotIn("first_token_sha256", left)
        self.assertNotIn("output_sha256", left)

    def test_valid_internal_observations_pass(self) -> None:
        report = _report("control")
        reproduction = report["reproduction"]
        self.assertEqual(
            diagnostic._validate_observations(
                reproduction["cold"],
                reproduction["warm"],
                report["ladder"],
            ),
            [],
        )

    def test_non_cold_reproduction_is_rejected(self) -> None:
        report = _report("control")
        report["reproduction"]["cold"]["cached_tokens"] = (
            diagnostic.TARGET_PROMPT_TOKENS - 32)
        reasons = diagnostic._validate_observations(
            report["reproduction"]["cold"],
            report["reproduction"]["warm"],
            report["ladder"],
        )
        self.assertIn("reproduction cold cold request was not cold", reasons)

    def test_repeat_output_drift_is_rejected(self) -> None:
        report = _report("control")
        report["ladder"][2]["warm_2"]["output_hmac_sha256"] = "e" * 64
        reasons = diagnostic._validate_observations(
            report["reproduction"]["cold"],
            report["reproduction"]["warm"],
            report["ladder"],
        )
        self.assertIn(
            "ladder max_tokens=4 repeated warm output differs", reasons)

    def test_secondary_reproduction_requires_internal_exactness(self) -> None:
        report = _report("control")
        report["secondary_reproduction"]["warm"][
            "output_hmac_sha256"] = "f" * 64
        reasons = diagnostic._validate_secondary_reproduction(
            report["secondary_reproduction"]["cold"],
            report["secondary_reproduction"]["warm"],
        )
        self.assertIn(
            "secondary reproduction cold/warm output differs", reasons)

    def test_request_summary_rejects_extra_raw_field(self) -> None:
        report = _report("control")
        report["reproduction"]["cold"]["raw_output"] = "forbidden"
        reasons = diagnostic._validate_observations(
            report["reproduction"]["cold"],
            report["reproduction"]["warm"],
            report["ladder"],
        )
        self.assertIn("reproduction cold fields are invalid", reasons)


class ComparisonTests(unittest.TestCase):
    def test_exact_output_authorizes_only_strict_quality_gate(self) -> None:
        result = comparison.compare(_report("control"), _report("candidate"))
        self.assertTrue(result["diagnostic_valid"])
        self.assertTrue(result["next_token_exact"])
        self.assertTrue(result["strict_output_exact"])
        self.assertTrue(result["strict_quality_non_regression_authorized"])
        self.assertFalse(result["production_promotion_authorized"])
        self.assertIsNone(result["first_divergent_max_tokens"])

    def test_later_output_divergence_requires_adjudication(self) -> None:
        control = _report("control")
        candidate = _report("candidate")
        for row in candidate["ladder"]:
            if row["max_tokens"] >= 8:
                row["warm_1"]["output_hmac_sha256"] = "e" * 64
                row["warm_2"]["output_hmac_sha256"] = "e" * 64
        candidate["reproduction"]["cold"]["output_hmac_sha256"] = "e" * 64
        candidate["reproduction"]["warm"]["output_hmac_sha256"] = "e" * 64

        result = comparison.compare(control, candidate)

        self.assertTrue(result["diagnostic_valid"])
        self.assertTrue(result["next_token_exact"])
        self.assertFalse(result["strict_output_exact"])
        self.assertTrue(result["quality_adjudication_required"])
        self.assertEqual(result["first_divergent_max_tokens"], 8)
        self.assertEqual(result["validation_reasons"], [])
        self.assertTrue(result["quality_reasons"])

    def test_first_token_divergence_is_a_quality_failure(self) -> None:
        control = _report("control")
        candidate = _report("candidate")
        candidate["ladder"][0]["warm_1"][
            "first_token_hmac_sha256"] = "e" * 64
        candidate["ladder"][0]["warm_2"][
            "first_token_hmac_sha256"] = "e" * 64

        result = comparison.compare(control, candidate)

        self.assertTrue(result["diagnostic_valid"])
        self.assertFalse(result["next_token_exact"])
        self.assertFalse(result["quality_adjudication_required"])
        self.assertFalse(result["strict_quality_non_regression_authorized"])

    def test_secondary_235k_divergence_requires_adjudication(self) -> None:
        control = _report("control")
        candidate = _report("candidate")
        candidate["secondary_reproduction"]["cold"][
            "output_hmac_sha256"] = "f" * 64
        candidate["secondary_reproduction"]["warm"][
            "output_hmac_sha256"] = "f" * 64

        result = comparison.compare(control, candidate)

        self.assertTrue(result["diagnostic_valid"])
        self.assertTrue(result["next_token_exact"])
        self.assertFalse(result["strict_output_exact"])
        self.assertTrue(result["quality_adjudication_required"])
        self.assertIsNone(result["first_divergent_max_tokens"])
        self.assertTrue(any(
            "secondary reproduction" in reason
            for reason in result["quality_reasons"]))

    def test_extra_runtime_change_invalidates_ab(self) -> None:
        control = _report("control")
        candidate = _report("candidate")
        candidate_contract = candidate["runtime_contract"]["contract"]
        candidate_contract["environment"]["BI100_DNN_CHUNK"] = "2048"
        candidate["runtime_contract"]["sha256"] = (
            runtime_contract.sha256_json(candidate_contract))

        result = comparison.compare(control, candidate)

        self.assertFalse(result["diagnostic_valid"])
        self.assertTrue(any(
            "environment" in reason for reason in result["reasons"]))

    def test_report_is_not_mutated(self) -> None:
        control = _report("control")
        candidate = _report("candidate")
        original_control = copy.deepcopy(control)
        original_candidate = copy.deepcopy(candidate)
        comparison.compare(control, candidate)
        self.assertEqual(control, original_control)
        self.assertEqual(candidate, original_candidate)

    def test_report_rejects_extra_raw_field(self) -> None:
        control = _report("control")
        candidate = _report("candidate")
        candidate["raw_output"] = "forbidden"
        result = comparison.compare(control, candidate)
        self.assertFalse(result["diagnostic_valid"])
        self.assertIn(
            "candidate report fields are invalid",
            result["validation_reasons"],
        )


class HarnessWiringTests(unittest.TestCase):
    def test_wrapper_selects_m1_116_without_changing_shared_defaults(self) -> None:
        wrapper = (
            TESTS.parent
            / "scripts/run_m1_116_fused_prefill_quality_adjudication_ab.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "BI100_QUALITY_AB_VARIANT=m1-116-fused-prefill-adjudication",
            wrapper,
        )
        self.assertIn("run_m1_85_admission64_quality_ab.sh", wrapper)

    def test_ab_runner_uses_fixed_private_diagnostic_contract(self) -> None:
        runner = (
            TESTS.parent / "scripts/run_m1_85_admission64_quality_ab.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("BI100_RUN_FUSED_OUTPUT_DIAGNOSTIC=1", runner)
        self.assertIn(
            "BI100_FUSED_OUTPUT_DIAGNOSTIC_RUN_ID=m1-109-pair-1-20260729",
            runner,
        )
        self.assertIn("secrets.token_hex(32)", runner)
        self.assertIn(
            "compare_m1_116_fused_prefill_output.py", runner)
        self.assertIn(
            "[[ $fused_output_comparison_rc -eq 0 ]]", runner)

    def test_service_gate_records_diagnostic_without_exporting_runner_flags(
        self,
    ) -> None:
        service_gate = (
            TESTS.parent / "scripts/run_quality_service_gate.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "diagnose_m1_116_fused_prefill_output.py", service_gate)
        self.assertIn('"fused_output_diagnostic": read_rc(', service_gate)
        self.assertIn(
            '"fused_output_diagnostic_sha256": (', service_gate)
        self.assertNotIn(
            "export BI100_RUN_FUSED_OUTPUT_DIAGNOSTIC", service_gate)
        self.assertNotIn(
            "export BI100_FUSED_OUTPUT_DIAGNOSTIC_RUN_ID", service_gate)
        self.assertIn(
            "unset BI100_FUSED_OUTPUT_DIAGNOSTIC_HMAC_KEY", service_gate)
        self.assertIn(
            "unset BI100_RUN_FUSED_OUTPUT_DIAGNOSTIC", service_gate)
        self.assertIn(
            "unset BI100_FUSED_OUTPUT_DIAGNOSTIC_RUN_ID", service_gate)
        self.assertLess(
            service_gate.index(
                "unset BI100_FUSED_OUTPUT_DIAGNOSTIC_HMAC_KEY"),
            service_gate.index('"$ROOT/launch_service"'),
        )


if __name__ == "__main__":
    unittest.main()
