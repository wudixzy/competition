from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(SCRIPTS))

SERVICE_PATH = TESTS / "short_tp4_p90_funnel_service.py"
SERVICE_SPEC = importlib.util.spec_from_file_location(
    "short_tp4_p90_funnel_service", SERVICE_PATH)
SERVICE = importlib.util.module_from_spec(SERVICE_SPEC)
assert SERVICE_SPEC.loader is not None
sys.modules[SERVICE_SPEC.name] = SERVICE
SERVICE_SPEC.loader.exec_module(SERVICE)

QUALIFIER_PATH = TESTS / "qualify_m1_152_short_tp4_p90_pair.py"
QUALIFIER_SPEC = importlib.util.spec_from_file_location(
    "qualify_m1_152_short_tp4_p90_pair", QUALIFIER_PATH)
QUALIFIER = importlib.util.module_from_spec(QUALIFIER_SPEC)
assert QUALIFIER_SPEC.loader is not None
sys.modules[QUALIFIER_SPEC.name] = QUALIFIER
QUALIFIER_SPEC.loader.exec_module(QUALIFIER)

RUNNER_PATH = SCRIPTS / "run_m1_152_short_tp4_p90_screen.py"
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_m1_152_short_tp4_p90_screen", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC.loader is not None
sys.modules[RUNNER_SPEC.name] = RUNNER
RUNNER_SPEC.loader.exec_module(RUNNER)

PROMPT_CHECK_PATH = TESTS / "check_m1_152_prompt_construction.py"
PROMPT_CHECK_SPEC = importlib.util.spec_from_file_location(
    "check_m1_152_prompt_construction", PROMPT_CHECK_PATH)
PROMPT_CHECK = importlib.util.module_from_spec(PROMPT_CHECK_SPEC)
assert PROMPT_CHECK_SPEC.loader is not None
sys.modules[PROMPT_CHECK_SPEC.name] = PROMPT_CHECK
PROMPT_CHECK_SPEC.loader.exec_module(PROMPT_CHECK)

CONTRACT_PATH = ROOT / "quality" / "short_tp4_p90_pair.v2.json"
CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="ascii"))
CONTRACT_SHA = hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def response(
    prompt_tokens: int,
    *,
    cached_tokens: int,
    completion_tokens: int,
    ttft_s: float,
    output_label: str,
) -> dict:
    return {
        "ok": True,
        "elapsed_s": ttft_s + 0.1,
        "ttft_s": ttft_s,
        "last_output_s": ttft_s + 0.05,
        "decode_window_s": 0.05,
        "output_tps": completion_tokens / 0.05,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": "length",
        "first_token_sha256": digest(output_label + ":first"),
        "output_sha256": digest(output_label + ":output"),
    }


def measurement(selector: str, *, speedup: float = 1.0) -> dict:
    cold_cases = []
    for index, target in enumerate(SERVICE.TARGETS):
        label = f"{selector}:cold:{target}"
        ttft = (1.0 + index) / speedup
        cold_cases.append({
            "target_prompt_tokens": target,
            "repetition": 0,
            "prompt_sha256": digest(f"shared:cold:{target}"),
            "cold": response(
                target,
                cached_tokens=0,
                completion_tokens=SERVICE.MAX_TOKENS,
                ttft_s=ttft,
                output_label=label,
            ),
            "warm": response(
                target,
                cached_tokens=target - 16,
                completion_tokens=SERVICE.MAX_TOKENS,
                ttft_s=0.2,
                output_label=label,
            ),
        })
    partial_cases = []
    for index, target in enumerate(SERVICE.PARTIAL_TARGETS):
        context = target - SERVICE.PARTIAL_RESIDUAL_TOKENS
        label = f"{selector}:partial:{target}"
        ttft = (2.0 + index) / speedup
        partial_cases.append({
            "target_prompt_tokens": target,
            "block_context_tokens": context,
            "partial_residual_tokens": SERVICE.PARTIAL_RESIDUAL_TOKENS,
            "shared_tokens_before_block_rounding": context + 7,
            "repetition": 0,
            "primer_prompt_sha256": digest(f"shared:primer:{target}"),
            "partial_prompt_sha256": digest(f"shared:partial:{target}"),
            "primer": response(
                context + 128,
                cached_tokens=0,
                completion_tokens=1,
                ttft_s=0.5,
                output_label=f"{selector}:primer:{target}",
            ),
            "partial": response(
                target,
                cached_tokens=context,
                completion_tokens=SERVICE.MAX_TOKENS,
                ttft_s=ttft,
                output_label=label,
            ),
            "warm": response(
                target,
                cached_tokens=target - 16,
                completion_tokens=SERVICE.MAX_TOKENS,
                ttft_s=0.2,
                output_label=label,
            ),
        })
    cold_ttfts = [case["cold"]["ttft_s"] for case in cold_cases]
    partial_ttfts = [
        case["partial"]["ttft_s"] for case in partial_cases]
    warm_ttfts = (
        [case["warm"]["ttft_s"] for case in cold_cases]
        + [case["warm"]["ttft_s"] for case in partial_cases]
    )
    value = {
        "schema": SERVICE.SCHEMA,
        "version": 2,
        "run_id": f"run-{selector}",
        "prompt_set_id": "shared-pair",
        "selector": selector,
        "targets": list(SERVICE.TARGETS),
        "partial_targets": list(SERVICE.PARTIAL_TARGETS),
        "partial_residual_tokens": SERVICE.PARTIAL_RESIDUAL_TOKENS,
        "block_size": SERVICE.BLOCK_SIZE,
        "max_tokens": SERVICE.MAX_TOKENS,
        "repetitions": SERVICE.REPETITIONS,
        "seed": SERVICE.SEED,
        "elapsed_s": 10.0,
        "cold_cases": cold_cases,
        "partial_cases": partial_cases,
        "cold_ttft_median_s": statistics.median(cold_ttfts),
        "partial_ttft_median_s": statistics.median(partial_ttfts),
        "uncached_ttft_p90_s": SERVICE._percentile(
            cold_ttfts + partial_ttfts, 90.0),
        "warm_ttft_median_s": statistics.median(warm_ttfts),
        "privacy": dict(QUALIFIER.MEASUREMENT_PRIVACY),
        "authorization": dict(QUALIFIER.MEASUREMENT_AUTHORIZATION),
    }
    value["evaluation"] = SERVICE.evaluate(value)
    value["qualified"] = value["evaluation"]["qualified"]
    value["reasons"] = value["evaluation"]["reasons"]
    return value


def status(selector: str, measurement_sha: str) -> dict:
    extension_sha = "b" * 64
    kernel_sha = "e" * 64
    l2 = {
        "qualification_sha256": digest("l2-qualification"),
        "runner_status_sha256": digest("l2-runner"),
        "candidate_extension_sha256": extension_sha,
        "candidate_extension_size_bytes": 247176,
        "capture_source_revision": "a" * 40,
        "replay_source_revision": "c" * 40,
        "runtime_identity": "capture-runtime",
        "activation_run_id": "capture-run",
    }
    p90 = {
        "runner_status_sha256": digest("p90-status"),
        "identity_sha256": digest("p90-identity"),
        "source_revision": "f" * 40,
        "candidate_extension_sha256": extension_sha,
        "kernel_source_sha256": kernel_sha,
        "minimum_speedup": 1.9,
        "median_speedup": 2.2,
        "case_count": 8,
    }
    artifacts = {
        name: digest(f"{selector}:{name}")
        for name in QUALIFIER.REQUIRED_ARTIFACTS
    }
    artifacts["measurement.json"] = measurement_sha
    return {
        "schema": QUALIFIER.RUNNER_SCHEMA,
        "version": 2,
        "qualified": True,
        "returncode": 0,
        "terminal_stage": "complete",
        "error_type": None,
        "run_id": f"run-{selector}",
        "source_revision": "d" * 40,
        "source_branch": "private-experiment",
        "instance": "instance",
        "selector": selector,
        "pair_id": "shared-pair",
        "runtime_identity": "bare-host-overlay-v1:1234567890",
        "gpu_count": 4,
        "tensor_parallel_size": 4,
        "service_startups": 1,
        "targets": list(SERVICE.TARGETS),
        "partial_targets": list(SERVICE.PARTIAL_TARGETS),
        "partial_residual_tokens": SERVICE.PARTIAL_RESIDUAL_TOKENS,
        "repetitions": SERVICE.REPETITIONS,
        "gates": {
            name: 0 for name in QUALIFIER.REQUIRED_GATES
        },
        "artifact_sha256": artifacts,
        "candidate_extension": (
            {
                "sha256": None,
                "size_bytes": None,
                "external_override_active": False,
            }
            if selector == "control"
            else {
                "sha256": extension_sha,
                "size_bytes": 247176,
                "external_override_active": True,
            }
        ),
        "dispatch_count": 0 if selector == "control" else 4,
        "kernel_source_sha256": kernel_sha,
        "l2_authorization": l2,
        "p90_operator_authorization": p90,
        "timing": {
            "wall_span_s": 600.0,
            "summed_stage_s": 600.0,
        },
        "authorization": dict(QUALIFIER.RUNNER_AUTHORIZATION),
        "privacy": dict(QUALIFIER.RUNNER_PRIVACY),
    }


def qualify_pair(control: dict, candidate: dict) -> dict:
    shas = {
        "control": digest(json.dumps(control, sort_keys=True)),
        "candidate": digest(json.dumps(candidate, sort_keys=True)),
    }
    return QUALIFIER.qualify(
        {
            "control": status("control", shas["control"]),
            "candidate": status("candidate", shas["candidate"]),
        },
        {
            "control": control,
            "candidate": candidate,
        },
        shas,
        CONTRACT,
        contract_sha256=CONTRACT_SHA,
    )


class M1152ShortTp4P90PairTest(unittest.TestCase):

    def test_cpu_prompt_construction_report_checks_exact_boundaries(self):
        value = {
            "schema": PROMPT_CHECK.SCHEMA,
            "version": 1,
            "cold": [
                {
                    "target_prompt_tokens": target,
                    "actual_prompt_tokens": target,
                    "prompt_sha256": digest(f"cold:{target}"),
                }
                for target in SERVICE.TARGETS
            ],
            "partial": [
                {
                    "target_prompt_tokens": target,
                    "actual_prompt_tokens": target,
                    "block_context_tokens": (
                        target - SERVICE.PARTIAL_RESIDUAL_TOKENS),
                    "shared_tokens_before_rounding": (
                        target - SERVICE.PARTIAL_RESIDUAL_TOKENS + 7),
                    "cached_prefix_tokens": (
                        target - SERVICE.PARTIAL_RESIDUAL_TOKENS),
                    "residual_prefill_tokens": (
                        SERVICE.PARTIAL_RESIDUAL_TOKENS),
                    "primer_prompt_tokens": target,
                    "primer_prompt_sha256": digest(f"primer:{target}"),
                    "partial_prompt_sha256": digest(f"partial:{target}"),
                }
                for target in SERVICE.PARTIAL_TARGETS
            ],
            "privacy": {
                "prompts_recorded": False,
                "token_ids_recorded": False,
                "credentials_recorded": False,
            },
        }
        result = PROMPT_CHECK.validate(value)
        self.assertTrue(result["qualified"], result)
        value["partial"][0]["residual_prefill_tokens"] += 16
        result = PROMPT_CHECK.validate(value)
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "prefix boundary differs" in reason
            for reason in result["reasons"]
        ))

    def test_service_hard_gate_accepts_valid_cold_and_partial_matrix(self):
        value = measurement("control")
        self.assertTrue(value["qualified"], value["reasons"])

    def test_service_hard_gate_rejects_partial_state_or_output_mismatch(self):
        value = measurement("control")
        value["partial_cases"][0]["partial"]["cached_tokens"] = 0
        value["partial_cases"][1]["warm"]["output_sha256"] = digest("wrong")
        result = SERVICE.evaluate(value)
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "partial cached-token boundary differs" in reason
            for reason in result["reasons"]
        ))
        self.assertTrue(any(
            "partial/warm output differs" in reason
            for reason in result["reasons"]
        ))

    def test_pair_qualifies_with_material_p90_and_partial_gain(self):
        result = qualify_pair(
            measurement("control"),
            measurement("candidate", speedup=1.2),
        )
        self.assertTrue(result["qualified"], result)
        self.assertGreaterEqual(
            result["aggregate"]["uncached_ttft_p90_speedup"], 1.08)
        self.assertEqual(
            result["cross_arm_output_diagnostics"][
                "cold_output_matches"],
            0,
        )
        self.assertTrue(
            result["authorization"][
                "long_context_confirmation_authorized"])

    def test_pair_rejects_no_end_to_end_gain_without_calling_it_invalid(self):
        result = qualify_pair(
            measurement("control"),
            measurement("candidate"),
        )
        self.assertFalse(result["qualified"])
        self.assertFalse(result["invalid_reasons"])
        self.assertTrue(result["performance_reasons"])

    def test_cross_arm_prompt_or_artifact_mismatch_is_invalid(self):
        control = measurement("control")
        candidate = measurement("candidate", speedup=1.2)
        candidate["partial_cases"][0]["partial_prompt_sha256"] = digest(
            "different")
        candidate["evaluation"] = SERVICE.evaluate(candidate)
        shas = {
            "control": digest(json.dumps(control, sort_keys=True)),
            "candidate": digest(json.dumps(candidate, sort_keys=True)),
        }
        statuses = {
            "control": status("control", shas["control"]),
            "candidate": status("candidate", shas["candidate"]),
        }
        statuses["candidate"]["p90_operator_authorization"][
            "candidate_extension_sha256"] = "0" * 64
        result = QUALIFIER.qualify(
            statuses,
            {"control": control, "candidate": candidate},
            shas,
            CONTRACT,
            contract_sha256=CONTRACT_SHA,
        )
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "cross-arm prompt differs" in reason
            for reason in result["invalid_reasons"]
        ))
        self.assertTrue(any(
            "runner identity or lifecycle differs" in reason
            for reason in result["invalid_reasons"]
        ))

    def test_contract_digest_is_frozen(self):
        result = QUALIFIER.qualify(
            {"control": {}, "candidate": {}},
            {"control": {}, "candidate": {}},
            {"control": "0" * 64, "candidate": "0" * 64},
            copy.deepcopy(CONTRACT),
            contract_sha256="0" * 64,
        )
        self.assertFalse(result["qualified"])
        self.assertIn(
            "P90 pair contract differs",
            result["invalid_reasons"],
        )

    def test_operator_authorization_binds_extension_and_kernel(self):
        with tempfile.TemporaryDirectory(
                prefix="m1-152-p90-", dir="/tmp") as temporary:
            root = Path(temporary)
            identity = {
                "extension_sha256": "b" * 64,
                "kernel_source_sha256": "e" * 64,
            }
            (root / "identity.json").write_text(
                json.dumps(identity) + "\n", encoding="ascii")
            status_value = {
                "schema": "bi100-m1-149-ttft-p90-prefill-grid-v1",
                "version": 1,
                "qualified": True,
                "source_revision": "a" * 40,
                "extension_sha256": "b" * 64,
                "fixed_cases": [
                    f"p90_total_{total // 1024:02d}k_q8176"
                    for total in range(8192, 65537, 8192)
                ],
                "gpu_count": 3,
                "gpus": [1, 2, 3],
                "screen": {
                    "qualified": True,
                    "reasons": [],
                    "minimum_speedup": 1.9,
                    "median_speedup": 2.2,
                    "rows": [
                        {
                            "case": (
                                f"p90_total_{total // 1024:02d}k_q8176"),
                            "total_kv_len": total - 16,
                            "qualified": True,
                            "finite": True,
                            "speedup": 2.0,
                            "output_relative_l2": 6.0e-6,
                            "lse_relative_l2": 2.0e-8,
                            "output_max_abs": 2.0e-4,
                        }
                        for total in range(8192, 65537, 8192)
                    ],
                },
                "lifecycle": {
                    "after_preflight_qualified": True,
                    "cleanup_reaped": True,
                    "fatal_scan_qualified": True,
                    "postflight_qualified": True,
                    "preflight_comparison_qualified": True,
                    "source_unchanged": True,
                },
                "privacy": {
                    "credentials_recorded": False,
                    "model_outputs_recorded": False,
                    "prompts_recorded": False,
                    "token_ids_recorded": False,
                },
                "authorization": {
                    "short_tp4_p90_screen_authorized": True,
                    "l2_capture_authorized": False,
                    "main_or_yaml_change_authorized": False,
                    "official_score_claim_authorized": False,
                },
            }
            status_path = root / "runner_status.json"
            status_path.write_text(
                json.dumps(status_value) + "\n", encoding="ascii")
            status_path.chmod(0o600)
            (root / "identity.json").chmod(0o600)
            result = RUNNER.validate_p90_operator_authorization(
                status_path,
                candidate_extension_sha256="b" * 64,
                kernel_source_sha256="e" * 64,
            )
            self.assertEqual(result["case_count"], 8)
            self.assertEqual(result["minimum_speedup"], 1.9)


if __name__ == "__main__":
    unittest.main()
