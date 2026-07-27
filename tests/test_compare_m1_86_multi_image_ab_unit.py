from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tests" / "compare_m1_86_multi_image_ab.py"
SPEC = importlib.util.spec_from_file_location(
    "compare_m1_86_multi_image_ab", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def generation(
    semantic: str,
    *,
    cached_tokens: int = 0,
    exact: bool = False,
    isolated: bool = False,
) -> dict:
    value = {
        "http_status": 200,
        "semantic_output_sha256": semantic,
        "finish_reason": "stop",
        "prompt_tokens": 200,
        "completion_tokens": 1,
        "cached_tokens": cached_tokens,
        "has_content": True,
        "has_reasoning_content": False,
        "tool_call_count": 0,
    }
    if exact:
        value["cold_generation_exact"] = True
    if isolated:
        value["content_specific_prefix_isolated"] = True
    return value


def report(candidate: bool) -> dict:
    cases = [
        {
            "name": "models_262144_contract",
            "ok": True,
            "evidence": {
                "http_status": 200,
                "served_model": "llm",
                "max_model_len": 262144,
            },
        },
        {
            "name": "stream_one_image_cold",
            "ok": True,
            "evidence": generation("one"),
        },
        {
            "name": "stream_two_images_cold",
            "ok": True,
            "evidence": (
                generation("two")
                if candidate else
                {"http_status": 400, "response_sha256": "a" * 64}
            ),
        },
        {
            "name": "stream_two_images_warm",
            "ok": True,
            "evidence": (
                generation("two", cached_tokens=192, exact=True)
                if candidate else
                {"skipped": True, "reason": "control_image_limit_one"}
            ),
        },
        {
            "name": "stream_two_images_reversed",
            "ok": True,
            "evidence": (
                {
                    **generation("reversed", cached_tokens=16),
                    "cache_isolation_deferred_to_trace": True,
                }
                if candidate else
                {"skipped": True, "reason": "control_image_limit_one"}
            ),
        },
        {
            "name": "stream_two_images_reversed_warm",
            "ok": True,
            "evidence": (
                generation("reversed", cached_tokens=192, exact=True)
                if candidate else
                {"skipped": True, "reason": "control_image_limit_one"}
            ),
        },
        {
            "name": "post_request_health",
            "ok": True,
            "evidence": {"http_status": 200, "response_sha256": "b" * 64},
        },
    ]
    return {
        "schema": "qwen36-diagnostic-multi-image-http-gate-v1",
        "version": 1,
        "qualified": True,
        "case_count": 7,
        "config": {
            "expected_two_image_status": 200 if candidate else 400,
            "stream": True,
            "temperature": 0,
            "seed": 20260728,
            "max_tokens": 8,
            "thinking": False,
        },
        "cases": cases,
        "privacy": {
            "contains_raw_request": False,
            "contains_raw_response": False,
            "contains_image_url_or_bytes": False,
            "contains_prompt_or_generated_text": False,
            "contains_credentials": False,
            "synthetic_images_only": True,
        },
        "semantic_quality_evaluated": False,
        "full_model_evaluated": False,
        "production_promotion_authorized": False,
    }


def attribution(candidate: bool) -> dict:
    count = 0 if candidate else 1
    return {
        "schema": "bi100-api-4xx-attribution-v3",
        "qualified": True,
        "complete": True,
        "classified": True,
        "chat_4xx_access_count": count,
        "attributed_count": count,
        "attribution_delta": 0,
        "by_reason": {} if candidate else {"image_count_limit": 1},
        "request_shapes": [] if candidate else [{
            "count": 1,
            "images": 2,
            "image_data": 2,
            "image_remote": 0,
            "image_other": 0,
            "stream": 1,
        }],
    }


def digests() -> dict[str, str]:
    names = (
        "control_report",
        "candidate_report",
        "control_attribution",
        "candidate_attribution",
        "control_status",
        "candidate_status",
        "control_contract",
        "candidate_contract",
        "control_capacity",
        "candidate_capacity",
        "control_trace",
        "candidate_trace",
        "control_process_group",
        "candidate_process_group",
    )
    return {
        name: hashlib.sha256(name.encode("ascii")).hexdigest()
        for name in names
    }


def status(candidate: bool, artifact_digests: dict[str, str]) -> dict:
    label = "candidate" if candidate else "control"
    return {
        "schema": "bi100-m1-86-multi-image-arm-v1",
        "version": 1,
        "qualified": True,
        "returncode": 0,
        "image_limit": 2 if candidate else 1,
        "gates": {
            "preflight_before": 0,
            "port_preflight": 0,
            "service_contract": 0,
            "process_group": 0,
            "startup": 0,
            "probe": 0,
            "capacity": 0,
            "cleanup": 0,
            "cache_trace": 0,
            "attribution": 0,
            "fatal_scan": 0,
            "service_postflight": 0,
            "preflight_after": 0,
            "preflight_comparison": 0,
        },
        "artifact_sha256": {
            "probe": artifact_digests[f"{label}_report"],
            "attribution": artifact_digests[f"{label}_attribution"],
            "capacity": artifact_digests[f"{label}_capacity"],
            "cache_trace": artifact_digests[f"{label}_trace"],
            "service_contract": artifact_digests[f"{label}_contract"],
            "process_group_identity":
                artifact_digests[f"{label}_process_group"],
            "service_postflight": "a" * 64,
            "preflight_comparison": "b" * 64,
        },
    }


def contract(candidate: bool) -> dict:
    command = [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "/model",
    ]
    if candidate:
        command += ["--limit-mm-per-prompt", "image=2"]
    return {
        "schema": "bi100-m1-86-service-contract-v1",
        "version": 1,
        "source_revision": "1" * 40,
        "source_branch": "test/M1-86",
        "runtime_tree_sha256": "2" * 64,
        "runtime_install_sha256": "3" * 64,
        "model_path": "/model",
        "model_manifest_sha256": "4" * 64,
        "environment": {
            "BI100_ATTN_COREX_PAGED_GATHER": "1",
            "BI100_BLOCK_MAJOR_CPU_KV": "0",
            "BI100_CPU_KV_OFFLOAD": "0",
            "BI100_GDN_CACHE_POLICY": "fine32",
            "BI100_GDN_COMBINED_QK_NORM": "0",
            "BI100_GDN_COREX_PACKED_DECODE": "0",
            "BI100_GDN_RESTORE_MODE": "direct",
            "BI100_HYBRID_KV_ACCOUNTING": "full_attention",
            "BI100_MOE_COREX_DIRECT_ROUTED": "0",
            "BI100_PREFIX_DTYPE": "float16",
            "BI100_PREFIX_MODEL_FINGERPRINT":
                "Qwen3.6-35B-A3B-diagnostic-4L-real",
            "BI100_PREFIX_TP_SIZE": "1",
        },
        "command": command,
        "tensor_parallel_size": 1,
        "max_model_len": 262144,
        "image_limit": 2 if candidate else 1,
        "runtime_source_files_match": True,
        "semantic_quality_evaluated": False,
        "production_promotion_authorized": False,
    }


def capacity(blocks: int = 20000) -> dict:
    return {
        "qualified": True,
        "max_model_len_required": 262144,
        "required_gpu_blocks": 16384,
        "observed_gpu_blocks": blocks,
    }


def trace(candidate: bool) -> dict:
    return {
        "schema": "bi100-m1-86-multi-image-trace-v1",
        "version": 1,
        "qualified": True,
        "mode": "candidate" if candidate else "control",
        "trace_version": 4,
        "trace_count": 5 if candidate else 1,
        "content_isolation": (
            {
                "normal_initial_prior_common_blocks": 0,
                "reversed_initial_prior_common_blocks": 0,
                "normal_reversed_common_blocks": 0,
            }
            if candidate else {}
        ),
        "privacy": {
            "contains_raw_tokens": False,
            "contains_raw_images": False,
            "contains_raw_prompt_or_output": False,
            "contains_request_id": False,
            "contains_credentials": False,
        },
        "semantic_quality_evaluated": False,
        "production_promotion_authorized": False,
    }


def process_group(candidate: bool) -> dict:
    pid = 2002 if candidate else 1001
    return {
        "schema": "bi100-process-session-v1",
        "version": 1,
        "pid": pid,
        "pgid": pid,
        "sid": pid,
    }


def compare(**overrides) -> dict:
    artifact_digests = digests()
    values = {
        "control_report": report(False),
        "candidate_report": report(True),
        "control_attribution": attribution(False),
        "candidate_attribution": attribution(True),
        "control_status": status(False, artifact_digests),
        "candidate_status": status(True, artifact_digests),
        "control_contract": contract(False),
        "candidate_contract": contract(True),
        "control_capacity": capacity(),
        "candidate_capacity": capacity(19800),
        "control_trace": trace(False),
        "candidate_trace": trace(True),
        "control_process_group": process_group(False),
        "candidate_process_group": process_group(True),
        "artifact_sha256": artifact_digests,
    }
    values.update(overrides)
    return MODULE.compare(**values)


class CompareM186MultiImageAbUnitTest(unittest.TestCase):

    def test_fixed_candidate_qualifies_without_authorizing_default(self):
        value = compare()
        self.assertTrue(value["qualified"])
        self.assertEqual(value["reasons"], [])
        self.assertEqual(
            value["observed"]["candidate_control_gpu_block_ratio"], 0.99)
        self.assertTrue(
            value["decision"]["single_gpu_diagnostic_phase_passed"])
        self.assertFalse(
            value["decision"]["default_image_limit_change_authorized"])
        self.assertFalse(
            value["decision"]["main_or_yaml_change_authorized"])

    def test_one_image_output_drift_fails(self):
        candidate = report(True)
        candidate["cases"][1]["evidence"]["semantic_output_sha256"] = "drift"
        value = compare(candidate_report=candidate)
        self.assertFalse(value["qualified"])
        self.assertIn(
            "one-image deterministic output differs across arms",
            value["reasons"],
        )

    def test_incomplete_4xx_reason_fails(self):
        control = attribution(False)
        control["by_reason"] = {"unclassified_chat_error": 1}
        value = compare(control_attribution=control)
        self.assertFalse(value["qualified"])
        self.assertIn(
            "control 4xx reason is not exactly image_count_limit",
            value["reasons"],
        )

    def test_privacy_contract_must_fail_closed(self):
        candidate = report(True)
        candidate["privacy"]["contains_image_url_or_bytes"] = True
        value = compare(candidate_report=candidate)
        self.assertFalse(value["qualified"])
        self.assertIn(
            "candidate privacy contract differs",
            value["reasons"],
        )

    def test_unrelated_command_delta_fails(self):
        candidate = contract(True)
        candidate["command"].insert(-2, "--enforce-eager")
        value = compare(candidate_contract=candidate)
        self.assertFalse(value["qualified"])
        self.assertIn(
            "service commands differ outside the image limit",
            value["reasons"],
        )

    def test_capacity_loss_over_two_percent_fails(self):
        value = compare(candidate_capacity=capacity(19000))
        self.assertFalse(value["qualified"])
        self.assertIn(
            "candidate image budget loses more than 2% GPU blocks",
            value["reasons"],
        )

    def test_postflight_failure_invalidates_result(self):
        candidate = status(True, digests())
        candidate["gates"]["service_postflight"] = 1
        candidate["qualified"] = False
        candidate["returncode"] = 1
        value = compare(candidate_status=candidate)
        self.assertFalse(value["qualified"])
        self.assertIn(
            "candidate arm lifecycle did not qualify",
            value["reasons"],
        )

    def test_missing_lifecycle_gate_invalidates_result(self):
        candidate = status(True, digests())
        del candidate["gates"]["service_contract"]
        value = compare(candidate_status=candidate)
        self.assertFalse(value["qualified"])
        self.assertIn(
            "candidate arm lifecycle did not qualify",
            value["reasons"],
        )

    def test_wrong_arm_image_limit_invalidates_result(self):
        candidate = status(True, digests())
        candidate["image_limit"] = 1
        value = compare(candidate_status=candidate)
        self.assertFalse(value["qualified"])
        self.assertIn(
            "candidate arm lifecycle did not qualify",
            value["reasons"],
        )

    def test_cross_arm_artifact_reuse_invalidates_result(self):
        artifact_digests = digests()
        candidate = status(True, artifact_digests)
        candidate["artifact_sha256"]["probe"] = (
            artifact_digests["control_report"]
        )
        value = compare(candidate_status=candidate)
        self.assertFalse(value["qualified"])
        self.assertIn(
            "candidate arm artifact binding differs",
            value["reasons"],
        )

    def test_unqualified_cache_trace_invalidates_result(self):
        candidate = trace(True)
        candidate["qualified"] = False
        value = compare(candidate_trace=candidate)
        self.assertFalse(value["qualified"])
        self.assertIn(
            "candidate cache trace did not qualify",
            value["reasons"],
        )

    def test_process_session_must_be_isolated(self):
        candidate = process_group(True)
        candidate["pgid"] = candidate["pid"] + 1
        value = compare(candidate_process_group=candidate)
        self.assertFalse(value["qualified"])
        self.assertIn(
            "candidate process session identity differs",
            value["reasons"],
        )


if __name__ == "__main__":
    unittest.main()
