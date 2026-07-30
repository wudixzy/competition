from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests" / "qualify_m1_141_short_tp4_pair.py"
SPEC = importlib.util.spec_from_file_location("qualify_m1_141_pair", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
CONTRACT_PATH = ROOT / "quality" / "short_tp4_pair.v1.json"
CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="ascii"))
CONTRACT_SHA = hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def response(
    target: int,
    *,
    warm: bool,
    ttft_s: float,
    output_label: str,
) -> dict:
    return {
        "ok": True,
        "elapsed_s": ttft_s + 0.1,
        "ttft_s": ttft_s,
        "last_output_s": ttft_s + 0.05,
        "decode_window_s": 0.05,
        "output_tps": 160.0,
        "prompt_tokens": target,
        "cached_tokens": target - 16 if warm else 0,
        "completion_tokens": 8,
        "finish_reason": "length",
        "first_token_sha256": digest(output_label + ":first"),
        "output_sha256": digest(output_label + ":output"),
    }


def measurement(selector: str, *, regression: float = 0.0) -> dict:
    cases = []
    for target_index, target in enumerate(MODULE.TARGETS):
        for repetition in range(MODULE.REPETITIONS):
            control_ttft = 1.0 + target_index + repetition * 0.05
            cold_ttft = control_ttft * (1.0 + regression)
            output_label = f"{selector}:{target}:{repetition}"
            cases.append({
                "target_prompt_tokens": target,
                "repetition": repetition,
                "prompt_sha256": digest(
                    f"shared:{target}:{repetition}"),
                "cold": response(
                    target,
                    warm=False,
                    ttft_s=cold_ttft,
                    output_label=output_label,
                ),
                "warm": response(
                    target,
                    warm=True,
                    ttft_s=0.2 + repetition * 0.01,
                    output_label=output_label,
                ),
            })
    return {
        "schema": MODULE.MEASUREMENT_SCHEMA,
        "version": 1,
        "run_id": f"run-{selector}",
        "prompt_set_id": "shared-pair",
        "selector": selector,
        "targets": list(MODULE.TARGETS),
        "max_tokens": 8,
        "repetitions": MODULE.REPETITIONS,
        "elapsed_s": 10.0,
        "qualified": True,
        "reasons": [],
        "cases": cases,
        "cold_ttft_median_s": statistics.median(
            case["cold"]["ttft_s"] for case in cases),
        "warm_ttft_median_s": statistics.median(
            case["warm"]["ttft_s"] for case in cases),
        "privacy": dict(MODULE.MEASUREMENT_PRIVACY),
        "authorization": dict(MODULE.MEASUREMENT_AUTHORIZATION),
    }


def status(selector: str, measurement_sha: str) -> dict:
    l2 = {
        "qualification_sha256": digest("l2-qualification"),
        "runner_status_sha256": digest("l2-runner"),
        "candidate_extension_sha256": "b" * 64,
        "candidate_extension_size_bytes": 247176,
        "capture_source_revision": "a" * 40,
        "replay_source_revision": "c" * 40,
        "runtime_identity": "capture-runtime",
        "activation_run_id": "capture-run",
    }
    artifacts = {
        name: digest(f"{selector}:{name}")
        for name in MODULE.REQUIRED_ARTIFACTS
    }
    artifacts["measurement.json"] = measurement_sha
    return {
        "schema": MODULE.RUNNER_SCHEMA,
        "version": 1,
        "qualified": True,
        "returncode": 0,
        "terminal_stage": "complete",
        "error_type": None,
        "run_id": f"run-{selector}",
        "source_revision": "d" * 40,
        "source_branch": "",
        "instance": "instance",
        "selector": selector,
        "pair_id": "shared-pair",
        "runtime_identity": "bare-host-overlay-v1:1234567890",
        "gpu_count": 4,
        "tensor_parallel_size": 4,
        "service_startups": 1,
        "targets": list(MODULE.TARGETS),
        "repetitions": MODULE.REPETITIONS,
        "gates": {
            name: 0 for name in MODULE.REQUIRED_GATES
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
                "sha256": "b" * 64,
                "size_bytes": 247176,
                "external_override_active": True,
            }
        ),
        "dispatch_count": 0 if selector == "control" else 4,
        "kernel_source_sha256": "e" * 64,
        "l2_authorization": l2,
        "timing": {
            "wall_span_s": 600.0,
            "summed_stage_s": 600.0,
        },
        "authorization": dict(MODULE.RUNNER_AUTHORIZATION),
        "privacy": dict(MODULE.RUNNER_PRIVACY),
    }


def qualify_pair(
    control: dict,
    candidate: dict,
) -> dict:
    shas = {
        "control": digest(json.dumps(control, sort_keys=True)),
        "candidate": digest(json.dumps(candidate, sort_keys=True)),
    }
    return MODULE.qualify(
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


class QualifyM1141ShortTp4PairTest(unittest.TestCase):

    def test_pair_qualifies_without_requiring_cross_arm_output_identity(self):
        result = qualify_pair(
            measurement("control"),
            measurement("candidate", regression=-0.1),
        )
        self.assertTrue(result["qualified"], result)
        self.assertGreater(
            result["overall"]["candidate_cold_speedup"], 1.0)
        self.assertEqual(
            result["cross_arm_output_diagnostics"][
                "output_match_count"],
            0,
        )
        self.assertTrue(
            result["authorization"]["long_context_authorized"])

    def test_material_ttft_regression_fails_performance_only(self):
        result = qualify_pair(
            measurement("control"),
            measurement("candidate", regression=0.2),
        )
        self.assertFalse(result["qualified"])
        self.assertFalse(result["invalid_reasons"])
        self.assertTrue(result["performance_reasons"])

    def test_cache_transparency_mismatch_is_a_hard_failure(self):
        control = measurement("control")
        candidate = measurement("candidate")
        candidate["cases"][0]["warm"]["output_sha256"] = digest("wrong")
        result = qualify_pair(control, candidate)
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "cold/warm output_sha256 differs" in reason
            for reason in result["invalid_reasons"]
        ))

    def test_cross_arm_prompt_or_artifact_mismatch_fails(self):
        control = measurement("control")
        candidate = measurement("candidate")
        candidate["cases"][0]["prompt_sha256"] = digest("different")
        shas = {
            "control": digest(json.dumps(control, sort_keys=True)),
            "candidate": digest(json.dumps(candidate, sort_keys=True)),
        }
        statuses = {
            "control": status("control", shas["control"]),
            "candidate": status("candidate", shas["candidate"]),
        }
        statuses["candidate"]["candidate_extension"]["sha256"] = "0" * 64
        result = MODULE.qualify(
            statuses,
            {"control": control, "candidate": candidate},
            shas,
            CONTRACT,
            contract_sha256=CONTRACT_SHA,
        )
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "prompt identity differs" in reason
            for reason in result["invalid_reasons"]
        ))
        self.assertTrue(any(
            "runner identity or lifecycle differs" in reason
            for reason in result["invalid_reasons"]
        ))

    def test_contract_digest_is_frozen(self):
        result = MODULE.qualify(
            {
                "control": {},
                "candidate": {},
            },
            {
                "control": {},
                "candidate": {},
            },
            {
                "control": "0" * 64,
                "candidate": "0" * 64,
            },
            copy.deepcopy(CONTRACT),
            contract_sha256="0" * 64,
        )
        self.assertFalse(result["qualified"])
        self.assertIn(
            "short TP4 pair contract differs",
            result["invalid_reasons"],
        )


if __name__ == "__main__":
    unittest.main()
