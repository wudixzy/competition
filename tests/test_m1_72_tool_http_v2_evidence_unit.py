from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = ROOT / "docs" / "experiments" / "evidence"
EVIDENCE = EVIDENCE_ROOT / "M1_72_TOOL_HTTP_V2_QUALIFIED_20260728"
V1_EVIDENCE = (
    EVIDENCE_ROOT / "M1_72_TOOL_HTTP_V1_TELEMETRY_GAP_20260728"
)
MANIFEST_SHA256 = (
    "539b427c9176c00c22e599fbc61eca95bbbcf62cc1992ef836e8b952f78ce1d5"
)
ARMS = ("baseline", "candidate")
GENERATED_CASES = (
    "function_tool_default",
    "function_tool_strict_false",
    "tool_arguments_json_string",
    "tool_arguments_json_object",
)
GENERATION_FIELDS = (
    "message_sha256",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "has_content",
    "has_reasoning_content",
    "tool_call_count",
)


def load(name: str, root: Path = EVIDENCE) -> dict:
    return json.loads((root / name).read_text(encoding="utf-8"))


def cases(arm: str, root: Path = EVIDENCE) -> dict[str, dict]:
    report = load(f"{arm}/probe.json", root)
    return {row["name"]: row for row in report["cases"]}


def generation_contract(case: dict) -> tuple:
    evidence = case["evidence"]
    return tuple(evidence.get(name) for name in GENERATION_FIELDS)


class M172ToolHttpV2EvidenceTest(unittest.TestCase):

    def test_manifest_binds_every_evidence_file(self):
        manifest_path = EVIDENCE / "SHA256SUMS"
        self.assertEqual(
            hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            MANIFEST_SHA256,
        )
        manifest = {}
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            digest, relative = line.split(maxsplit=1)
            manifest[relative.removeprefix("./")] = digest
        actual_files = {
            path.relative_to(EVIDENCE).as_posix()
            for path in EVIDENCE.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        }
        self.assertEqual(set(manifest), actual_files)
        for relative, expected in manifest.items():
            self.assertEqual(
                hashlib.sha256((EVIDENCE / relative).read_bytes()).hexdigest(),
                expected,
                relative,
            )

    def test_runner_and_all_http_gates_qualified(self):
        qualification = load("qualification.json")
        runner = load("runner_status.json")
        comparison = load("comparison.json")
        self.assertTrue(qualification["qualified"])
        self.assertTrue(all(qualification["behavior"].values()))
        self.assertTrue(runner["qualified"])
        self.assertEqual(runner["returncode"], 0)
        self.assertEqual(runner["terminal_stage"], "completed")
        self.assertTrue(all(value == 0 for value in runner["gates"].values()))
        self.assertTrue(comparison["qualified"])
        self.assertEqual(comparison["reasons"], [])
        self.assertTrue(all(comparison["checks"].values()))
        self.assertFalse(runner["full_model_evaluated"])
        self.assertFalse(runner["semantic_quality_evaluated"])
        self.assertFalse(runner["performance_evaluated"])
        self.assertFalse(runner["production_promotion_authorized"])

    def test_expected_compatibility_statuses_and_exact_generation(self):
        baseline = cases("baseline")
        candidate = cases("candidate")
        self.assertTrue(all(row["ok"] for row in baseline.values()))
        self.assertTrue(all(row["ok"] for row in candidate.values()))
        self.assertEqual(
            candidate["models_262144_contract"]["evidence"][
                "max_model_len"],
            262144,
        )

        for name in (
                "function_tool_strict_false",
                "tool_arguments_json_object"):
            self.assertEqual(
                baseline[name]["evidence"]["http_status"], 400, name)
            self.assertEqual(
                candidate[name]["evidence"]["http_status"], 200, name)
        self.assertTrue(
            candidate["function_tool_strict_false"]["evidence"][
                "default_generation_exact"])
        self.assertTrue(
            candidate["tool_arguments_json_object"]["evidence"][
                "string_generation_exact"])

        for name in (
                "tool_arguments_invalid_json_400",
                "function_tool_strict_true_400",
                "tool_choice_required_400"):
            self.assertEqual(
                baseline[name]["evidence"]["http_status"], 400, name)
            self.assertEqual(
                candidate[name]["evidence"]["http_status"], 400, name)
        self.assertEqual(
            candidate["post_4xx_health"]["evidence"]["http_status"], 200)

    def test_successful_tool_generation_is_exact_across_v1_and_v2(self):
        current = cases("candidate")
        prior = cases("candidate", V1_EVIDENCE)
        self.assertEqual(set(current), set(prior))
        for name in current:
            self.assertEqual(
                current[name]["evidence"]["http_status"],
                prior[name]["evidence"]["http_status"],
                name,
            )
        for name in GENERATED_CASES:
            with self.subTest(name=name):
                self.assertEqual(
                    current[name]["evidence"]["http_status"], 200)
                self.assertEqual(
                    prior[name]["evidence"]["http_status"], 200)
                self.assertEqual(
                    generation_contract(current[name]),
                    generation_contract(prior[name]),
                )

        qualification = load("qualification.json")
        self.assertTrue(
            qualification["cross_run"][
                "candidate_http_statuses_exact_vs_v1"])
        self.assertTrue(
            qualification["cross_run"][
                "successful_tool_generation_contract_exact_vs_v1"])

    def test_candidate_4xx_attribution_is_exact_and_private(self):
        attribution = load("candidate/attribution.json")
        qualification = load("qualification.json")
        expected = {
            "invalid_tool_arguments_json": 1,
            "request_validation_tool_strict": 1,
            "unsupported_tool_choice_required": 1,
        }
        self.assertTrue(attribution["qualified"])
        self.assertTrue(attribution["complete"])
        self.assertTrue(attribution["classified"])
        self.assertEqual(attribution["chat_4xx_access_count"], 3)
        self.assertEqual(attribution["attributed_count"], 3)
        self.assertEqual(attribution["attribution_delta"], 0)
        self.assertEqual(attribution["malformed_marker_count"], 0)
        self.assertEqual(attribution["by_reason"], expected)
        self.assertEqual(qualification["telemetry"]["by_reason"], expected)
        self.assertEqual(qualification["telemetry"]["unknown_4xx_count"], 0)
        self.assertTrue(all(
            value is False for value in attribution["privacy"].values()))
        self.assertTrue(all(
            value is False for value in qualification["privacy"].values()))

    def test_runtime_install_pair_and_checkpoint_are_bound(self):
        install = load("candidate_install.json")
        runtime = load("runtime_pair.json")
        checkpoint = load("checkpoint_verify.json")
        candidate_revision = (
            "d2eed78371ef78aee36682c2322fb9ea44ebb5f2"
        )
        candidate_tree = (
            "6c69c7a0452f183073308d3a440d51097d512c7af6179350d093e583a79f4511"
        )
        self.assertTrue(install["qualified"])
        self.assertEqual(install["source_revision"], candidate_revision)
        self.assertEqual(install["runtime_tree_sha256"], candidate_tree)
        self.assertTrue(install["source_tree_clean"])
        self.assertFalse(install["system_site_packages_modified"])
        self.assertEqual(install["versions"], {
            "torch": "2.1.0+corex.3.2.3",
            "transformers": "4.55.3",
            "vllm": "0.6.3+corex.3.2.3",
        })
        self.assertTrue(runtime["qualified"])
        self.assertEqual(
            runtime["observed_runtime_file_delta"],
            ["api_server", "chat_utils", "protocol"],
        )
        self.assertEqual(
            runtime["candidate"]["source_revision"], candidate_revision)
        self.assertEqual(
            runtime["candidate"]["runtime_tree_sha256"], candidate_tree)
        self.assertTrue(checkpoint["qualified"])
        self.assertTrue(checkpoint["full_hash_checked"])
        self.assertTrue(checkpoint["tensor_contract_preserved"])
        self.assertEqual(checkpoint["layer_count"], 4)
        self.assertEqual(checkpoint["mtp_weight_count"], 19)

    def test_cleanup_scans_and_gpu_state_qualified(self):
        for path in (
                "fatal_scan.txt",
                "timeout_scan.txt",
                "baseline/fatal_scan.txt",
                "candidate/fatal_scan.txt"):
            self.assertEqual((EVIDENCE / path).read_bytes(), b"", path)

        for arm in ARMS:
            status = load(f"{arm}/status.json")
            postflight = load(f"{arm}/service_postflight.json")
            self.assertTrue(status["qualified"], arm)
            self.assertTrue(postflight["qualified"], arm)
            self.assertEqual(postflight["api_server_pids"], [])
            self.assertEqual(postflight["worker_pids"], [])
            self.assertEqual(postflight["gpu_processes"], [])
            self.assertEqual(
                postflight["settling"]["final_clean_streak"], 3)

        final_postflight = load("final_postflight.json")
        self.assertTrue(final_postflight["qualified"])
        self.assertEqual(final_postflight["api_server_pids"], [])
        self.assertEqual(final_postflight["worker_pids"], [])
        self.assertEqual(final_postflight["gpu_processes"], [])

        initial = load("initial_preflight.json")["results"][0]
        final = load("final_preflight.json")["results"][0]
        self.assertEqual(initial["free"], 34057748480)
        self.assertEqual(final["free"], initial["free"])
        self.assertEqual(initial["total"], final["total"])
        self.assertEqual(final["checksum"], initial["checksum"])
        comparison = load("final_preflight_comparison.json")
        self.assertTrue(comparison["qualified"])
        self.assertEqual(
            comparison["stages"][1][
                "free_memory_drop_from_first_bytes"],
            {"1": 0},
        )

    def test_qualification_does_not_authorize_production_promotion(self):
        decision = load("qualification.json")["decision"]
        self.assertFalse(decision["default_change_authorized"])
        self.assertFalse(decision["main_merge_authorized"])
        self.assertFalse(decision["production_promotion_authorized"])
        self.assertFalse(decision["yaml_change_authorized"])
        self.assertTrue(decision["tp4_full_model_gate_required"])


if __name__ == "__main__":
    unittest.main()
