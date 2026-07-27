from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs" / "experiments" / "evidence"
    / "M1_72_TOOL_HTTP_V1_TELEMETRY_GAP_20260728"
)
MANIFEST_SHA256 = (
    "bb0d9c37a38eb8d73b0527df930b77b2cee96c15d4434d5d09be2600fbe85f47"
)
ARMS = ("baseline", "candidate")


def load(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def cases(arm: str) -> dict[str, dict]:
    report = load(f"{arm}/probe.json")
    return {row["name"]: row for row in report["cases"]}


class M172ToolHttpV1EvidenceTest(unittest.TestCase):

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

    def test_behavior_passed_but_telemetry_failed_closed(self):
        diagnosis = load("diagnosis.json")
        comparison = load("comparison.json")
        runner = load("runner_status.json")
        self.assertFalse(diagnosis["qualified"])
        self.assertTrue(all(diagnosis["behavior"].values()))
        self.assertFalse(diagnosis["telemetry"]["qualified"])
        self.assertEqual(diagnosis["telemetry"]["unknown_4xx_count"], 2)
        self.assertFalse(comparison["qualified"])
        self.assertEqual(comparison["checks"], {
            "candidate_4xx_attribution_qualified": False,
            "object_history_http_fix_qualified": True,
            "strict_false_http_fix_qualified": True,
        })
        self.assertEqual(runner["terminal_stage"], "comparison")
        self.assertEqual(runner["gates"]["comparison"], 1)
        self.assertFalse(runner["production_promotion_authorized"])

    def test_http_behavior_is_exact_and_expected(self):
        baseline = cases("baseline")
        candidate = cases("candidate")
        self.assertTrue(all(row["ok"] for row in baseline.values()))
        self.assertTrue(all(row["ok"] for row in candidate.values()))

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

    def test_unknown_reasons_are_the_only_failed_attribution_contract(self):
        attribution = load("candidate/attribution.json")
        diagnosis = load("diagnosis.json")
        self.assertTrue(attribution["qualified"])
        self.assertTrue(attribution["complete"])
        self.assertEqual(attribution["chat_4xx_access_count"], 3)
        self.assertEqual(attribution["attributed_count"], 3)
        self.assertEqual(
            attribution["by_reason"],
            diagnosis["telemetry"]["observed_by_reason"],
        )
        self.assertEqual(
            diagnosis["telemetry"]["expected_by_reason"], {
                "invalid_tool_arguments_json": 1,
                "request_validation_tool_strict": 1,
                "unsupported_tool_choice_required": 1,
            })
        self.assertTrue(all(
            value is False for value in attribution["privacy"].values()))
        self.assertTrue(all(
            value is False for value in diagnosis["privacy"].values()))

    def test_runtime_cleanup_and_gpu_state_are_intact(self):
        runtime = load("runtime_pair.json")
        checkpoint = load("checkpoint_verify.json")
        self.assertTrue(runtime["qualified"])
        self.assertEqual(
            runtime["observed_runtime_file_delta"],
            ["api_server", "chat_utils", "protocol"],
        )
        self.assertTrue(checkpoint["qualified"])
        self.assertEqual(checkpoint["layer_count"], 4)
        self.assertTrue(checkpoint["tensor_contract_preserved"])

        for path in (
                "fatal_scan.txt",
                "timeout_scan.txt",
                "baseline/fatal_scan.txt",
                "candidate/fatal_scan.txt"):
            self.assertEqual((EVIDENCE / path).read_bytes(), b"", path)

        for arm in ARMS:
            postflight = load(f"{arm}/service_postflight.json")
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
        self.assertEqual(final["checksum"], initial["checksum"])
        comparison = load("final_preflight_comparison.json")
        self.assertTrue(comparison["qualified"])
        self.assertEqual(
            comparison["stages"][1][
                "free_memory_drop_from_first_bytes"],
            {"1": 0},
        )


if __name__ == "__main__":
    unittest.main()
