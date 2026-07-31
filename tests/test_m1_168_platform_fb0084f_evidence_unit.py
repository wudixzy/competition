import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs" / "experiments" / "evidence"
    / "M1_168_PLATFORM_FB0084F_20260801"
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


class M1168PlatformFb0084fEvidenceUnitTest(unittest.TestCase):

    def test_comparison_does_not_overclaim_attribution(self):
        value = _load("comparison.json")
        outcomes = value["request_outcomes"]
        performance = value["performance"]
        attribution = value["code_attribution"]
        interpretation = value["interpretation"]

        self.assertFalse(outcomes["failure_population_changed"])
        self.assertEqual(outcomes["candidate_errors"], 250)
        self.assertEqual(outcomes["baseline_errors"], 250)
        self.assertEqual(outcomes["tool_4xx_candidate"], 226)
        self.assertGreater(
            performance["candidate_vs_baseline_percent"]["ttft_p90"], 0)
        self.assertLess(
            performance["candidate_vs_baseline_percent"]["output_tps_p10"],
            0,
        )
        self.assertFalse(attribution["max_completion_tokens_fix_present"])
        self.assertFalse(
            attribution["capture_boundary_performance_causality_proven"])
        self.assertTrue(interpretation["performance_regression_observed"])
        self.assertFalse(
            interpretation["performance_regression_code_cause_proven"])

    def test_incomplete_success_population_blocks_score_comparison(self):
        qualification = _load("comparison.json")["qualification"]
        self.assertAlmostEqual(
            qualification["request_success_rate"], 631 / 881)
        self.assertTrue(qualification["timed_out"])
        self.assertFalse(qualification["performance_population_complete"])
        self.assertAlmostEqual(
            qualification["diagnostic_weighted_linear"], 4769.580193333333)
        self.assertFalse(qualification["official_score_comparable"])

    def test_incomplete_excerpt_is_not_promoted_to_complete_attribution(self):
        value = _load("excerpt_4xx_summary.json")
        self.assertFalse(value["qualified"])
        self.assertFalse(value["complete"])
        self.assertFalse(value["classified"])
        self.assertEqual(value["attributed_count"], 2)
        self.assertEqual(value["chat_4xx_access_count"], 0)

    def test_runtime_probe_matches_repository_behavior(self):
        value = _load("request_stage_runtime_probe.json")
        self.assertTrue(value["qualified"])
        self.assertFalse(value["logger_failure_raised"])
        self.assertEqual(value["reasons"], {
            "chat_template": "chat_template_failed",
            "multimodal": "multimodal_load_failed",
        })
        self.assertEqual(
            value["files"]["api_server.py"]["sha256"],
            _sha256_bytes(
                (ROOT / "qwen3_6_scripts" / "api_server.py").read_bytes()),
        )
        serving_bytes = (
            ROOT / "qwen3_6_scripts" / "serving_chat.py").read_bytes()
        self.assertEqual(
            value["files"]["serving_chat.py"]["sha256"],
            _sha256_bytes(serving_bytes.replace(b"\r\n", b"\n")),
        )

    def test_manifest_covers_json_evidence(self):
        observed = {}
        for line in (EVIDENCE / "SHA256SUMS").read_text(
                encoding="ascii").splitlines():
            digest, name = line.split("  ", 1)
            observed[name] = digest
        expected = {
            path.name: _sha256_bytes(path.read_bytes())
            for path in EVIDENCE.glob("*.json")
        }
        self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
