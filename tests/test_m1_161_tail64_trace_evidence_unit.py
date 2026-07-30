import hashlib
import json
from pathlib import Path
import unittest


ROOT = (
    Path(__file__).parents[1]
    / "docs"
    / "experiments"
    / "evidence"
    / "M1_161_TAIL64_P90_TRACE_20260730"
)


class M1161Tail64TraceEvidenceTest(unittest.TestCase):
    def _load(self, name, expected_sha256):
        path = ROOT / name
        payload = path.read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), expected_sha256)
        report = json.loads(payload)
        self.assertEqual(
            report["schema"], "bi100-tail64-trace-diagnostic-v1")
        self.assertEqual(report["analysis_mode"], "tail64_pair_only")
        self.assertEqual(
            set(report["policy_metrics"]), {"admission64", "tail64"})
        self.assertFalse(
            report["promotion"]["main_or_yaml_change_authorized"])
        self.assertFalse(
            report["promotion"]["default_policy_change_authorized"])
        self.assertFalse(
            report["promotion"]["official_score_claim_authorized"])
        return report

    def test_partial_32k_directional_result(self):
        report = self._load(
            "partial32_pair.json",
            "be86ba84e2ef9bc9dd7dac82631f0384562e0610e17095f32500275d4375bf6e",
        )
        self.assertEqual(report["requests"], 4)
        self.assertEqual(
            report["request_delta_counts"],
            {"improved": 1, "unchanged": 3, "regressed": 0},
        )
        self.assertAlmostEqual(
            report["delta"]["effective_hit_gain_percentage_points"], 18.75)
        self.assertGreater(
            report["delta"]["projected_ttft_p90_reduction_fraction"], 0.26)

    def test_full_quality_trace_is_not_promotion_evidence(self):
        report = self._load(
            "full12_pair.json",
            "9f0782ae38435cfe12226481a0d793ff12481d07ca6720accf8f4ff6301718c1",
        )
        self.assertEqual(report["requests"], 31)
        self.assertEqual(
            report["request_delta_counts"],
            {"improved": 2, "unchanged": 29, "regressed": 0},
        )
        self.assertGreater(
            report["delta"]["effective_hit_gain_percentage_points"], 7.0)
        self.assertLess(
            report["delta"]["projected_ttft_p90_reduction_fraction"], 0.01)
        self.assertEqual(
            report["length_bucket_metrics"]["32k_64k"]
            ["regressed_requests"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
