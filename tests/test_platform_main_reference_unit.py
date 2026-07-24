import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (
    ROOT / "docs/experiments/evidence/PLATFORM_MAIN_REFERENCE_20260724.json"
)


class PlatformMainReferenceUnitTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.value = json.loads(REFERENCE.read_text(encoding="utf-8"))

    def test_unbound_reference_cannot_authorize_attribution_or_promotion(self):
        source = self.value["source"]
        attribution = self.value["attribution"]
        self.assertIsNone(source["source_revision"])
        self.assertIsNone(source["runtime_overlay_sha256"])
        self.assertIsNone(source["official_score"])
        self.assertFalse(attribution["eligible"])
        self.assertFalse(attribution["promotion_authorized"])
        self.assertFalse(
            attribution["comparison_to_latest_candidate_authorized"])

    def test_request_accounting_and_failure_rates_are_consistent(self):
        workload = self.value["workload_881"]
        self.assertEqual(
            workload["attempted_requests"],
            workload["successful_requests"] + workload["error_requests"],
        )
        self.assertAlmostEqual(
            workload["computed_success_rate"],
            workload["successful_requests"] / workload["attempted_requests"],
        )
        keys = (
            ("tool_requests_total", "tool_call_4xx_errors",
             "tool_call_4xx_rate"),
            ("image_requests_total", "image_4xx_errors", "image_4xx_rate"),
            ("multi_system_requests", "multi_system_4xx_errors",
             "multi_system_4xx_rate"),
        )
        for total_key, error_key, rate_key in keys:
            self.assertAlmostEqual(
                workload[rate_key], workload[error_key] / workload[total_key])

    def test_latest_candidate_gap_is_explicit(self):
        topology = self.value["local_topology_snapshot"]
        self.assertFalse(topology["remote_fetch_performed"])
        self.assertGreaterEqual(topology["candidate_unique_commits"], 100)
        self.assertGreaterEqual(topology["changed_files"], 200)
        self.assertNotEqual(
            topology["modelhub_tracking_main"], topology["latest_candidate"])


if __name__ == "__main__":
    unittest.main()
