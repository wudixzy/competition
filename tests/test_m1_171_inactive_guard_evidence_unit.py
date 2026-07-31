import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs" / "experiments" / "evidence"
    / "M1_171_INACTIVE_PREFILL_GUARD_20260801"
)


class M1171InactiveGuardEvidenceUnitTest(unittest.TestCase):

    def test_measurement_closes_platform_regression_attribution(self):
        value = json.loads(
            (EVIDENCE / "result.json").read_text(encoding="ascii"))
        self.assertEqual(value["iterations"], 100000)
        self.assertGreater(value["guard_net_ns_per_call"], 0)
        self.assertLess(value["hypothetical_64_call_upper_bound_ms"], 1.0)
        self.assertTrue(value["finite"])
        evaluation = value["evaluation"]
        self.assertFalse(evaluation["platform_regression_explained"])
        self.assertFalse(evaluation["service_ab_authorized"])
        self.assertFalse(evaluation["tp4_authorized"])
        self.assertFalse(evaluation["main_or_yaml_change_authorized"])

    def test_manifest_and_privacy_boundary(self):
        result = EVIDENCE / "result.json"
        expected = hashlib.sha256(result.read_bytes()).hexdigest()
        digest, name = (EVIDENCE / "SHA256SUMS").read_text(
            encoding="ascii").strip().split("  ", 1)
        self.assertEqual(name, "result.json")
        self.assertEqual(digest, expected)
        value = json.loads(result.read_text(encoding="ascii"))
        self.assertTrue(all(flag is False for flag in value["privacy"].values()))


if __name__ == "__main__":
    unittest.main()
