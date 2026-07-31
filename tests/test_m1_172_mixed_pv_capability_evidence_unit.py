import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs" / "experiments" / "evidence"
    / "M1_172_COREX_MIXED_PV_CAPABILITY_20260801"
)


class M1172MixedPvCapabilityEvidenceUnitTest(unittest.TestCase):

    def test_unsupported_contract_fails_closed(self):
        value = json.loads(
            (EVIDENCE / "result.json").read_text(encoding="ascii"))
        self.assertTrue(value["compile"]["qualified_attempt_succeeded"])
        self.assertEqual(value["cell"]["cublas_status"], 15)
        self.assertEqual(
            value["cell"]["error_type"], "CUBLAS_STATUS_NOT_SUPPORTED")
        for field in (
            "finite_evaluated",
            "numeric_gate_evaluated",
            "repeat_exact_evaluated",
            "speedup_evaluated",
        ):
            self.assertFalse(value["cell"][field])
        evaluation = value["evaluation"]
        self.assertFalse(evaluation["qualified"])
        self.assertTrue(evaluation["route_closed"])
        self.assertFalse(evaluation["three_cell_screen_authorized"])
        self.assertFalse(evaluation["service_or_tp4_authorized"])
        self.assertFalse(evaluation["main_or_yaml_change_authorized"])

    def test_lifecycle_manifest_and_privacy(self):
        result = EVIDENCE / "result.json"
        value = json.loads(result.read_text(encoding="ascii"))
        self.assertTrue(value["lifecycle"]["temporary_tree_removed"])
        self.assertEqual(value["lifecycle"]["experiment_processes_remaining"], 0)
        self.assertTrue(value["lifecycle"]["gpu_postflight_qualified"])
        self.assertFalse(value["lifecycle"]["tp4_started"])
        self.assertTrue(all(flag is False for flag in value["privacy"].values()))
        expected = hashlib.sha256(result.read_bytes()).hexdigest()
        digest, name = (EVIDENCE / "SHA256SUMS").read_text(
            encoding="ascii").strip().split("  ", 1)
        self.assertEqual(name, "result.json")
        self.assertEqual(digest, expected)


if __name__ == "__main__":
    unittest.main()
