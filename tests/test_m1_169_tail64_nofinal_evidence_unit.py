import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs" / "experiments" / "evidence"
    / "M1_169_TAIL64_NOFINAL_TP1_20260801"
)


def _load(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class M1169Tail64NoFinalEvidenceUnitTest(unittest.TestCase):

    def test_complete_runner_is_development_only(self):
        value = _load("runner_status.json")
        self.assertEqual(value["returncode"], 0)
        self.assertTrue(value["qualified_development_screen"])
        self.assertEqual(set(value["gates"].values()), {0})
        self.assertFalse(value["model_quality_evaluated"])
        self.assertFalse(value["tp4_evaluated"])
        self.assertFalse(value["production_promotion_authorized"])

    def test_candidate_is_rejected_by_cache_and_ttft(self):
        value = _load("comparison.json")
        control = value["aggregate"]["control"]
        candidate = value["aggregate"]["candidate"]
        delta = value["aggregate"]["candidate_relative_improvement"]

        self.assertTrue(value["qualified_analysis"])
        self.assertEqual(value["request_count"], 18)
        self.assertGreater(control["effective_hit_rate"], 0.62)
        self.assertLess(candidate["effective_hit_rate"], 0.15)
        self.assertLess(delta["weighted"], -0.45)
        self.assertLess(delta["ttft_p90"], -0.06)
        self.assertAlmostEqual(
            candidate["output_tps_p10"],
            control["output_tps_p10"],
            delta=0.01,
        )
        self.assertFalse(
            value["scope"]["production_promotion_authorized"])
        self.assertFalse(value["scope"]["yaml_or_main_change_authorized"])

    def test_cache_transparency_scope_is_not_overclaimed(self):
        value = _load("comparison.json")["cache_transparency"]
        self.assertTrue(value["control_cold_warm_exact"])
        self.assertTrue(value["candidate_cold_warm_exact"])
        self.assertFalse(value["cross_policy_exact_required_for_analysis"])
        self.assertEqual(value["cross_policy_output_identity_matches"], 16)

        for name in (
            "admission64_measurement.json",
            "tail64_nofinal_measurement.json",
        ):
            measurement = _load(name)
            self.assertTrue(measurement["qualified_measurement"])
            self.assertFalse(measurement["privacy"]["contains_raw_prompt"])
            self.assertFalse(measurement["privacy"]["contains_raw_output"])

    def test_fixed_runtime_contains_diagnostic_bypass_and_formal_path(self):
        value = _load("profile_override_runtime.json")
        self.assertTrue(value["install_qualified"])
        self.assertTrue(value["override_marker_present"])
        self.assertTrue(value["override_before_method_empty_cache"])
        self.assertTrue(value["formal_profile_path_present"])
        self.assertTrue(value["startup_guard_present"])
        self.assertTrue(value["diagnostic_only"])
        self.assertFalse(value["production_yaml_changed"])
        self.assertEqual(
            value["source_revision"],
            "91178057d5a29e9d592a3f9e35281d3b1918f4b8",
        )

    def test_manifest_covers_all_json_evidence(self):
        observed = {}
        for line in (EVIDENCE / "SHA256SUMS").read_text(
                encoding="ascii").splitlines():
            digest, name = line.split("  ", 1)
            observed[name] = digest
        expected = {
            path.name: _sha256(path)
            for path in EVIDENCE.glob("*.json")
        }
        self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
