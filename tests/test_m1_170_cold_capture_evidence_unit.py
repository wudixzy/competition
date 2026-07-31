from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs" / "experiments" / "evidence"
    / "M1_170_COLD_CAPTURE_OVERHEAD_TP1_20260801"
)
REVISION = "70e6d318408c75265b0934bc355b64a4610dc23e"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class M1170ColdCaptureEvidenceUnitTest(unittest.TestCase):

    def test_order_balanced_summary_is_bounded(self) -> None:
        report = load(EVIDENCE / "order_balance.json")
        self.assertEqual(
            report["schema"], "bi100-m1-170-order-balanced-summary-v1")
        self.assertTrue(report["qualified_order_balanced_timing"])
        self.assertEqual(report["source_revision"], REVISION)
        self.assertAlmostEqual(
            report["median"][
                "order_balanced_geometric_overhead_fraction"],
            0.023284375473967556,
        )
        self.assertAlmostEqual(
            report["by_shape"]["4096"][
                "order_balanced_geometric_overhead_fraction"],
            0.07542066496810018,
        )
        self.assertFalse(report["scope"]["statistical_significance_claimed"])
        self.assertFalse(report["scope"]["tp4_evaluated"])
        self.assertFalse(
            report["scope"]["production_promotion_authorized"])

    def test_both_orders_are_cold_and_lifecycle_clean(self) -> None:
        expected_orders = {
            "forward": ["admission64", "off"],
            "reverse": ["off", "admission64"],
        }
        for label, order in expected_orders.items():
            runner = load(EVIDENCE / label / "runner_status.json")
            comparison = load(EVIDENCE / label / "comparison.json")
            self.assertEqual(runner["source_revision"], REVISION)
            self.assertEqual(runner["arm_order"], order)
            self.assertEqual(runner["bench_tool_count"], 0)
            self.assertTrue(runner["qualified_development_screen"])
            self.assertTrue(all(value == 0
                                for value in runner["gates"].values()))
            self.assertTrue(comparison["qualified_analysis"])
            self.assertTrue(comparison["cold_isolation"]["qualified"])
            self.assertEqual(
                comparison["cold_isolation"][
                    "admission64_cold_cached_tokens"], 0)
            self.assertEqual(
                comparison["cold_isolation"]["off_cold_cached_tokens"], 0)
            for policy in ("admission64", "off"):
                measurement = load(
                    EVIDENCE / label / policy / "measurement.json")
                self.assertTrue(measurement["qualified_measurement"])
                self.assertEqual(measurement["aggregate"][
                    "cold_cached_tokens"], 0)
                self.assertFalse(
                    measurement["privacy"]["contains_raw_prompt"])
                self.assertFalse(
                    measurement["privacy"]["contains_raw_output"])

    def test_runtime_identity_and_import_contract_match(self) -> None:
        runtime = load(EVIDENCE / "forward" / "runtime_identity.json")
        install = load(EVIDENCE / "runtime" / "install.json")
        imports = load(
            EVIDENCE / "runtime" / "runtime_import_contract.json")
        self.assertTrue(runtime["qualified"])
        self.assertEqual(runtime["source_revision"], REVISION)
        self.assertEqual(install["source_revision"], REVISION)
        self.assertTrue(install["qualified"])
        self.assertTrue(imports["qualified"])
        self.assertEqual(
            runtime["runtime_tree_sha256"],
            "93f65cce49f5455401ae35c97371b0e8dbf0f94e7ab23878d2e7c5991603e849",
        )

    def test_sha_manifest_and_privacy_boundary(self) -> None:
        lines = (EVIDENCE / "SHA256SUMS").read_text(
            encoding="utf-8").splitlines()
        self.assertGreater(len(lines), 60)
        for line in lines:
            expected, relative = line.split("  ", 1)
            path = EVIDENCE / relative.removeprefix("./")
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             expected)
        retained = [path.name for path in EVIDENCE.rglob("*") if path.is_file()]
        self.assertFalse(any(name.endswith(".log") for name in retained))
        self.assertFalse(any(name.endswith(".stdout") for name in retained))


if __name__ == "__main__":
    unittest.main()
