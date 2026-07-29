from __future__ import annotations

import unittest

import paired_noninferiority as paired


class PairedNoninferiorityTests(unittest.TestCase):

    def test_zero_regression_power_floor_is_derived(self) -> None:
        self.assertEqual(
            paired.minimum_zero_regression_samples(0.02, 0.95),
            149,
        )
        self.assertEqual(
            paired.minimum_zero_regression_samples(0.05, 0.95),
            59,
        )

    def test_contract_mode_rejects_one_baseline_only_failure(self) -> None:
        result = paired.paired_noninferiority(
            [True, True, False],
            [True, False, True],
            margin=0.05,
            confidence=0.95,
            bootstrap_samples=1000,
            seed=7,
            mode="contract",
        )
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["paired_counts"]["baseline_only"], 1)
        self.assertFalse(result["qualified"])

    def test_contract_mode_allows_candidate_only_improvement(self) -> None:
        result = paired.paired_noninferiority(
            [True, False, False],
            [True, True, False],
            margin=0.05,
            confidence=0.95,
            bootstrap_samples=1000,
            seed=7,
            mode="contract",
        )
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["qualified"])

    def test_64_clean_pairs_support_five_percent_screen(self) -> None:
        result = paired.paired_noninferiority(
            [True] * 64,
            [True] * 64,
            margin=0.05,
            confidence=0.95,
            bootstrap_samples=2000,
            seed=7,
        )
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["qualified"])
        self.assertEqual(
            result["statistics"]["minimum_zero_regression_samples"],
            59,
        )

    def test_64_clean_pairs_are_inconclusive_at_two_percent(self) -> None:
        result = paired.paired_noninferiority(
            [True] * 64,
            [True] * 64,
            margin=0.02,
            confidence=0.95,
            bootstrap_samples=2000,
            seed=7,
        )
        self.assertEqual(result["status"], "inconclusive")
        self.assertFalse(result["qualified"])
        self.assertEqual(paired.exit_code(result["status"]), 3)

    def test_clear_regression_fails_even_with_enough_samples(self) -> None:
        baseline = [True] * 200
        candidate = [False] * 20 + [True] * 180
        result = paired.paired_noninferiority(
            baseline,
            candidate,
            margin=0.02,
            confidence=0.95,
            bootstrap_samples=4000,
            seed=7,
        )
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["qualified"])
        self.assertLess(
            result["statistics"]["one_sided_lower_bound"],
            -0.02,
        )

    def test_report_does_not_retain_sample_outcomes(self) -> None:
        result = paired.paired_noninferiority(
            [True] * 64,
            [True] * 64,
            margin=0.05,
            confidence=0.95,
            bootstrap_samples=1000,
            seed=7,
        )
        self.assertNotIn("baseline", result)
        self.assertNotIn("candidate", result)
        self.assertTrue(all(
            value is False for value in result["privacy"].values()
        ))


if __name__ == "__main__":
    unittest.main()
