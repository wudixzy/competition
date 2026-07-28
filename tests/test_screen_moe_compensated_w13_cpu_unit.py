from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import unittest

from tests import screen_moe_compensated_w13_cpu as screen


class CompensatedW13CpuScreenUnitTests(unittest.TestCase):
    def test_ieee_rounding_helpers(self) -> None:
        self.assertEqual(screen.f16(1.0), 1.0)
        self.assertEqual(screen.f32(1.0), 1.0)
        self.assertNotEqual(screen.f16(1.0001), 1.0001)

    def test_warp_reduce_requires_fixed_lane_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 32 lanes"):
            screen.warp_reduce([0.0] * 31)
        self.assertEqual(screen.warp_reduce([1.0] * 32), 32.0)

    def test_single_row_is_finite_and_deterministic(self) -> None:
        inputs = [screen.f16(0.01 * ((index % 17) - 8))
                  for index in range(screen.HIDDEN)]
        left = screen.simulate_row(inputs, random.Random(7))
        right = screen.simulate_row(inputs, random.Random(7))
        self.assertEqual(left, right)
        self.assertTrue(all(math.isfinite(value) for value in left))

    def test_screen_cannot_mark_itself_qualified(self) -> None:
        source = Path(screen.__file__).read_text(encoding="utf-8")
        self.assertIn('"qualified": False', source)
        self.assertIn("not CoreX vendor evidence", source)

    def test_committed_evidence_binds_exact_screen_source(self) -> None:
        source_path = Path(screen.__file__)
        evidence_path = (
            source_path.parents[1]
            / "docs"
            / "experiments"
            / "evidence"
            / "M1_91_COMPENSATED_W13_CPU_SCREEN_20260728.json"
        )
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        observed_sha256 = hashlib.sha256(
            source_path.read_bytes()).hexdigest()
        self.assertEqual(
            evidence["runtime"]["script_sha256"],
            observed_sha256,
        )
        self.assertFalse(evidence["qualified"])
        self.assertEqual(evidence["config"]["seed"], screen.SEED)
        self.assertEqual(evidence["config"]["steps"], screen.STEPS)
        self.assertEqual(
            evidence["config"]["rows_per_step"],
            screen.ROWS_PER_STEP,
        )


if __name__ == "__main__":
    unittest.main()
