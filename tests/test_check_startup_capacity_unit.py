from __future__ import annotations

import unittest

from scripts.check_startup_capacity import evaluate


class StartupCapacityTest(unittest.TestCase):

    def test_accepts_full_262144_window(self):
        report = evaluate(
            "max_seq_len=262144\n# GPU blocks: 16878, # CPU blocks: 6553\n",
            262144,
            16,
        )
        self.assertTrue(report["qualified"])
        self.assertEqual(report["required_gpu_blocks"], 16384)
        self.assertEqual(report["observed_physical_tokens"], 270048)

    def test_rejects_insufficient_physical_blocks(self):
        report = evaluate(
            "max_seq_len=262144\n# GPU blocks: 16383, # CPU blocks: 6553\n",
            262144,
            16,
        )
        self.assertTrue(report["logical_window_ok"])
        self.assertFalse(report["physical_capacity_ok"])
        self.assertFalse(report["qualified"])

    def test_rejects_missing_or_short_logical_window(self):
        missing = evaluate("# GPU blocks: 16878\n", 262144, 16)
        short = evaluate(
            "max_seq_len=100000\n# GPU blocks: 16878\n", 262144, 16)
        self.assertFalse(missing["qualified"])
        self.assertFalse(short["qualified"])


if __name__ == "__main__":
    unittest.main()
