import tempfile
import unittest
from pathlib import Path

from tests.summarize_bi100_profile import summarize


class ProfileSummaryTest(unittest.TestCase):

    def test_groups_complete_forwards_by_process_and_skips_prefill(self):
        lines = []
        for process_prefix, offset in [("", 0.0),
                                       ("(VllmWorkerProcess pid=42) ", 1.0)]:
            for forward in range(3):
                for _layer in range(2):
                    lines.append(
                        f"{process_prefix}[BI100_PROFILE] layer.input_norm "
                        f"{1 + offset:.3f} ms")
                    lines.append(
                        f"{process_prefix}[BI100_PROFILE] moe.routed "
                        f"{10 + forward + offset:.3f} ms")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "service.log"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            report = summarize(path, layers=2, skip_prefill=1)

        self.assertEqual(report["processes"]["driver"]["decode_forwards"], 2)
        self.assertEqual(report["processes"]["42"]["decode_forwards"], 2)
        self.assertEqual(report["regions"]["moe.routed"]["samples"], 4)
        self.assertAlmostEqual(
            report["regions"]["moe.routed"]["mean_ms_per_token_per_rank"],
            24.0,
        )


if __name__ == "__main__":
    unittest.main()
