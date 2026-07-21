from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBE = ROOT / "tests" / "probe_fused_paged_prefill_dispatch.py"


class FusedPagedPrefillDispatchProbeUnitTest(unittest.TestCase):

    def test_probe_requires_real_dispatch_and_nonidentity_paging(self):
        source = PROBE.read_text(encoding="utf-8")
        self.assertIn("_TrackedExtension", source)
        self.assertIn("len(tracked.calls) == 1", source)
        self.assertIn("active_blocks", source)
        self.assertIn("padded_block_tables", source)
        self.assertIn("non_identity_physical_blocks", source)
        self.assertIn("_forward_prefix_segment_pytorch", source)
        self.assertIn("maximum_absolute_error <= MAX_ABS_LIMIT", source)
        self.assertIn("relative_l2 <= RELATIVE_L2_LIMIT", source)


if __name__ == "__main__":
    unittest.main()
