from __future__ import annotations

import unittest

from tests.prefix_cache_stress import eviction_target_mods


class PrefixCacheStressTest(unittest.TestCase):

    def test_eviction_target_mod_sequence_starts_at_requested_suffix(self):
        self.assertEqual(
            eviction_target_mods(17, 2),
            tuple(range(2, 16)) + (0, 1, 2),
        )

    def test_eviction_target_mod_sequence_validates_inputs(self):
        for count, target_mod in ((0, 1), (17, -1), (17, 16)):
            with self.subTest(count=count, target_mod=target_mod):
                with self.assertRaises(ValueError):
                    eviction_target_mods(count, target_mod)


if __name__ == "__main__":
    unittest.main()
