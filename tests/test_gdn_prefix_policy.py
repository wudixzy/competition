from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from qwen3_6_scripts.gdn_prefix import (
    GdnPrefixStatePolicy,
    capture_points_for_step,
    final_capture_key,
    gdn_cache_policy_from_env,
    gdn_restore_mode_from_env,
    key_at_strict_boundary,
    keys_from_block_hashes,
    make_prefix_key,
)


def digest(value: int) -> bytes:
    return bytes([value]) * 32


class GdnPrefixPolicyTest(unittest.TestCase):

    def test_key_validation_and_strict_boundary(self):
        hashes = [digest(i) for i in range(1, 5)]
        self.assertEqual(keys_from_block_hashes(hashes)[2], (3, digest(3)))
        self.assertEqual(key_at_strict_boundary(hashes, 64, 16),
                         (3, digest(3)))
        self.assertIsNone(key_at_strict_boundary(hashes, 1, 16))
        with self.assertRaises(ValueError):
            make_prefix_key(1, b"short")

    def test_selects_longest_live_resident_key(self):
        policy = GdnPrefixStatePolicy("fine32")
        keys = keys_from_block_hashes([digest(i) for i in range(1, 5)])
        policy.admit([keys[0], keys[2]])
        self.assertEqual(policy.select_restore(keys, 4), keys[2])
        self.assertEqual(policy.select_restore(keys, 2), keys[0])

    def test_direct_and_aligned_final_capture_keys(self):
        hashes = [digest(i % 255) for i in range(14687)]
        self.assertEqual(
            final_capture_key(hashes, 235000, 16, "direct", 8192)[0],
            14687)
        self.assertEqual(
            final_capture_key(hashes, 235000, 16, "aligned", 8192)[0],
            14336)

    def test_capture_points_are_relative_to_physical_context(self):
        targets = [(512, digest(1)), (544, digest(2))]
        self.assertEqual(
            capture_points_for_step(targets, 8000, 8712, 16),
            ((192, targets[0]), (704, targets[1])))

    def test_admission64_admits_a_repeated_raw_kv_branch(self):
        policy = GdnPrefixStatePolicy("admission64")
        keys = keys_from_block_hashes([digest(1), digest(2)])
        self.assertEqual(policy.repeated_branch_candidate(keys, 2), keys[1])
        policy.admit([keys[1]])
        self.assertIsNone(policy.repeated_branch_candidate(keys, 2))

    def test_capacity_emits_explicit_oldest_evictions(self):
        policy = GdnPrefixStatePolicy("fine32")
        keys = [make_prefix_key(i + 1, digest(i % 255)) for i in range(34)]
        self.assertEqual(policy.admit(keys[:32]), ())
        self.assertEqual(policy.admit(keys[32:]), (keys[0], keys[1]))
        self.assertEqual(len(policy), 32)

    def test_environment_modes_fail_closed(self):
        with patch.dict(os.environ, {
                "BI100_GDN_CACHE_POLICY": "admission64",
                "BI100_GDN_RESTORE_MODE": "aligned",
        }, clear=False):
            self.assertEqual(gdn_cache_policy_from_env(), "admission64")
            self.assertEqual(gdn_restore_mode_from_env(), "aligned")
        with patch.dict(os.environ, {"BI100_GDN_CACHE_POLICY": "typo"},
                        clear=False):
            with self.assertRaises(RuntimeError):
                gdn_cache_policy_from_env()


if __name__ == "__main__":
    unittest.main()
