from __future__ import annotations

import os
import random
import unittest
from collections import OrderedDict
from unittest.mock import patch

from qwen3_6_scripts.gdn_prefix import (
    GdnPrefixStatePolicy,
    canonical_direct_segment_offsets,
    capture_points_for_step,
    final_capture_key,
    gdn_cache_policy_from_env,
    gdn_restore_alignment,
    gdn_restore_mode_from_env,
    key_at_strict_boundary,
    keys_from_block_hashes,
    make_prefix_key,
    restore_key_is_eligible,
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

    def test_direct_hybrid_chunk64_and_aligned_final_capture_keys(self):
        hashes = [digest(i % 255) for i in range(14687)]
        self.assertEqual(
            final_capture_key(hashes, 235000, 16, "direct", 8192)[0],
            14687)
        self.assertEqual(
            final_capture_key(hashes, 235000, 16, "hybrid64", 64)[0],
            14687)
        self.assertEqual(
            final_capture_key(hashes, 235000, 16, "chunk64", 64)[0],
            14684)
        self.assertEqual(
            final_capture_key(hashes, 235000, 16, "aligned", 8192)[0],
            14336)

    def test_direct_avoids_single_token_prefill_replay(self):
        hashes = [digest(i % 255) for i in range(700)]
        self.assertEqual(
            final_capture_key(hashes, 10593, 16, "direct", 16)[0], 661)
        self.assertEqual(
            final_capture_key(hashes, 10594, 16, "direct", 16)[0], 662)
        self.assertFalse(restore_key_is_eligible(
            (662, hashes[661]), 10593, 16, "direct", 16))
        self.assertTrue(restore_key_is_eligible(
            (661, hashes[660]), 10593, 16, "direct", 16))

    def test_direct_capture_leaves_minimal_safe_suffix_for_all_remainders(self):
        hashes = [digest(i % 255) for i in range(32)]
        for prompt_tokens in range(18, 16 * 20 + 1):
            with self.subTest(prompt_tokens=prompt_tokens):
                key = final_capture_key(
                    hashes, prompt_tokens, 16, "direct", 16)
                self.assertIsNotNone(key)
                remaining = prompt_tokens - key[0] * 16
                self.assertGreaterEqual(remaining, 2)
                self.assertLessEqual(remaining, 17)
                self.assertTrue(restore_key_is_eligible(
                    key, prompt_tokens, 16, "direct", 16))

    def test_aligned_restore_eligibility_uses_fixed_boundary(self):
        key = (512, digest(1))
        self.assertTrue(restore_key_is_eligible(
            key, 10000, 16, "aligned", 8192))
        self.assertFalse(restore_key_is_eligible(
            (511, digest(2)), 10000, 16, "aligned", 8192))

    def test_restore_alignment_matches_execution_granularity(self):
        self.assertEqual(gdn_restore_alignment("direct", 16, 8192), 16)
        self.assertEqual(gdn_restore_alignment("hybrid64", 16, 8192), 64)
        self.assertEqual(gdn_restore_alignment("chunk64", 16, 8192), 64)
        self.assertEqual(gdn_restore_alignment("aligned", 16, 8192), 8192)
        with self.assertRaises(ValueError):
            gdn_restore_alignment("chunk64", 24, 8192)

    def test_hybrid_restore_accepts_aligned_branch_or_exact_final_only(self):
        aligned = (192, digest(1))  # 3072 tokens
        unaligned_branch = (194, digest(2))  # 3104 tokens
        exact_final = (255, digest(3))  # 4080 tokens
        self.assertTrue(restore_key_is_eligible(
            aligned, 4096, 16, "hybrid64", 64,
            direct_final_key=exact_final))
        self.assertFalse(restore_key_is_eligible(
            unaligned_branch, 4096, 16, "hybrid64", 64,
            direct_final_key=exact_final))
        self.assertTrue(restore_key_is_eligible(
            exact_final, 4096, 16, "hybrid64", 64,
            direct_final_key=exact_final))
        self.assertFalse(restore_key_is_eligible(
            exact_final, 8192, 16, "hybrid64", 64,
            direct_final_key=(511, digest(4))))

    def test_capture_points_are_relative_to_physical_context(self):
        targets = [(512, digest(1)), (544, digest(2))]
        self.assertEqual(
            capture_points_for_step(targets, 8000, 8712, 16),
            ((192, targets[0]), (704, targets[1])))

    def test_canonical_segments_reproduce_crossed_scheduler_steps(self):
        hashes = [digest(i % 255) for i in range(1000)]
        self.assertEqual(
            canonical_direct_segment_offsets(
                hashes, 3072, 16000, 16, 8192),
            (5104, 5120, 12912))
        self.assertEqual(
            canonical_direct_segment_offsets(
                hashes, 3072, 7799, 16, 8192),
            (4720,))
        self.assertEqual(
            canonical_direct_segment_offsets(
                hashes, 4080, 4096, 16, 8192), ())

    def test_canonical_segments_cover_262k_without_exceeding_wire_cap(self):
        hashes = [digest(i % 255) for i in range(16384)]
        offsets = canonical_direct_segment_offsets(
            hashes, 3072, 262144, 16, 8192)
        self.assertLessEqual(len(offsets), 128)
        self.assertEqual(offsets[:2], (5104, 5120))
        self.assertEqual(offsets[-2:], (250880, 259056))

    def test_admission64_admits_a_repeated_raw_kv_branch(self):
        policy = GdnPrefixStatePolicy("admission64")
        keys = keys_from_block_hashes([digest(1), digest(2)])
        self.assertEqual(policy.repeated_branch_candidate(keys, 2), keys[1])
        policy.admit([keys[1]])
        self.assertIsNone(policy.repeated_branch_candidate(keys, 2))
        self.assertIsNone(policy.repeated_branch_candidate([], 1))

    def test_admission64_retains_canonical_final_state(self):
        key = make_prefix_key(2, digest(2))
        admission = GdnPrefixStatePolicy("admission64")
        self.assertTrue(admission.should_capture_final(key))
        admission.admit([key])
        self.assertFalse(admission.should_capture_final(key))

        fine = GdnPrefixStatePolicy("fine32")
        fine.admit([key])
        self.assertTrue(fine.should_capture_final(key))
        self.assertFalse(GdnPrefixStatePolicy("off").should_capture_final(key))

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
        with patch.dict(os.environ, {
                "BI100_GDN_RESTORE_MODE": "chunk64",
        }, clear=False):
            self.assertEqual(gdn_restore_mode_from_env(), "chunk64")
        with patch.dict(os.environ, {
                "BI100_GDN_RESTORE_MODE": "hybrid64",
        }, clear=False):
            self.assertEqual(gdn_restore_mode_from_env(), "hybrid64")
        with patch.dict(os.environ, {"BI100_GDN_CACHE_POLICY": "typo"},
                        clear=False):
            with self.assertRaises(RuntimeError):
                gdn_cache_policy_from_env()
        for policy in ("fine32", "admission64", "off"):
            with self.subTest(policy=policy), patch.dict(os.environ, {
                    "BI100_GDN_CACHE_POLICY": policy,
                    "BI100_GDN_RESTORE_MODE": "typo",
            }, clear=False):
                self.assertEqual(gdn_cache_policy_from_env(), policy)
                with self.assertRaises(RuntimeError):
                    gdn_restore_mode_from_env()

    def test_off_policy_is_explicit_and_does_not_mask_restore_validation(self):
        with patch.dict(os.environ, {
                "BI100_GDN_CACHE_POLICY": "off",
                "BI100_GDN_RESTORE_MODE": "direct",
        }, clear=False):
            self.assertEqual(gdn_cache_policy_from_env(), "off")
            self.assertEqual(gdn_restore_mode_from_env(), "direct")

    def test_fixed_seed_policy_state_machine_matches_reference_lru(self):
        keys = [make_prefix_key(i + 1, digest(i % 251))
                for i in range(96)]
        rng = random.Random(20260728)
        for policy_name, capacity in (
                ("fine32", 32), ("admission64", 64), ("off", 0)):
            with self.subTest(policy=policy_name):
                policy = GdnPrefixStatePolicy(policy_name)
                reference = OrderedDict()
                for _ in range(1000):
                    operation = rng.choice(
                        ("admit", "restore", "forget", "capture"))
                    sample = rng.sample(keys, rng.randint(0, 4))
                    if operation == "admit":
                        expected_evictions = []
                        if capacity:
                            for key in sample:
                                if key in reference:
                                    reference.move_to_end(key)
                                else:
                                    reference[key] = None
                                while len(reference) > capacity:
                                    evicted, _ = reference.popitem(last=False)
                                    expected_evictions.append(evicted)
                        self.assertEqual(
                            policy.admit(sample), tuple(expected_evictions))
                    elif operation == "restore":
                        max_blocks = rng.randint(0, len(sample))
                        expected = None
                        if capacity and max_blocks:
                            for key in sample[:max_blocks]:
                                if key in reference:
                                    expected = key
                            if expected is not None:
                                reference.move_to_end(expected)
                        self.assertEqual(
                            policy.select_restore(sample, max_blocks), expected)
                    elif operation == "forget":
                        policy.forget(sample)
                        for key in sample:
                            reference.pop(key, None)
                    else:
                        key = rng.choice(keys)
                        expected = (
                            False if policy_name == "off"
                            else key not in reference
                            if policy_name == "admission64" else True)
                        self.assertEqual(
                            policy.should_capture_final(key), expected)
                    self.assertEqual(
                        policy.resident_keys(), tuple(reference))
                    self.assertLessEqual(len(policy), capacity)


if __name__ == "__main__":
    unittest.main()
