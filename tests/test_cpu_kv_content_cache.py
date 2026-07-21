import hashlib
import importlib.util
import pathlib
import random
import unittest
from collections import OrderedDict


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "vllm" / "core" / "block" / "cpu_kv_content_cache.py")
SPEC = importlib.util.spec_from_file_location(
    "m1_45_cpu_kv_content_cache_under_test", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)

CpuKvContentCache = module.CpuKvContentCache
cpu_kv_offload_enabled = module.cpu_kv_offload_enabled


def digest(value: int) -> bytes:
    return hashlib.sha256(value.to_bytes(8, "big")).digest()


class CpuKvContentCacheTest(unittest.TestCase):
    def test_environment_selector_is_strict_and_defaults_off(self):
        self.assertFalse(cpu_kv_offload_enabled({}))
        self.assertFalse(cpu_kv_offload_enabled({"BI100_CPU_KV_OFFLOAD": "0"}))
        self.assertTrue(cpu_kv_offload_enabled({"BI100_CPU_KV_OFFLOAD": "1"}))
        for invalid in ("", "true", "on", "2", " 1"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(RuntimeError, "exactly '0' or '1'"):
                    cpu_kv_offload_enabled({
                        "BI100_CPU_KV_OFFLOAD": invalid,
                    })

    def test_pending_store_is_not_visible_until_step_drains(self):
        cache = CpuKvContentCache(2)
        key = digest(1)
        self.assertTrue(cache.stage_store(key, gpu_block=4))
        self.assertFalse(cache.is_ready(key))
        self.assertIsNone(cache.claim_load(key))
        self.assertEqual(cache.drain_step(), ([], [(4, 0)]))
        self.assertFalse(cache.is_ready(key))
        cache.begin_step()
        self.assertTrue(cache.is_ready(key))

        slot = cache.claim_load(key)
        self.assertEqual(slot, 0)
        cache.stage_load(key, slot, gpu_block=9)
        self.assertEqual(cache.drain_step(), ([(0, 9)], []))
        self.assertTrue(cache.is_ready(key))

    def test_same_gpu_slot_can_store_victim_then_load_requested_content(self):
        cache = CpuKvContentCache(3)
        first, victim, replacement = digest(1), digest(2), digest(3)
        cache.stage_store(first, gpu_block=10)
        cache.drain_step()
        cache.begin_step()
        cache.stage_store(victim, gpu_block=11)
        cache.drain_step()
        cache.begin_step()

        first_slot = cache.claim_load(first)
        self.assertEqual(first_slot, 0)
        self.assertTrue(cache.stage_store(replacement, gpu_block=7))
        cache.stage_load(first, first_slot, gpu_block=7)
        swap_in, swap_out = cache.drain_step()

        self.assertEqual(swap_in, [(0, 7)])
        self.assertEqual(swap_out, [(7, 2)])
        self.assertEqual(cache.resident_slot(victim), 1)
        self.assertEqual(cache.resident_slot(replacement), 2)
        self.assertFalse(cache.is_ready(replacement))
        cache.begin_step()
        self.assertTrue(cache.is_ready(replacement))

    def test_saturated_promotion_does_not_evict_later_prefix_sources(self):
        cache = CpuKvContentCache(4)
        keys = [digest(index) for index in range(5)]
        for gpu_block, key in enumerate(keys[:4]):
            self.assertTrue(cache.stage_store(key, gpu_block))
        cache.drain_step()
        cache.begin_step()

        first_slot = cache.claim_load(keys[0])
        self.assertEqual(first_slot, 0)
        self.assertFalse(cache.stage_store(keys[4], gpu_block=9))
        cache.stage_load(keys[0], first_slot, gpu_block=9)

        # Sequential block allocation has not claimed these entries yet, but
        # they belong to the same restorable prefix and must remain available.
        self.assertEqual(
            [cache.resident_slot(key) for key in keys[1:4]], [1, 2, 3])
        self.assertEqual(cache.drain_step(), ([(0, 9)], []))
        self.assertEqual(cache.skipped_stores, 1)

    def test_pure_store_step_still_replaces_multiple_lru_entries(self):
        cache = CpuKvContentCache(2)
        keys = [digest(index) for index in range(4)]
        self.assertTrue(cache.stage_store(keys[0], gpu_block=0))
        self.assertTrue(cache.stage_store(keys[1], gpu_block=1))
        cache.drain_step()
        cache.begin_step()

        self.assertTrue(cache.stage_store(keys[2], gpu_block=2))
        self.assertTrue(cache.stage_store(keys[3], gpu_block=3))
        self.assertEqual(cache.drain_step(), ([], [(2, 0), (3, 1)]))
        self.assertIsNone(cache.resident_slot(keys[0]))
        self.assertIsNone(cache.resident_slot(keys[1]))
        self.assertEqual(cache.resident_slot(keys[2]), 0)
        self.assertEqual(cache.resident_slot(keys[3]), 1)

    def test_claimed_slots_are_not_overwritten(self):
        cache = CpuKvContentCache(2)
        keys = [digest(index) for index in range(3)]
        for gpu_block, key in enumerate(keys[:2]):
            cache.stage_store(key, gpu_block)
            cache.drain_step()
            cache.begin_step()

        slots = [cache.claim_load(key) for key in keys[:2]]
        self.assertEqual(slots, [0, 1])
        self.assertFalse(cache.stage_store(keys[2], gpu_block=5))
        self.assertEqual(cache.skipped_stores, 1)
        for gpu_block, (key, slot) in enumerate(zip(keys[:2], slots), 20):
            cache.stage_load(key, slot, gpu_block)
        self.assertEqual(cache.drain_step(), ([(0, 20), (1, 21)], []))

    def test_inclusive_copy_deduplicates_store_after_load(self):
        cache = CpuKvContentCache(1)
        key = digest(4)
        cache.stage_store(key, gpu_block=0)
        cache.drain_step()
        cache.begin_step()
        slot = cache.claim_load(key)
        cache.stage_load(key, slot, gpu_block=3)
        cache.drain_step()

        self.assertFalse(cache.stage_store(key, gpu_block=3))
        self.assertEqual(cache.deduplicated_stores, 1)
        self.assertEqual(cache.drain_step(), ([], []))
        self.assertTrue(cache.is_ready(key))

    def test_uncommitted_claim_fails_closed_and_can_be_cancelled(self):
        cache = CpuKvContentCache(1)
        key = digest(5)
        cache.stage_store(key, gpu_block=0)
        cache.drain_step()
        cache.begin_step()
        slot = cache.claim_load(key)
        with self.assertRaisesRegex(RuntimeError, "uncommitted slot claim"):
            cache.drain_step()
        cache.cancel_load(key, slot)
        self.assertEqual(cache.drain_step(), ([], []))

    def test_invalid_keys_and_indices_fail_closed(self):
        with self.assertRaises(ValueError):
            CpuKvContentCache(0)
        cache = CpuKvContentCache(1)
        for invalid in (1, b"short", bytearray(32)):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "32-byte"):
                    cache.stage_store(invalid, 0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            cache.stage_store(digest(1), -1)
        with self.assertRaisesRegex(TypeError, "integer"):
            cache.stage_store(digest(1), True)

    def test_fixed_random_lifecycle_matches_lru_oracle(self):
        capacity = 8
        cache = CpuKvContentCache(capacity)
        rng = random.Random(20260721)
        keys = [digest(index) for index in range(32)]
        oracle: OrderedDict[bytes, int] = OrderedDict()
        free_slots = list(range(capacity))

        for tick in range(10_000):
            cache.begin_step()
            key = keys[rng.randrange(len(keys))]
            if rng.random() < 0.55:
                expected_slot = oracle.get(key)
                actual_slot = cache.claim_load(key)
                self.assertEqual(actual_slot, expected_slot)
                if expected_slot is not None:
                    oracle.move_to_end(key)
                    cache.stage_load(key, actual_slot, gpu_block=tick % 19)
                    expected_maps = ([(actual_slot, tick % 19)], [])
                else:
                    expected_maps = ([], [])
            else:
                if key in oracle:
                    oracle.move_to_end(key)
                    expected_stored = False
                    expected_maps = ([], [])
                else:
                    if free_slots:
                        slot = free_slots.pop(0)
                    else:
                        _, slot = oracle.popitem(last=False)
                    oracle[key] = slot
                    expected_stored = True
                    expected_maps = ([], [(tick % 19, slot)])
                self.assertEqual(
                    cache.stage_store(key, gpu_block=tick % 19),
                    expected_stored)

            self.assertEqual(cache.drain_step(), expected_maps)
            self.assertEqual(cache.resident_count, len(oracle))
            for resident_key, resident_slot in oracle.items():
                self.assertEqual(
                    cache.resident_slot(resident_key), resident_slot)
                if expected_maps[1] and resident_key == key:
                    self.assertFalse(cache.is_ready(resident_key))
                else:
                    self.assertTrue(cache.is_ready(resident_key))

    def test_submission_stays_default_off_while_runtime_files_are_installed(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        patch_ops = (
            ROOT / "qwen3_6_scripts" / "patch_ops.sh"
        ).read_text(encoding="utf-8")
        scheduler = (
            ROOT / "qwen3_6_scripts" / "scheduler.py"
        ).read_text(encoding="utf-8")
        run_config = (
            ROOT / "computility-run.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("cpu_kv_content_cache.py", dockerfile)
        self.assertIn("cpu_gpu_block_allocator.py", dockerfile)
        self.assertIn("core/block/cpu_kv_content_cache.py", patch_ops)
        self.assertIn("core/block/cpu_gpu_block_allocator.py", patch_ops)
        self.assertIn("patch_worker_cache_transfer_order.py", patch_ops)
        self.assertIn("begin_prefix_cache_step", scheduler)
        self.assertIn("get_and_reset_prefix_swaps", scheduler)
        self.assertNotIn("BI100_CPU_KV_OFFLOAD", run_config)


if __name__ == "__main__":
    unittest.main()
