import hashlib
import importlib.util
import pathlib
import random
import unittest
from unittest.mock import patch


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "vllm" / "core" / "evictor_v2.py"
SPEC = importlib.util.spec_from_file_location(
    "m1_41_evictor_v2_under_test", MODULE_PATH)
evictor_v2 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(evictor_v2)

EvictionPolicy = evictor_v2.EvictionPolicy
FrequencyAwareEvictor = evictor_v2.FrequencyAwareEvictor
LRUEvictor = evictor_v2.LRUEvictor
eviction_policy_from_env = evictor_v2.eviction_policy_from_env
make_evictor = evictor_v2.make_evictor


def digest(value: int) -> bytes:
    return hashlib.sha256(value.to_bytes(8, "big")).digest()


class FrequencyAwareContentEvictorTest(unittest.TestCase):
    def setUp(self):
        self.evictor = FrequencyAwareEvictor()

    def add(self, block_id, value, tokens=1, accessed=0.0):
        content_hash = value if isinstance(value, bytes) else digest(value)
        self.evictor.add(block_id, content_hash, tokens, accessed)
        return content_hash

    def test_victim_order_uses_fixed_key(self):
        hashes = {
            block_id: self.add(
                block_id, block_id, tokens=tokens, accessed=accessed)
            for block_id, tokens, accessed in (
                (10, 1, 3.0),
                (20, 1, 2.0),
                (30, 7, 2.0),
                (40, 1, 1.0),
                (50, 4, 1.0),
            )
        }
        for block_id in (50, 40, 30, 20, 10):
            self.assertEqual(
                self.evictor.evict(), (block_id, hashes[block_id]))

    def test_frequency_persists_across_physical_reuse(self):
        hash_11 = self.add(1, 11)
        hash_22 = self.add(2, 22)
        self.add(3, hash_22)
        self.assertEqual(self.evictor.evict(), (1, hash_11))

        self.add(4, hash_11)
        hash_33 = self.add(5, 33)
        self.assertEqual(self.evictor.frequency_by_hash, {
            hash_11: 2,
            hash_22: 2,
            hash_33: 1,
        })
        self.assertEqual(self.evictor.evict(), (5, hash_33))

        self.evictor.remove(2)
        self.add(6, hash_22)
        self.assertEqual(self.evictor.frequency_by_hash[hash_22], 3)
        self.assertEqual(self.evictor.evict(), (4, hash_11))

    def test_frequency_change_invalidates_other_heap_entry(self):
        hash_7 = self.add(1, 7, accessed=1.0)
        hash_8 = self.add(2, 8, accessed=2.0)
        self.add(3, hash_7, accessed=3.0)
        self.assertEqual(self.evictor.evict(), (2, hash_8))

    def test_update_and_block_id_tie_break_are_deterministic(self):
        hash_1 = self.add(30, 1, tokens=5, accessed=1.0)
        hash_2 = self.add(10, 2, tokens=5, accessed=1.0)
        self.add(20, 3, tokens=5, accessed=1.0)
        self.evictor.update(30, 2.0)
        self.assertEqual(self.evictor.evict(), (10, hash_2))
        self.assertIn(hash_1, self.evictor.frequency_by_hash)

    def test_heap_compaction_is_bounded(self):
        for index in range(40):
            self.add(1, 1, accessed=float(index))
            self.assertLessEqual(
                len(self.evictor._heap),
                2 * self.evictor.num_blocks + 1,
            )

    def test_fixed_random_lifecycle_matches_full_scan_oracle(self):
        rng = random.Random(20260721)
        free_ids = list(range(64))
        metadata = {}
        frequencies = {}
        reusable_hashes = [digest(value) for value in range(512)]

        def oracle_victim():
            return min(metadata, key=lambda block_id: (
                frequencies[metadata[block_id][0]],
                metadata[block_id][2],
                -metadata[block_id][1],
                block_id,
            ))

        for tick in range(10_000):
            live_ids = list(metadata)
            if not live_ids or (free_ids and rng.random() < 0.45):
                block_id = free_ids.pop(rng.randrange(len(free_ids)))
                live_hashes = {item[0] for item in metadata.values()}
                choices = [value for value in reusable_hashes
                           if value not in live_hashes]
                content_hash = choices[rng.randrange(len(choices))]
                tokens = 16 * rng.randint(1, 1024)
                accessed = float(rng.randint(0, tick + 1))
                frequencies[content_hash] = (
                    frequencies.get(content_hash, 0) + 1)
                metadata[block_id] = (content_hash, tokens, accessed)
                self.evictor.add(
                    block_id, content_hash, tokens, accessed)
            elif rng.random() < 0.35:
                block_id = live_ids[rng.randrange(len(live_ids))]
                accessed = float(rng.randint(0, tick + 1))
                content_hash, tokens, _ = metadata[block_id]
                metadata[block_id] = (content_hash, tokens, accessed)
                self.evictor.update(block_id, accessed)
            elif rng.random() < 0.50:
                block_id = live_ids[rng.randrange(len(live_ids))]
                metadata.pop(block_id)
                self.evictor.remove(block_id)
                free_ids.append(block_id)
            else:
                expected_id = oracle_victim()
                expected_hash = metadata[expected_id][0]
                self.assertEqual(
                    self.evictor.evict(), (expected_id, expected_hash))
                metadata.pop(expected_id)
                free_ids.append(expected_id)

            self.assertEqual(self.evictor.num_blocks, len(metadata))
            self.assertLessEqual(
                len(self.evictor._heap),
                2 * self.evictor.num_blocks + 1,
            )

    def test_hash_and_api_inputs_fail_closed(self):
        for invalid in (1, b"short", bytearray(32)):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "32-byte"):
                    self.evictor.add(1, invalid, 16, 0.0)
        with self.assertRaises(ValueError):
            self.evictor.evict()
        with self.assertRaises(KeyError):
            self.evictor.update(1, 1.0)
        with self.assertRaises(ValueError):
            self.evictor.remove(1)
        with self.assertRaises(ValueError):
            make_evictor("frequency")

    def test_environment_selector_defaults_off_and_rejects_typos(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                eviction_policy_from_env(), EvictionPolicy.LRU)
        self.assertEqual(
            eviction_policy_from_env({"BI100_KV_EVICTION_POLICY": "lru"}),
            EvictionPolicy.LRU,
        )
        self.assertEqual(
            eviction_policy_from_env({
                "BI100_KV_EVICTION_POLICY": "frequency",
            }),
            EvictionPolicy.FREQUENCY_AWARE,
        )
        for invalid in ("", "lfu", "frequency64"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "frequency, lru"):
                    eviction_policy_from_env({
                        "BI100_KV_EVICTION_POLICY": invalid,
                    })

    def test_factory_and_lru_non_regression(self):
        self.assertIsInstance(
            make_evictor(EvictionPolicy.FREQUENCY_AWARE),
            FrequencyAwareEvictor,
        )
        lru = LRUEvictor()
        hashes = [digest(value) for value in (101, 102, 103)]
        lru.add(1, hashes[0], 1, 1.0)
        lru.add(2, hashes[1], 3, 1.0)
        lru.add(3, hashes[2], 9, 2.0)
        self.assertEqual(lru.evict(), (2, hashes[1]))
        self.assertEqual(lru.evict(), (1, hashes[0]))
        self.assertEqual(lru.evict(), (3, hashes[2]))

    def test_override_is_installed_but_not_enabled_in_yaml(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        patch_ops = (
            ROOT / "qwen3_6_scripts" / "patch_ops.sh"
        ).read_text(encoding="utf-8")
        run_config = (
            ROOT / "computility-run.yaml"
        ).read_text(encoding="utf-8")
        allocator = (
            ROOT / "vllm" / "core" / "block" /
            "prefix_caching_block.py"
        ).read_text(encoding="utf-8")
        self.assertIn("vllm/core/evictor_v2.py", dockerfile)
        self.assertIn("VLLM_ROOT}/core/evictor_v2.py", patch_ops)
        self.assertIn("eviction_policy_from_env()", allocator)
        self.assertNotIn("BI100_KV_EVICTION_POLICY", run_config)


if __name__ == "__main__":
    unittest.main()
