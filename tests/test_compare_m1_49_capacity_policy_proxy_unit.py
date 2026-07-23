import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests/compare_m1_49_capacity_policy_proxy.py"
SPEC = importlib.util.spec_from_file_location("m1_49_capacity_proxy", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class M149CapacityPolicyProxyUnitTest(unittest.TestCase):

    def test_fixed_capacity_is_exactly_four_times_legacy(self):
        self.assertEqual(
            MODULE.M1_49_ESTIMATED_CAPACITY_BLOCKS,
            4 * MODULE.LEGACY_CAPACITY_BLOCKS,
        )

    def test_heap_lru_evicts_oldest_then_deepest(self):
        policy = MODULE.HeapLruOracle()
        digest = lambda value: bytes([value]) * 32
        policy.add(1, digest(1), 16, 1.0)
        policy.add(2, digest(2), 32, 1.0)
        policy.add(3, digest(3), 48, 2.0)

        self.assertEqual(policy.evict(), (2, digest(2)))
        self.assertEqual(policy.evict(), (1, digest(1)))
        self.assertEqual(policy.evict(), (3, digest(3)))

    def test_percentage_point_gain_requires_matching_access_counts(self):
        control = {"hit_rate": 0.10, "total_accesses": 100}
        candidate = {"hit_rate": 0.25, "total_accesses": 100}
        self.assertAlmostEqual(
            MODULE._percentage_point_gain(candidate, control), 15.0)

        candidate["total_accesses"] = 99
        with self.assertRaises(ValueError):
            MODULE._percentage_point_gain(candidate, control)


if __name__ == "__main__":
    unittest.main()
