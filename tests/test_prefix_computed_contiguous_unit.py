import ast
import pathlib
import typing
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
ALLOCATOR = (
    ROOT / "vllm" / "core" / "block" / "prefix_caching_block.py")


def load_get_computed_block_ids():
    tree = ast.parse(ALLOCATOR.read_text(encoding="utf-8"))
    allocator = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "PrefixCachingBlockAllocator")
    function = next(
        node for node in allocator.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_computed_block_ids")
    namespace = {"List": typing.List}
    module = ast.fix_missing_locations(ast.Module(
        body=[function], type_ignores=[]))
    exec(compile(module, str(ALLOCATOR), "exec"), namespace)
    return namespace["get_computed_block_ids"]


class FakeAllocator:
    get_computed_block_ids = load_get_computed_block_ids()

    def __init__(self, computed):
        self.computed = set(computed)

    def block_is_computed(self, block_id):
        return block_id in self.computed


class ContiguousComputedPrefixTest(unittest.TestCase):
    def test_first_gap_stops_later_physical_hits(self):
        allocator = FakeAllocator({10, 12, 13})
        self.assertEqual(
            allocator.get_computed_block_ids(
                [], [10, 11, 12, 13], skip_last_block_id=False),
            [10],
        )

    def test_existing_prefix_extends_only_until_next_gap(self):
        allocator = FakeAllocator({12, 13, 15})
        previous = [10, 11]
        self.assertEqual(
            allocator.get_computed_block_ids(
                previous, [10, 11, 12, 13, 14, 15],
                skip_last_block_id=False),
            [10, 11, 12, 13],
        )

    def test_skip_last_preserves_full_prompt_guard(self):
        allocator = FakeAllocator({1, 2, 3})
        self.assertEqual(
            allocator.get_computed_block_ids(
                [], [1, 2, 3], skip_last_block_id=True),
            [1, 2],
        )


if __name__ == "__main__":
    unittest.main()
