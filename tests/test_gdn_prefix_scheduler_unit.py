import ast
import pathlib
import typing
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEDULER = ROOT / "qwen3_6_scripts" / "scheduler.py"


def _load_helpers():
    tree = ast.parse(SCHEDULER.read_text(), filename=str(SCHEDULER))
    names = {
        "_select_gdn_prefix_checkpoint",
        "_make_gdn_prefix_checkpoint",
        "_limit_gdn_blocks_to_strict_prefix",
        "_accumulate_gdn_cached_tokens",
    }
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Iterable": typing.Iterable,
        "List": typing.List,
        "Optional": typing.Optional,
        "Tuple": typing.Tuple,
    }
    exec(compile(module, str(SCHEDULER), "exec"), namespace)
    return namespace


class GdnPrefixSchedulerTest(unittest.TestCase):

    def test_selects_longest_available_prefix(self):
        helpers = _load_helpers()
        select = helpers["_select_gdn_prefix_checkpoint"]
        self.assertEqual(
            select([(1, 2), (1, 2, 3, 4)], [1, 2, 3, 4, 5]),
            [1, 2, 3, 4])

    def test_rejects_prefix_without_state(self):
        helpers = _load_helpers()
        select = helpers["_select_gdn_prefix_checkpoint"]
        self.assertEqual(select([(7, 8)], [1, 2, 3]), [])

    def test_checkpoint_requires_exact_block_boundary(self):
        helpers = _load_helpers()
        make = helpers["_make_gdn_prefix_checkpoint"]
        self.assertEqual(
            make(list(range(300)), 0, 3678, 16), tuple(range(229)))
        self.assertEqual(
            make(list(range(300)), 0, 3680, 16), tuple(range(229)))

    def test_checkpoint_rejects_short_block_table(self):
        helpers = _load_helpers()
        make = helpers["_make_gdn_prefix_checkpoint"]
        self.assertIsNone(make([1], 0, 48, 16))

    def test_checkpoint_requires_new_complete_block(self):
        helpers = _load_helpers()
        make = helpers["_make_gdn_prefix_checkpoint"]
        self.assertIsNone(make(list(range(600)), 8192, 8200, 16))
        self.assertEqual(
            make(list(range(600)), 8192, 8712, 16), tuple(range(544)))

    def test_full_hit_is_limited_to_strict_prefix(self):
        helpers = _load_helpers()
        limit = helpers["_limit_gdn_blocks_to_strict_prefix"]
        self.assertEqual(limit(list(range(512)), 8192, 16), list(range(511)))
        self.assertEqual(limit([1], 1, 16), [])

    def test_staged_checkpoint_hits_accumulate_actual_skips(self):
        accumulate = _load_helpers()["_accumulate_gdn_cached_tokens"]
        cached = accumulate(None, 0, 511, 16)
        self.assertEqual(cached, 8176)
        cached = accumulate(cached, 8192, 1023, 16)
        self.assertEqual(cached, 16352)
        cached = accumulate(cached, 16384, 1535, 16)
        self.assertEqual(cached, 24528)

    def test_checkpoint_behind_computed_position_is_not_counted(self):
        accumulate = _load_helpers()["_accumulate_gdn_cached_tokens"]
        self.assertEqual(accumulate(None, 8192, 511, 16), 0)
        self.assertEqual(accumulate(8176, 16384, 1023, 16), 8176)


if __name__ == "__main__":
    unittest.main()
