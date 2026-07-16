import ast
from dataclasses import dataclass, field
import pathlib
import typing
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEDULER = ROOT / "qwen3_6_scripts" / "scheduler.py"
MODEL = ROOT / "qwen3_6_scripts" / "qwen3_5.py"


def _assigned_attribute_int(path, attribute):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        if not isinstance(value, ast.Constant) or not isinstance(value.value, int):
            continue
        if any(isinstance(target, ast.Attribute)
               and target.attr == attribute for target in targets):
            return value.value
    raise AssertionError(f"missing integer assignment for {attribute} in {path}")


def _load_helpers():
    tree = ast.parse(SCHEDULER.read_text(), filename=str(SCHEDULER))
    names = {
        "_select_gdn_prefix_checkpoint",
        "_make_gdn_prefix_checkpoint",
        "_limit_gdn_blocks_to_strict_prefix",
        "_accumulate_gdn_cached_tokens",
        "_plan_gdn_prefix_fast_forward",
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


def _load_scheduling_budget():
    tree = ast.parse(SCHEDULER.read_text(), filename=str(SCHEDULER))
    budget_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SchedulingBudget")
    module = ast.Module(body=[budget_class], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "dataclass": dataclass,
        "field": field,
        "Dict": typing.Dict,
        "Optional": typing.Optional,
        "Set": typing.Set,
    }
    exec(compile(module, str(SCHEDULER), "exec"), namespace)
    return namespace["SchedulingBudget"]


class GdnPrefixSchedulerTest(unittest.TestCase):

    def test_budget_separates_physical_and_logical_tokens(self):
        budget = _load_scheduling_budget()(token_budget=8192, max_num_seqs=1)
        budget.add_num_batched_tokens(
            "m1-12", 8, num_scheduled_tokens=235000)
        self.assertEqual(budget.num_batched_tokens, 8)
        self.assertEqual(budget.num_scheduled_tokens, 235000)
        self.assertEqual(budget.remaining_token_budget(), 8184)

        budget.subtract_num_batched_tokens("m1-12", 8)
        self.assertEqual(budget.num_batched_tokens, 0)
        self.assertEqual(budget.num_scheduled_tokens, 0)

    def test_scheduler_outputs_report_logical_token_count(self):
        source = SCHEDULER.read_text()
        self.assertEqual(
            source.count(
                "num_batched_tokens=budget.num_scheduled_tokens"), 2)

    def test_checkpoint_capacity_covers_native_context(self):
        scheduler_capacity = _assigned_attribute_int(
            SCHEDULER, "_gdn_prefix_checkpoint_max")
        worker_capacity = _assigned_attribute_int(
            MODEL, "_gdn_prefix_cache_max")
        required_capacity = (262144 + 8192 - 1) // 8192
        self.assertEqual(scheduler_capacity, worker_capacity)
        self.assertGreaterEqual(scheduler_capacity, required_capacity)

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

    def test_direct_fast_forward_uses_longest_exact_checkpoint(self):
        plan = _load_helpers()["_plan_gdn_prefix_fast_forward"]
        computed = list(range(62))
        checkpoints = [tuple(range(31)), tuple(range(62))]
        self.assertEqual(
            plan(checkpoints, computed, 0, 1000, 128, 128, 16),
            (1000, 8))

    def test_direct_fast_forward_matches_235k_profile_boundary(self):
        plan = _load_helpers()["_plan_gdn_prefix_fast_forward"]
        checkpoint = tuple(range(14687))
        self.assertEqual(
            plan([checkpoint], list(range(14687)), 0,
                 235000, 8192, 8192, 16),
            (235000, 8))

    def test_direct_fast_forward_can_leave_a_physical_suffix(self):
        plan = _load_helpers()["_plan_gdn_prefix_fast_forward"]
        checkpoint = tuple(range(50))
        self.assertEqual(
            plan([checkpoint], list(range(62)), 0, 1000, 128, 128, 16),
            (928, 128))

    def test_direct_fast_forward_fails_closed(self):
        plan = _load_helpers()["_plan_gdn_prefix_fast_forward"]
        fallback = (128, 128)
        self.assertEqual(plan([], list(range(62)), 0, 1000, 128, 128, 16),
                         fallback)
        self.assertEqual(
            plan([tuple(range(62))], list(range(31)), 0,
                 1000, 128, 128, 16), fallback)
        self.assertEqual(
            plan([tuple(range(62))], list(range(62)), 128,
                 1000, 128, 128, 16), fallback)

    def test_direct_fast_forward_keeps_full_hit_last_token(self):
        plan = _load_helpers()["_plan_gdn_prefix_fast_forward"]
        strict_checkpoint = tuple(range(63))
        self.assertEqual(
            plan([strict_checkpoint], list(range(64)), 0,
                 1024, 128, 128, 16),
            (1024, 16))


if __name__ == "__main__":
    unittest.main()
