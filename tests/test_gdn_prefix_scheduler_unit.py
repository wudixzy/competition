import ast
from dataclasses import dataclass, field
import hashlib
import pathlib
import typing
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEDULER = ROOT / "qwen3_6_scripts" / "scheduler.py"


def _digest(value: int) -> bytes:
    return hashlib.sha256(f"{value}".encode("ascii")).digest()


def _load_plan() -> dict:
    tree = ast.parse(SCHEDULER.read_text(), filename=str(SCHEDULER))
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_plan_gdn_prefix_fast_forward"
    ]
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Iterable": typing.Iterable,
        "List": typing.List,
        "Optional": typing.Optional,
        "Tuple": typing.Tuple,
        "GdnPrefixKey": typing.Tuple[int, bytes],
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

    def test_direct_fast_forward_stretches_logical_chunk(self):
        plan = _load_plan()["_plan_gdn_prefix_fast_forward"]
        restore = (14687, _digest(14687))
        self.assertEqual(
            plan(restore, 0, 235000, 8192, 8192, 16),
            (235000, 8),
        )

    def test_direct_fast_forward_can_leave_a_physical_suffix(self):
        plan = _load_plan()["_plan_gdn_prefix_fast_forward"]
        restore = (32, _digest(32))
        # logical=512+8192, physical budget carries only 8192 tokens.
        self.assertEqual(
            plan(restore, 0, 20000, 256, 8192, 16),
            (8704, 8192),
        )

    def test_direct_fast_forward_does_not_advance_without_gain(self):
        plan = _load_plan()["_plan_gdn_prefix_fast_forward"]
        self.assertEqual(plan(None, 0, 2000, 512, 512, 16), (512, 512))
        self.assertEqual(plan((1, _digest(1)), 0, 40, 128, 128, 16), (128, 128))
        self.assertEqual(plan((4, _digest(4)), 16, 2000, 512, 512, 16), (512, 512))

    def test_direct_fast_forward_fails_when_budget_is_invalid(self):
        plan = _load_plan()["_plan_gdn_prefix_fast_forward"]
        restore = (8, _digest(8))
        self.assertEqual(plan(restore, 0, 2000, 512, 0, 16), (512, 512))
        self.assertEqual(plan(restore, 0, 2000, 512, 512, 0), (512, 512))

    def test_plan_inputs_are_expected_stable_key_shape(self):
        plan = _load_plan()["_plan_gdn_prefix_fast_forward"]
        restore = (64, _digest(64))
        num_new_tokens, num_physical_tokens = plan(
            restore, 0, 4096, 256, 128, 16)
        self.assertEqual((num_new_tokens, num_physical_tokens), (1152, 128))
        self.assertIsInstance(restore[0], int)
        self.assertEqual(len(restore[1]), 32)


if __name__ == "__main__":
    unittest.main()
