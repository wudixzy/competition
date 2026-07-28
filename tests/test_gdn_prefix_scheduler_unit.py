import ast
from dataclasses import dataclass, field
import hashlib
import pathlib
from types import SimpleNamespace
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


def _load_namespace_release():
    tree = ast.parse(SCHEDULER.read_text(), filename=str(SCHEDULER))
    scheduler_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler")
    function = next(
        node for node in scheduler_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_free_seq_group_cross_attn_blocks")
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"SequenceGroup": typing.Any}
    exec(compile(module, str(SCHEDULER), "exec"), namespace)
    return namespace["_free_seq_group_cross_attn_blocks"]


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

    def test_hybrid_fast_forward_preserves_canonical_chunk_boundary(self):
        plan = _load_plan()["_plan_gdn_prefix_fast_forward"]
        restore = (192, _digest(192))
        self.assertEqual(
            plan(restore, 0, 16000, 8192, 8192, 16, 8192),
            (8192, 5120),
        )

    def test_hybrid_fast_forward_reports_exact_short_suffix(self):
        plan = _load_plan()["_plan_gdn_prefix_fast_forward"]
        restore = (255, _digest(255))
        self.assertEqual(
            plan(restore, 0, 4096, 4096, 8192, 16, 8192),
            (4096, 16),
        )

    def test_hybrid_fast_forward_rejects_invalid_alignment(self):
        plan = _load_plan()["_plan_gdn_prefix_fast_forward"]
        restore = (192, _digest(192))
        with self.assertRaises(ValueError):
            plan(restore, 0, 16000, 8192, 8192, 16, 1000)

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

    def test_request_namespace_release_covers_all_group_types(self):
        release = _load_namespace_release()

        class BlockManager:

            def __init__(self):
                self.cross_freed = []
                self.released = []

            def free_cross(self, seq_group):
                self.cross_freed.append(seq_group.request_id)

            def release_request_cache_namespace(self, request_id):
                self.released.append(request_id)

        manager = BlockManager()
        scheduler = SimpleNamespace(block_manager=manager)
        decoder = SimpleNamespace(
            request_id="decoder", is_encoder_decoder=lambda: False)
        encoder = SimpleNamespace(
            request_id="encoder", is_encoder_decoder=lambda: True)

        release(scheduler, decoder)
        release(scheduler, encoder)
        self.assertEqual(manager.cross_freed, ["encoder"])
        self.assertEqual(manager.released, ["decoder", "encoder"])

    def test_all_terminal_group_paths_use_namespace_release_helper(self):
        source = SCHEDULER.read_text()
        self.assertEqual(
            source.count("self._free_seq_group_cross_attn_blocks("), 3)
        self.assertIn(
            "for aborted_group in aborted_groups:", source)
        self.assertIn(
            "if seq_group.is_finished():", source)
        self.assertIn(
            "if self._async_stopped:", source)

    def test_request_namespace_release_is_legacy_manager_compatible(self):
        release = _load_namespace_release()
        legacy_manager = SimpleNamespace()
        scheduler = SimpleNamespace(block_manager=legacy_manager)
        decoder = SimpleNamespace(
            request_id="decoder", is_encoder_decoder=lambda: False)
        release(scheduler, decoder)

    def test_request_namespace_releases_when_cross_free_raises(self):
        release = _load_namespace_release()

        class BlockManager:

            def __init__(self):
                self.released = []

            def free_cross(self, seq_group):
                raise RuntimeError("cross free failed")

            def release_request_cache_namespace(self, request_id):
                self.released.append(request_id)

        manager = BlockManager()
        scheduler = SimpleNamespace(block_manager=manager)
        encoder = SimpleNamespace(
            request_id="encoder", is_encoder_decoder=lambda: True)
        with self.assertRaisesRegex(RuntimeError, "cross free failed"):
            release(scheduler, encoder)
        self.assertEqual(manager.released, ["encoder"])


if __name__ == "__main__":
    unittest.main()
