from __future__ import annotations

import importlib.util
import unittest
from collections import OrderedDict, deque
from types import SimpleNamespace


@unittest.skipIf(importlib.util.find_spec("torch") is None,
                 "runtime vLLM dependencies are unavailable")
class GdnPrefixSchedulerIntegrationTest(unittest.TestCase):
    def test_prefill_separates_logical_progress_from_physical_budget(self):
        from qwen3_6_scripts.scheduler import (
            AllocStatus,
            Scheduler,
            SchedulingBudget,
            SequenceStatus,
        )

        class Data:
            @staticmethod
            def get_num_computed_tokens():
                return 0

            @staticmethod
            def get_len():
                return 235000

        class Sequence:
            def __init__(self):
                self.status = SequenceStatus.WAITING
                self.data = Data()

        class Group:
            request_id = "m1-12"
            lora_int_id = 0

            def __init__(self, sequence):
                self.sequence = sequence

            def get_seqs(self, status=None):
                if status is None or self.sequence.status == status:
                    return [self.sequence]
                return []

            @staticmethod
            def get_max_num_running_seqs():
                return 1

            @staticmethod
            def init_multi_step_from_lookahead_slots(*args, **kwargs):
                return None

        class BlockManager:
            allocated = False

            @staticmethod
            def can_allocate(*args, **kwargs):
                return AllocStatus.OK

            def allocate(self, seq_group):
                self.allocated = True

            @staticmethod
            def get_common_computed_block_ids(sequences):
                return list(range(14687))

        sequence = Sequence()
        group = Group(sequence)
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.waiting = deque([group])
        scheduler.block_manager = BlockManager()
        scheduler.scheduler_config = SimpleNamespace(
            is_multi_step=False, num_scheduler_steps=1)
        scheduler.cache_config = SimpleNamespace(
            enable_prefix_caching=True, block_size=16)
        scheduler.lora_config = None
        scheduler.prev_prompt = False
        scheduler._gdn_prefix_checkpoints = OrderedDict(
            [(tuple(range(14687)), None)])
        scheduler._passed_delay = lambda now: True
        scheduler._get_num_new_tokens = lambda *args, **kwargs: 8192
        scheduler._get_prompt_limit = lambda seq_group: 262144
        scheduler._get_num_lookahead_slots = lambda *args, **kwargs: 0

        budget = SchedulingBudget(token_budget=8192, max_num_seqs=1)
        output = scheduler._schedule_prefills(
            budget, curr_loras=None, enable_chunking=True)

        self.assertTrue(scheduler.block_manager.allocated)
        self.assertEqual(sequence.status, SequenceStatus.RUNNING)
        self.assertEqual(len(output.seq_groups), 1)
        self.assertEqual(output.seq_groups[0].token_chunk_size, 235000)
        self.assertEqual(budget.num_batched_tokens, 8)


if __name__ == "__main__":
    unittest.main()
