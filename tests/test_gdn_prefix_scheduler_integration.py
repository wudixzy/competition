from __future__ import annotations

import hashlib
import importlib.util
import unittest
from collections import deque
from types import SimpleNamespace


@unittest.skipIf(importlib.util.find_spec("torch") is None,
                 "runtime vLLM dependencies are unavailable")
class GdnPrefixSchedulerIntegrationTest(unittest.TestCase):

    def test_prefill_separates_logical_progress_from_physical_budget(self):
        from qwen3_6_scripts.gdn_prefix import (
            GdnPrefixStatePolicy,
            final_capture_key,
            keys_from_block_hashes,
            strict_prefix_block_count,
        )
        from qwen3_6_scripts.scheduler import (
            AllocStatus,
            Scheduler,
            SchedulingBudget,
            SequenceStatus,
        )

        prompt_len = 235000
        block_size = 16
        block_count = strict_prefix_block_count(prompt_len, block_size)

        def digest(value: int) -> bytes:
            return hashlib.sha256(f"{value}".encode("ascii")).digest()

        class Data:
            @staticmethod
            def get_num_computed_tokens():
                return 0

            @staticmethod
            def get_len():
                return prompt_len

        class Sequence:
            def __init__(self):
                self.status = SequenceStatus.WAITING
                self.data = Data()

            def get_num_new_tokens(self):
                return prompt_len

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

            @staticmethod
            def is_prefill():
                return True

        class BlockManager:
            allocated = False
            content_hashes = keys_from_block_hashes(
                [digest(i) for i in range(1, block_count + 1)])
            policy: GdnPrefixStatePolicy = GdnPrefixStatePolicy("fine32")
            policy.admit([content_hashes[-1]])

            @staticmethod
            def can_allocate(*args, **kwargs):
                return AllocStatus.OK

            def allocate(self, seq_group):
                self.allocated = True

            @staticmethod
            def get_common_computed_block_ids(sequences):
                return list(range(block_count))

            def get_content_hashes(self, seq):
                return list(BlockManager.content_hashes)

            @staticmethod
            def get_block_table(seq):
                return []

            @staticmethod
            def access_all_blocks_in_seq(seq, now):
                return None

            @staticmethod
            def mark_blocks_as_computed(seq_group, token_chunk_size):
                return None

        sequence = Sequence()
        group = Group(sequence)
        block_manager = BlockManager()
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.waiting = deque([group])
        scheduler.block_manager = block_manager
        scheduler.scheduler_config = SimpleNamespace(
            is_multi_step=False,
            num_scheduler_steps=1,
            max_num_batched_tokens=8192,
        )
        scheduler.cache_config = SimpleNamespace(
            enable_prefix_caching=True,
            block_size=16,
        )
        scheduler.lora_config = None
        scheduler.prev_prompt = False
        scheduler._gdn_prefix_policy = block_manager.policy
        scheduler._gdn_restore_mode = "direct"
        scheduler._gdn_replay_alignment = scheduler.scheduler_config.max_num_batched_tokens
        scheduler._gdn_request_restore_keys = {}
        scheduler._gdn_request_capture_targets = {}
        scheduler._passed_delay = lambda now: True
        scheduler._get_num_new_tokens = lambda *args, **kwargs: 8192
        scheduler._get_prompt_limit = lambda seq_group: 262144
        scheduler._get_num_lookahead_slots = lambda *args, **kwargs: 0

        budget = SchedulingBudget(token_budget=8192, max_num_seqs=1)
        output = scheduler._schedule_prefills(
            budget, curr_loras=None, enable_chunking=True)

        expected_restore_key = block_manager.content_hashes[-1]
        self.assertTrue(block_manager.allocated)
        self.assertEqual(sequence.status, SequenceStatus.RUNNING)
        self.assertEqual(len(output.seq_groups), 1)
        self.assertEqual(output.seq_groups[0].token_chunk_size, 235000)
        self.assertEqual(budget.num_batched_tokens, 8)
        self.assertEqual(budget.num_scheduled_tokens, 235000)

        restore_key = scheduler._gdn_request_restore_keys["m1-12"]
        self.assertEqual(restore_key, expected_restore_key)
        self.assertIsInstance(restore_key[1], (bytes, bytearray))
        self.assertEqual(len(restore_key[1]), 32)

        capture_targets = scheduler._gdn_request_capture_targets["m1-12"]
        self.assertEqual(capture_targets, (final_capture_key(
            [k[1] for k in block_manager.content_hashes],
            prompt_len,
            block_size,
            "direct",
            scheduler._gdn_replay_alignment),))

        self.assertEqual(capture_targets[0][1], restore_key[1])
        self.assertEqual(capture_targets[0][0], restore_key[0])


if __name__ == "__main__":
    unittest.main()
