"""
Fix: prefix_cache_hit stays True for chunked-prefill chunk 2+ even when past cache.

Root cause:
  model_runner.py _compute_for_prefix_cache_hit has three cases:
    Case 1: prefix_cache_len <= context_len  → "already past cache, do normal"
    Case 2: context_len < prefix_cache_len < seq_len  → partial hit, correct
    Case 3: seq_len <= prefix_cache_len  → full hit, reduce to 1 token

  Case 1 does nothing (leaves prefix_cache_hit = True). Then in utils.py:
    if inter_data.prefix_cache_hit:
        block_table = computed_block_nums   ← ONLY the original prefix blocks!

  But context_len > prefix_cache_len means chunk 1 tokens (between prefix_cache_len
  and context_len) are ALSO in KV cache and need to be in block_table.
  block_table = computed_block_nums misses all chunk-1 blocks.

  In _forward_prefix_pytorch:
    num_ctx_blocks = ceil(context_len / block_size)  # e.g. 268
    block_tables.shape[1] = len(computed_block_nums)  # e.g. 12  <-- too small!
    At tile_blk >= 12: blk_ids is empty → k_t shape [..., 0] → amax crash.

Fix:
  Set prefix_cache_hit = False for Case 1, so utils.py falls through to:
    elif chunked_prefill_enabled:
        block_table = block_tables[seq_id]   ← full block table (prefix + chunk1)
"""

from patch_utils import package_root, replace_once

MODEL_RUNNER = package_root("vllm") / "worker" / "model_runner.py"

OLD_BLOCK = """\
        if prefix_cache_len <= context_len:
            # We already passed the cache hit region,
            # so do normal computation.
            pass"""

NEW_BLOCK = """\
        if prefix_cache_len <= context_len:
            # We already passed the cache hit region,
            # so do normal computation.
            # Must clear prefix_cache_hit so _add_seq_group uses the full
            # block_tables (prefix + previous-chunk blocks) instead of only
            # computed_block_nums (prefix only).  Without this, block_tables
            # passed to _forward_prefix_pytorch is too narrow for context_len,
            # causing an empty blk_ids slice and a zero-dim amax() crash.
            inter_data.prefix_cache_hit = False"""

replace_once(MODEL_RUNNER, OLD_BLOCK, NEW_BLOCK, required=True)
