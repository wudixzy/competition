# M1-12 Direct Prefix Fast-Forward - 2026-07-16

## Hypothesis

M1-11 showed that a 235000-token warm request with 234544 reported cached
tokens still executes 29 complete model forwards. Each staged pass restores a
successive GDN checkpoint and replays 16 tokens, except for the final 8-token
pass. The 29 passes spend 23.46 seconds in full attention and 9.54 seconds in
MoE; GDN state restore itself takes only 0.225 seconds.

The cold request already saved an exact GDN state at token 234992 and the KV
cache has the same contiguous prefix. Selecting that checkpoint on the first
scheduler step should make the existing model-runner partial-hit path execute
only the final 8 tokens. This removes redundant complete-model work without
changing model mathematics, cache contents, evaluator YAML, or kernel code.

## Contract

The scheduler now distinguishes:

- logical progress: tokens the engine may add to `num_computed_tokens` after
  the step;
- physical query tokens: suffix tokens sent to the model and charged to the
  scheduling budget.

Direct fast-forward is considered only for the first prefill step after the
block table is allocated. It requires a checkpoint key that is both an exact
scheduler-mirrored GDN state and a contiguous computed KV prefix strictly
shorter than the prompt. The logical boundary is:

```text
min(prompt_len, checkpoint_tokens + remaining_token_budget)
```

The physical budget is the difference between that boundary and the
checkpoint. If there is no useful exact checkpoint, the request is not the
first prefill, the suffix is invalid, or direct progress would not exceed the
normal chunk, the helper returns the unchanged staged plan.

`SequenceGroupMetadata.token_chunk_size` carries logical progress. The
existing model-runner prefix-hit path uses `computed_block_nums` to crop input
tokens to the physical suffix. `SchedulerOutputs.num_batched_tokens` counts
that suffix, and the engine updates sequence progress using the logical size.

## Fixed Gate

1. Pure scheduling tests cover no state, stale/non-contiguous state,
   non-first prefill, partial suffix, full-hit last-token behavior, and the
   exact `235000 -> checkpoint 234992 -> query 8` boundary.
2. Existing scheduler, GDN capture, paged-attention, and P0 static tests pass.
3. A runtime trace must show one warm prefill forward with context 234992 and
   query length 8, not 29 alignment forwards.
4. Cold and warm output hashes must match, reported cached tokens must not
   exceed the exact selected checkpoint, and cold behavior must be unchanged.
5. TP4 235K warm wall time must improve by at least 15%; full smoke and short
   decode performance must not regress.

No chunk, block, budget, YAML, or threshold scan is permitted. Any scheduler,
model-runner, state, cache, correctness, or restoration failure rejects the
candidate and restores production main.

## Status

`INTEGRATION`: 140 local tests pass with 13 environment skips. An additional
test invokes the real vLLM 0.6.3 `Scheduler._schedule_prefills` under the
CoreX runtime with a mocked block manager. It produced logical
`token_chunk_size=235000`, physical `num_batched_tokens=8`, one scheduled
group, and a running sequence as required.

The isolated instance exposes one healthy 32 GiB BI100 card, which is
insufficient to load the 35B FP16 model without the unavailable second healthy
GPU. TP4 API qualification therefore remains pending on the main four-card
instance. Production has not been changed and remains healthy.

## Repository Contract

Push only to private ModelHub. Do not push GitHub until the owner confirms that
repository is private. No automation may alter repository visibility.
