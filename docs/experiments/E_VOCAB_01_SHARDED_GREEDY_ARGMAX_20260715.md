# E-VOCAB-01: Sharded greedy argmax

## Goal

Avoid gathering the full 248,320-token vocabulary to rank 0 for every greedy
decode token. Each TP rank contributes only its local maximum value and global
token id through the existing IxFormer IPC all-reduce path.

## Eligibility

The fast path is limited to TP=4 decode-only batches with one greedy sample per
sequence and no penalties, logprobs, guided decoding, logit bias, minimum-token
constraint, LoRA vocabulary extension, or speculative/deferred sampler output.
All other requests use the original full-vocabulary gather and sampler.

Runtime switch: `BI100_SHARDED_GREEDY_ARGMAX=0|1`.

## Microbenchmark

New BI100 four-GPU instance, 1,000 random exactness cases plus cross-rank ties,
positive infinity, NaN, and all-negative-infinity cases:

- Exactness: all passed.
- Full gather median: 8.52 ms per token.
- Candidate-table all-reduce median: 0.218 ms per token.
- Median saving: 8.31 ms per token (39.1x for this operation).

## Status

Candidate integration is implemented on the experiment branch. Production
merge remains blocked on API correctness and repeated same-instance TP4 A/B.
