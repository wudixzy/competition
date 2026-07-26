# M1-58 TP4 Block-Major A/B Harness

## Purpose

`scripts/run_m1_58_block_major_kv_ab.sh` is the fixed model-level gate for the
M1-57/M1-58 block-major CacheEngine candidate. It requires one immutable
bare-host runtime and four healthy BI100 cards. Control and candidate use the
same model, tokenizer, command, full-attention accounting, CPU content cache,
GDN policy, restore mode, sampling semantics, and request order. The only A/B
selector is:

```text
control:   BI100_BLOCK_MAJOR_CPU_KV=0
candidate: BI100_BLOCK_MAJOR_CPU_KV=1
```

Both arms keep `BI100_CPU_KV_OFFLOAD=1`,
`BI100_HYBRID_KV_ACCOUNTING=full_attention`,
`BI100_GDN_CACHE_POLICY=admission64`,
`BI100_GDN_RESTORE_MODE=direct`, and all trace modes disabled. The formal
29-argument command remains unchanged.

## Fixed Workload

Each arm starts a fresh TP4 service and executes the same greedy sequence:

- one exact 65,536-token target cold request;
- one immediate target repeat;
- nine distinct exact 135,040-token cold pressure requests;
- the target again after pressure;
- one final target repeat;
- `max_tokens=8`, `temperature=0`, seed `20260721`;
- run identity `m158-block-major-fixed-20260726`.

The target plus pressure requests represent about 80,000 logical blocks, above
the observed full-attention GPU capacity even before the candidate's fixed
1,024-block staging reserve. The 4 GiB CPU tier is still large enough to retain
the evicted target prefix.

## Qualification

The harness fails closed unless all of the following pass:

1. hash-bound runtime identity, source revision, generated worker and
   CacheEngine, block-major module, and CoreX extension;
2. four-card deterministic preflight before control, after control, and after
   candidate, with at most 256 MiB post-cleanup free-memory drift;
3. 262144 startup capacity and exactly four rank-local runtime reports;
4. no block-major marker in control, and exact 1,024-block reservation plus
   block-major cache allocation on every candidate rank;
5. both arms restore at least 65,504 target tokens after pressure;
6. every request matches across arms for prompt, cached and completion token
   counts, `finish_reason`, and privacy-safe full-message SHA-256;
7. candidate restored-request elapsed time is at least `1.20x` faster;
8. aggregate cold and pure GPU-warm elapsed time each regress by no more than
   2%;
9. no fatal, OOM, SIGSEGV, traceback, Gloo/NCCL failure, worker loss, process
   leak, or GPU health drift.

The fixed comparison is implemented by
`tests/compare_m1_58_block_major_ab.py`. Parameters and thresholds are not
exposed as tuning arguments. Existing output directories are never overwritten.

## Scope

This harness has not been run because the current host has only three healthy
cards and TP3 is invalid for this model. Its local shell syntax and 30 focused
unit tests pass. The complete branch suite is 716 passed and 25 skipped;
submission preflight is 9/9.

Even a qualified M1-58 A/B establishes only model-level offload correctness,
capacity, and the candidate's incremental latency effect. It does not replace:

- the complete functional API gate;
- cold/warm long-context output checks at 4K through 262144;
- tool calling, reasoning, multimodal, structured output, and quality suites;
- admissible Output/Input/Cache TPS, TTFT, success-rate, hit-rate, and weighted
  score measurement;
- official 881-request evidence.

No `main`, formal YAML, default selector, model precision, request semantics,
or repository visibility change is authorized by this harness alone.
