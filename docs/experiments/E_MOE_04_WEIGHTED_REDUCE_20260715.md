# E-MOE-04: Fuse decode expert weighted reduction

## Scope

E-MOE-04 tested whether the T=1 routed-expert output weighting and top-k
reduction could be collapsed from an elementwise multiply plus reduction into
one CoreX GEMV dispatch. The fixed evaluator command and 256K context settings
were not changed.

```text
base:   2103876 (qualified E-MOE-03 winner)
branch: exp/E-MOE-04-weighted-reduce
code:   4ae0522
test:   2ca76d0
```

## Four-card microbenchmark

`tests/bench_moe_weighted_reduce.py` uses the real per-rank decode shapes:

```text
experts=256, top_k=8, hidden=2048, local_intermediate=128, dtype=float16
```

The candidate replaces:

```python
(expert_out * weights.unsqueeze(-1)).sum(0, keepdim=True)
```

with:

```python
torch.matmul(weights.unsqueeze(0), expert_out)
```

| GPU | Reduction speedup | Full routed-path speedup | Exact | Max abs |
| ---: | ---: | ---: | --- | ---: |
| 0 | 3.2404x | 1.0869x | no | 6.1035e-5 |
| 1 | 2.8662x | 1.0538x | no | 6.1035e-5 |
| 2 | 2.8523x | 1.0534x | no | 6.1035e-5 |
| 3 | 3.2087x | 1.0526x | no | 6.1035e-5 |

The reduction-only result is strong and the complete path improves on every
card, but the GEMV accumulation is not bit-exact with the previous FP16
multiply followed by sum.

Other scanned reductions (`mv`, `F.linear`, `einsum`, and pre-weighting the
activation) also changed FP16 results. None provided an exact faster path.

Artifacts are under the untracked remote directory:

```text
bench_runs/20260715_E_MOE_04/gpu0.json ... gpu3.json
```

## Integration gates

CoreX remote unit/static validation passed:

```text
MoE parity                 4/4
router/shared-gate tests   3/3
P0 static                 40/40
```

The candidate service started at max model length 262,144 with 16,878 GPU
blocks and 6,553 CPU blocks. Full API smoke passed 15/15, and the service log
contained no fatal error, OOM, non-finite value, worker loss, or segfault.

The sustained decode gate used the same 21-token prompt as E-MOE-02 and
E-MOE-03, with `min_tokens=max_tokens=1000`. It returned 1,000 completion
tokens with `finish_reason=length` in 76.693 seconds, but failed token equality:

```text
E-MOE-03 SHA256  1766c3c44bfb672e32b2e35419c5e06490e539e54250ab2fc1012c539e68835f
E-MOE-04 SHA256  be4ee3bd60f3f44646a16215cc51bc151babb87ea394864d5ef605dc25956c04
```

Prompt token count, completion token count, reasoning token count, and the
complete reasoning text matched. Generated content diverged immediately after
the first short `DECODE-1000` segment, confirming that this was model output
drift rather than a request mismatch.

## Decision

`REJECT AS PERFORMANCE WINNER`. Skip service A/B and long-context gates because
the sustained token-exact gate already failed. Keep the benchmark and reusable
sustained-decode client as diagnostic tools. Restore E-MOE-03 commit `2103876`
as the qualified runtime and leave `integration/perf-winners` unchanged.
