# E-GDN-04: Elide the read-only GDN recurrence submatrix copy

Date: 2026-07-14

## Hypothesis

Each of the 63 GDN intra-chunk row updates cloned the already-computed
`attn[..., :i, :i]` submatrix before reading it. The assignment only writes
row `i`, so the submatrix cannot overlap the destination. Reusing the view
should remove quadratic data movement without changing any floating-point
operation, result, state, evaluator argument, or cache behavior.

## Manifest

```text
baseline: b1d95009d52135a5b00bbac1c5ccc682c4539644
candidate: 2aa779be46cba1c91e49b1c680f4ba267470cad9
branch: exp/E-GDN-04-recurrence-copy-elision
model: /root/public-storage/models/Qwen/Qwen3.6-35B-A3B
hardware: 4 x BI-V100-50C-200G, TP=4
seed: 123
fixed evaluator contract changed: no
```

## Change

The production change is one line in `qwen3_6_scripts/qwen3_5.py`:

```python
# Before
sub = attn[..., :i, :i].clone()

# Candidate
sub = attn[..., :i, :i]
```

`row.clone()`, elementwise multiplication, reduction order, addition, and the
in-place row assignment remain unchanged. The experiment adds a four-device
benchmark, a static guard, and a bitwise production/reference assertion.

## Qualification gates

- per-device tensor preflight: 4/4 pass
- four-rank collective preflight: 4/4 pass, value `10.0`
- P0 static tests: 41/41 pass
- GDN production/reference tests: 2/2 pass with `torch.equal`
- complete CoreX unit discovery: 123 pass, 1 skipped
- Python compilation and `git diff --check`: pass
- installed runtime SHA equals repository SHA:
  `368edbf0d259956f29568ea669812f2f38e62ffeea16e1764e7725d38cda113c`
- candidate startup profile: two consecutive service starts pass
- deterministic API oracle: 3/3 exact for message, finish reason, and usage
- full API smoke: 15/15 pass, including image, tool-call, JSON-schema, and
  prefix-cache cases
- long-output hash: 3/3 exact for complete output, SHA256, HTTP status,
  finish reason, and usage
- candidate service error-log scan: zero fatal matches

## Primitive results

The benchmark used 12 heads per rank, chunk size 64, float32 recurrence
matrices, and all four devices. Every current/candidate output was bitwise
equal and finite.

| Tokens | Current median (ms) | Candidate median (ms) | Speedup | Peak bytes saved |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 4.503-4.767 | 3.726-3.886 | 1.208-1.241x | 190,976 |
| 256 | 4.908-5.161 | 4.027-4.387 | 1.174-1.219x | 762,368 |
| 1,024 | 5.832-6.362 | 4.875-5.194 | 1.196-1.230x | 3,048,448 |
| 8,192 | 10.490-11.200 | 8.200-8.832 | 1.264-1.279x | 25,165,824 |
| 16,384 | 16.371-17.393 | 12.549-13.411 | 1.288-1.305x | 48,771,072 |

Artifacts:

```text
bench_runs/20260715_E_GDN_04/gpu0.json
bench_runs/20260715_E_GDN_04/gpu1.json
bench_runs/20260715_E_GDN_04/gpu2.json
bench_runs/20260715_E_GDN_04/gpu3.json
```

## Service capacity

Both candidate starts reported 16,852 GPU blocks and 6,553 CPU blocks. The
interleaved baseline start reported 16,871 and 6,553. The repeatable 19-block
candidate difference is small (`0.11%`) but is not explained by a retained
candidate tensor and cannot be credited as a memory improvement.

## Strict long-prefill pairs

Every sample used four serial requests, prompt repeat 600, 16 generated
tokens, seed 123, and the same salt on candidate and baseline. Each sample had
33,804 prompt tokens, 8,508 uncached prompt tokens, 25,296 cached tokens, and
64 completion tokens with 100% success.

Candidate run 1 versus the interleaved baseline:

| Pair | Weighted score change | TTFT P90 change | Wall change |
| --- | ---: | ---: | ---: |
| P1 | +13.67% | -17.72% | -12.38% |
| P2 | -0.94% | +2.13% | +0.88% |
| P3 | +0.24% | +1.59% | -0.25% |

The P1 baseline was an outlier. The median candidate result was only `+0.42%`
for weighted score and `+0.45%` for input/cache TPS, while TTFT regressed by
`0.94%`.

The reverse-order candidate run removed the P1 outlier advantage:

| Pair | Weighted score change | TTFT P90 change | Wall change |
| --- | ---: | ---: | ---: |
| P1 | +0.78% | +0.92% | -0.69% |
| P2 | -1.07% | +2.06% | +1.14% |
| P3 | -0.89% | +1.75% | +0.83% |

The reverse-run median changed weighted score by `-1.07%`, TTFT by `+2.06%`,
and wall time by `+1.14%`. Across six candidate samples, candidate and baseline
each won three weighted-score comparisons. The endpoint effect is within
device and startup variability and is not a qualifying improvement.

Artifacts:

```text
bench_runs/20260715_E_GDN_04/oracle-comparison.json
bench_runs/20260715_E_GDN_04/output-hash-comparison.json
bench_runs/20260715_E_GDN_04/smoke-full.json
bench_runs/20260715_E_GDN_04/candidate-LONG-P*.json
bench_runs/20260715_E_GDN_04/candidate-r2-LONG-P*.json
bench_runs/20260715_E_GDN_04/baseline-LONG-P*.json
```

## Decision

`REJECT` the production-path change. It is bitwise exact and improves the
isolated recurrence by 1.17x to 1.31x, but it does not produce stable endpoint
gains, the reverse-run median is negative, and both candidate starts expose
19 fewer GPU blocks than the interleaved baseline. Keep the benchmark and
implementation on the experiment branch as evidence; do not merge it into
`integration/perf-winners`.

Further GDN work should target a larger region than one recurrence copy, or
move to the decode preparation/recurrent-update path where launch overhead is
paid for every generated token.
