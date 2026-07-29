# M1-126 grouped-QK split512 paged prefill

Date: 2026-07-29

Status: exact CoreX build and fixed four-card component A/B completed.
Numerical qualification passed, but all four performance cases regressed.
The candidate is rejected and the grouped-QK route is stopped.

## Decision basis

M1-124 replaced four 512-token QK/PV partitions with one 2048-token QK/PV
partition. Its fixed four-card component A/B was lifecycle-clean but failed
the predeclared numerical and performance gates:

- output relative L2 ranged from `1.133e-5` to `2.267e-5`, above `1e-5`;
- median M1-109/M1-124 speedup was `0.9914x`;
- only the 235K case improved, by approximately `0.22%`.

M1-124 therefore cannot enter a TP4 service experiment.

The M1-113 design allowed one fallback before stopping this route. M1-126
keeps one grouped QK GEMM over at most 2048 key tokens and all four GQA query
heads, but restores M1-109's 512-token FP32 online-softmax and PV boundaries.
This isolates whether reducing four QK calls to one has value without the
floating-point partition change and 2048-token PV behavior that invalidated
M1-124.

## Fixed contract

- FP16 inputs and outputs with FP32 QK, softmax, PV, and accumulation;
- head dimension 256, block size 16, GQA 4:1;
- query length at most 8192 and total sequence length at most 262144;
- one QK call per 2048-token group;
- sequential 512-token online-softmax updates and one PV call per active
  512-token split;
- explicit zero fill and masking for partial tail splits;
- bounded workspaces and no full-sequence logits;
- no runtime selector, request-semantic, model, tokenizer, YAML, or default
  change.

The unchanged fixed component gate compares the qualified M1-109 binary with
M1-126 on dense, 65K, 128K, and 235K shapes. It requires finite outputs,
relative L2 at most `1e-5`, output max-absolute error at most `1e-3`, median
speedup at least `1.10x`, at least three positive cases, no case regression
over 2%, and clean scoped lifecycle gates.

If M1-126 fails numerical or performance qualification, the grouped-QK
route stops. No additional tile, split, or YAML scan is authorized.

Passing this component gate would authorize only a fresh TP4 service A/B.
It would not authorize changing the runtime default, `computility-run.yaml`,
`main`, repository visibility, or making an official-score claim.

## Verification

Local validation at source commit
`214e4af9320b94c186aaa4f83ed4ba177bef4db9`:

- focused M1-109/M1-113/M1-126 tests: 18 passed;
- complete unit suite: 1101 passed, 13 skipped;
- submission preflight: 9/9 passed;
- shell syntax and `git diff --check`: passed.

The exact source compiled under CoreX 3.2.3 for `ivcore10`. The resulting
247,184-byte extension has SHA-256
`9eb96ad611df0837ad972a14135af8d22a136d6b6620db7856a7d1a18b542fbf`
and no unresolved dynamic dependencies under the production library path.

## Fixed BI100 result

The component A/B compared M1-109 with M1-126 on the same GPU for each fixed
shape and alternated execution order by GPU.

| Case | M1-109 ms | M1-126 ms | Old/new | Output rel-L2 |
|---|---:|---:|---:|---:|
| Dense q8176 | 39.829 | 52.924 | 0.753x | 4.723e-6 |
| 65K q8176 | 293.935 | 407.736 | 0.721x | 6.174e-6 |
| 128K q8176 | 516.400 | 717.253 | 0.720x | 6.054e-6 |
| 235K q5616 | 658.516 | 915.963 | 0.719x | 6.625e-6 |

All outputs and LSE values were finite. Output relative L2 was at most
`6.625e-6`, LSE relative L2 was at most `1.911e-8`, and output maximum
absolute error was at most `2.441e-4`. Restoring the split512 arithmetic
therefore recovered the M1-109 numerical behavior exactly within the
reported metrics.

Performance did not qualify. Median M1-109/M1-126 speedup was `0.7204x`;
no case improved, and all cases regressed by more than 2%. This shows that
flattening the four GQA heads into the GEMM column dimension is materially
worse on this CoreX/cuBLAS path even when the softmax/PV arithmetic contract
is restored.

All four cells, scoped recovery, cleanup qualification, pre/postflight,
repeated four-GPU compute checks, fatal scan, and timeout scan passed. The
runner return code of one is solely the expected performance rejection.

## Decision

M1-124 changed both QK/PV grouping and online-softmax partitioning and failed
numerical and performance gates. M1-126 retained the original partition and
passed numerical gates, but performed substantially worse. These are the two
predeclared reasonable grouped-QK designs. This route is closed; no further
group size, tile size, split count, or YAML scan is justified.

M1-109 remains the only component-qualified fused-prefill binary. Work returns
to its complete TP4 quality/cache validation and to profile-driven data-flow
changes that do not flatten GQA heads into this cuBLAS matrix shape.
