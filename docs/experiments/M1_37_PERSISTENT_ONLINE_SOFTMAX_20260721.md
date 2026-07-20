# M1-37 Persistent Online-Softmax Capability - 2026-07-21

## Context

M1-35 raised effective cache hit from `49.9301%` to `61.0671%`, but its
weighted proxy improved only `4.1381%`, below the fixed `5%` cache-stage gate.
M1-29 has no complete same-run 881-request content trace, so frequency-aware
KV eviction cannot be promoted from synthetic feasibility evidence.

The 235K profile attributes `68.788%` of model time to the ten full-attention
layers. Previous complete-fusion capability work was closed by the M1-28 WMMA
QK numerical gate. The optimization plan permits one split-reduction backup,
so M1-37 revisits only the dominant FP32 online-softmax update identified by
E-PREFIX-03.

## Fixed Design

E-PREFIX-06 used one 256-thread block per 512-score row. It reached
`5.2x-6.8x`, but five sparse rows were wrong when the q=8192 launch contained
49,152 blocks. M1-37 keeps its arithmetic unchanged and replaces only the
launch topology:

- tile width `512` and 256 threads are fixed;
- at most 1,024 persistent blocks are launched;
- each block processes `row += gridDim.x` serially;
- max, exponentiation, sum, correction, and `(m, l)` stay FP32;
- q=456 and q=8192 must both pass `max_abs <= 1e-3`,
  `relative_l2 <= 1e-5`, and speedup `>= 1.5x`.

This is an isolated capability probe. It does not patch `paged_attn.py`, build
a production extension, change Docker/YAML, or start or stop the service.

## CoreX Result

The extension compiled successfully on CoreX 3.2.3. Evidence is under
`/root/competition-m1-32-latest/bench_runs/m1_37/persistent_softmax`.

| Query | PyTorch median | Candidate median | Speedup | Result |
| ---: | ---: | ---: | ---: | --- |
| 456 | `0.2313 ms` | `0.0450 ms` | `5.1363x` | parity pass |
| 8192 | `3.9001 ms` | `0.6392 ms` | `6.1018x` | parity fail |

For q=456, exponentiated scores, running maxima, and corrections were exact;
running-sum maximum error was `6.1035e-5` and relative L2 was `5.2801e-8`.
For q=8192:

| State | Maximum absolute error | Relative L2 |
| --- | ---: | ---: |
| exponentiated scores | `0.3150419` | `1.0341e-3` |
| running sum | `37.8974` | `6.8682e-4` |
| running maximum | `0` | `0` |
| correction | `0` | `0` |

The persistent topology therefore preserves the speed potential but does not
remove the sparse large-row numerical corruption. Stderr contained only the
existing `pynvml` deprecation warning.

The remote monitoring command piped stdout and accidentally recorded the
pipeline's final command as `probe.rc=0`. The JSON is authoritative:
`qualified=false`, `parity_ok=false`, `speed_ok=true`. A fail-closed check of
that JSON produced `qualification.rc=1` without rerunning the benchmark.

## Decision

Status: `NUMERICAL_REJECTED`.

Do not integrate this kernel and do not scan grid size, block count, thread
count, tile width, compiler flags, random inputs, or tolerance. Together with
the already rejected complete-fusion capability, this closes the predeclared
split-reduction backup. Production prefix attention remains unchanged. The
next step is another architecture review, not a variant of this reduction.
