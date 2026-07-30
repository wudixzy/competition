# M1-162 calibrated FP16-QK reassessment

## Status

The frozen fresh-seed screen passed on `ssh-73ca29ba` at source
`eea631a68beaa66a2b4e84346cef2912d5f59e8f`. It authorizes only
real-activation capture/replay; it does not authorize TP4 service, quality,
`main`, YAML, or production changes.

## Why this route is reopened

M1-157 and M1-158 reduced the qualified M1-109 operator latency by 15.3%-17.2%
at the 16K, 32K, and 64K positions relevant to submitted TTFT P90. Their
outputs and LSE were finite, max-absolute output error was at most
`6.104e-5`, and LSE relative L2 was at most `3.66e-8`. They were rejected
because candidate-versus-rounded-reference output relative L2 was
`1.811e-5` to `1.930e-5`, above a fixed `1e-5` threshold.

That threshold conflated two roles. Candidate-versus-rounded output drift is
important distribution evidence, but the hard operator question is whether
the candidate remains within the ordinary FP16 representation error around
the same FP32 reference. Task semantics may not waive a true numeric failure,
and a changed greedy trajectory is not by itself such a failure.

## Frozen v2 numeric contract

`quality/fused_prefill_numeric_adjudication.v2.json` requires the candidate,
FP32 reference, rounded FP16 reference, and repeated candidate result to be
finite. Candidate error versus the same FP32 online-softmax reference must be
no more than twice the normal FP16-rounding error for both relative L2 and
maximum absolute error. The denominator floor is `1e-12`, candidate LSE
relative L2 must be at most `1e-5`, and the repeated candidate output and LSE
must be bit-exact.

Candidate-versus-rounded-reference relative L2 and max-absolute error remain
reported as distribution-drift diagnostics. They are not allowed to hide a
failed FP32-calibrated ratio, nor can later semantic tests override a numeric
failure.

## Fresh screen

The new screen does not reuse the historical seed:

| Case | Context | Query | Physical GPU | Seed |
| --- | ---: | ---: | ---: | ---: |
| P90 16K | 8192 | 8176 | 1 | 20260730 |
| P90 32K | 24576 | 8176 | 2 | 20260731 |
| P90 64K | 57344 | 8176 | 3 | 20260732 |

Each cell interleaves five M1-109 and FP16-QK trials on identical tensors.
The per-cell speed floor is `0.98x`; the three-cell median must be at least
`1.08x`. Source revision, extension hashes, contract hash, GPU assignment,
cleanup, postflight, repeated preflight, memory comparison, and fatal scan
remain fail-closed.

The runner reuses the M1-157 lifecycle implementation but changes its cell
script, report identities, runtime label, and next-stage authorization. The
original M1-157 defaults and historical evidence are unchanged.

## Observed result

The baseline extension SHA-256 was
`ad4ea7707bb2f2bfe04e07a7ad5fd58a647232be70a3056937a0d738c8254bff`.
The freshly compiled FP16-QK candidate SHA-256 was
`36e043f138aa87c635178e4aa6a30af710b87c3f3d7c2a3f1838fc0e365bd368`.

| Case | Baseline ms | Candidate ms | Speedup | Relative-L2 rounding multiple | Max-abs rounding multiple |
| --- | ---: | ---: | ---: | ---: | ---: |
| P90 16K | 72.086 | 62.487 | 1.154x | 1.000007 | 1.000000 |
| P90 32K | 136.016 | 116.734 | 1.165x | 1.000007 | 1.000733 |
| P90 64K | 263.829 | 225.356 | 1.171x | 1.000008 | 1.000000 |

Candidate-versus-rounded-reference relative L2 remained
`1.880e-5` to `1.943e-5`, reproducing the legacy rejection. Against the same
FP32 reference, however, candidate error was within `1.000008x` of ordinary
FP16 rounding error, and the maximum-absolute-error ratio was at most
`1.000733x`. Candidate LSE relative L2 was at most `3.662e-8`; all repeated
outputs and LSE values were bit-exact.

The median speedup was `1.165x` and the minimum was `1.154x`. All three cells,
scoped cleanup, postflight, repeated preflight, memory comparison, fatal scan,
and source-identity checks passed. The complete privacy-safe evidence is in
`docs/experiments/evidence/M1_162_CALIBRATED_FP16_QK_20260730`.

## Promotion boundary

A passing M1-162 result means only that this historically high-performance
candidate deserves the next numerical layer. It must still pass:

1. same-activation replay from a baseline TP4 capture on all ranks;
2. short TP4 dispatch, cache transparency, and cold TTFT A/B;
3. teacher-forced distribution characterization;
4. tool, reasoning, multimodal, structured-output and long-context capability;
5. long-context TP4 stability and final performance gates.

GPU0 is currently unhealthy, so this three-GPU screen can run now while the
TP4-dependent stages remain pending.
