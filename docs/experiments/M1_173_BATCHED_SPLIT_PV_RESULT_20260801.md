# M1-173 batched split-PV result

## Status

The fixed three-GPU screen closed this route. All numerical and lifecycle
checks passed, but the median operator speedup was only `1.00437x`, below the
predeclared `1.08x` gate. M1-173 does not authorize real-activation replay,
TP4 service testing, a production overlay, YAML, or `main` changes.

## Harness correction

The first invocation stopped before tensor allocation. The M1-162 cell loader
assumed that every baseline exported `PyInit_corex_fused_paged_prefill`, while
the selected M1-162 baseline correctly exported
`PyInit_corex_fused_paged_prefill_fp16_qk`. Commit
`b3d20731d31a8b67d1ec44bc1725d34fd2752508` made the baseline module name an
explicit, identity-checked runner input. Historical M1-157/M1-162 defaults
remain `corex_fused_paged_prefill`; only M1-173 selects the FP16-QK name.

Two orphaned serial GPU health probes from earlier diagnostics were also found
before the valid run. Their complete process groups received SIGTERM and
exited during the required 60-second grace period; SIGKILL was not used. They
were not service or user processes. The valid run started only after clean
postflight and independent GPU1/2/3 preflight.

## Identity

- source: `b3d20731d31a8b67d1ec44bc1725d34fd2752508`;
- instance: `ssh-73ca29ba`;
- physical GPUs: 1, 2, and 3;
- baseline module: `corex_fused_paged_prefill_fp16_qk`;
- baseline SHA-256:
  `36e043f138aa87c635178e4aa6a30af710b87c3f3d7c2a3f1838fc0e365bd368`;
- candidate module: `corex_fused_paged_prefill_batched_split_pv`;
- candidate SHA-256:
  `4319ade649d9501df8ff605c9099f5b82cb2bc38c07f7b661ab5ad95b49cbe2f`;
- numeric contract SHA-256:
  `1be3ccf34cef906fdc8345c1754960bb4485259f51c3963ab9ca15fd3a4bdb05`.

## Result

| Cell | Baseline CUDA ms | Candidate CUDA ms | Speedup |
| --- | ---: | ---: | ---: |
| 16K | 62.731 | 62.788 | 0.9991x |
| 32K | 117.076 | 116.508 | 1.0049x |
| 64K | 225.683 | 224.702 | 1.0044x |

All candidate outputs and LSE values were finite. Candidate versus baseline
output and LSE were bit-exact in all three cells, and repeated candidate output
and LSE were bit-exact. Relative-L2 error versus the same FP32 reference was at
most `1.000008x` ordinary FP16 rounding error; maximum absolute error was at
most `1.000733x` rounding error. Every cell passed its numerical and `0.98x`
no-regression gate.

The aggregate failed only because the median speedup was `1.00437x` instead of
at least `1.08x`. The minimum was `0.99910x`. Combining split and head into one
FP32 SGEMM batch therefore proves that PV API/launch count is not a material
remaining bottleneck at these shapes. The extra value-tile writes roughly
offset any launch reduction.

## Lifecycle and decision

All three cell processes returned zero. Scoped cleanup reaped every child;
postflight before and after, repeated preflight, and the preflight comparison
passed. Fatal, OOM, CoreX, Gloo reset, worker-loss, timeout, and segmentation
fault scans were empty. `failure.json` records the expected aggregate-gate
failure at `paired_operator_cells`, not an execution failure.

Do not continue scanning PV batch layouts or thresholds. Retain M1-162 FP16-QK
as the mature operator candidate pending real TP4 activation capture. Further
three-card work should return to the measured QK or normalization data flow,
and must preserve FP32 probability and accumulation semantics.

Privacy-safe structured evidence is under
`docs/experiments/evidence/M1_173_BATCHED_SPLIT_PV_20260801`.
