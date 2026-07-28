# M1-96 compensated W13 v2 result

Date: 2026-07-28

## Identity

- source revision:
  `d8f32ea7aa29462b3343d6d02302919ea762e2e1`;
- branch: `exp/M1-96-gate-policy-v2-20260728`;
- instance: `ssh-73ca29ba`, physical GPU 0;
- run root: `/tmp/m1-96-w13-d8f32ea-20260728T072002Z`;
- immutable runtime:
  `/root/m1-96-runtime-d8f32ea/site-packages`;
- runtime tree:
  `2ee781205fa722a323adcf1aa6b3d97ac22629a3a0c9aabf45b2d299e40b7f0a`.

The candidate kernel was unchanged from M1-91. Only the predeclared numerical
oracle and evidence schema changed.

## Numerical result

Every fixed output was finite. For both seeds, compensated W13 was bit-exact
with the CPU float64 dot product rounded once to FP16.

| Seed | Implementation | Sequence relative L2 | Maximum step L2 | Maximum absolute error | Mismatches |
|---|---|---:|---:|---:|---:|
| 20260716 | vendor | 1.5089e-5 | 3.8065e-5 | 9.765625e-4 | 46 |
| 20260716 | direct | 8.8906e-6 | 2.3029e-5 | 9.765625e-4 | 32 |
| 20260716 | compensated | 2.5023e-6 | 8.0849e-6 | 2.44140625e-4 | 15 |
| 20260727 | vendor | 1.3302e-5 | 2.8744e-5 | 9.765625e-4 | 48 |
| 20260727 | direct | 9.2091e-6 | 2.3658e-5 | 9.765625e-4 | 39 |
| 20260727 | compensated | 5.2664e-6 | 1.1849e-5 | 4.8828125e-4 | 13 |

The table uses the fixed 12-step stratified sample for each seed. The
compensated candidate is non-inferior to the vendor control on every declared
metric and is materially closer to the high-precision reference.

The complete 500-step candidate-versus-vendor relative L2 remains
`1.4451e-5` and `1.4216e-5`. This is expected diagnostic disagreement between
two FP16 reduction orders, not evidence that the more accurate candidate is
incorrect.

## Performance result

- fixed W13 speedup versus vendor: `6.1931x`;
- routed W13 speedup versus vendor: `5.4727x`;
- required continuation screens: `1.5x` and `1.25x`.

All nine timing trials were retained in `benchmark.json`.

## Lifecycle

Every runner gate returned zero:

- runtime and source identity;
- build, benchmark, qualification, and artifact binding;
- preflight and postflight;
- scoped TERM/reap cleanup;
- complete fatal and timeout scans.

GPU0 had `34,057,748,480 / 34,057,748,480` bytes free both before and after.
There was no fatal, OOM, timeout, Gloo/NCCL error, worker loss, or residual
process. `candidate_qualified`, `evidence_valid`, and `experiment_valid` are
all true.

## Decision

M1-91 is reopened under policy v2 and may proceed to an integration branch
with deterministic next-token and full-model TP4 A/B gates. This result does
not authorize a production default, YAML change, or `main` merge.

The next integration must retain:

- the production dtype, model structure, sampling semantics, and output
  budgets;
- a disabled-by-default internal switch;
- exact baseline/candidate runtime identities;
- full 53-case functional and 11-case Agent matrices;
- paired full-model quality and performance evidence.

Structured evidence is stored in
`docs/experiments/evidence/M1_96_COMPENSATED_W13_V2_BI100_QUALIFIED_20260728`.
