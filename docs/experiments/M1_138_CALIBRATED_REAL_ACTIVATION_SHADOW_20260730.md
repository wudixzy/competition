# M1-138 calibrated real-activation shadow

Date: 2026-07-30

Status: implementation and local validation in progress on the private
experiment branch. TP4 evidence is pending. No default, YAML, `main`, or
repository visibility change is authorized.

## Motivation

M1-136 compared the fused FP16 output with the existing PyTorch FP16 output on
the same real activations. All eight collected 65K observations were finite
and the maximum relative L2 error was `7.1011427252343464e-6`. Two observations
failed only the fixed `0.001` max-absolute-error limit, with an error of exactly
`0.001953125`. The legacy fail-fast then prevented collection of the 131K
bucket.

A fixed absolute threshold is scale dependent. For FP16, the representable
spacing grows with magnitude, so `0.001953125` can be an ordinary rounding
step rather than evidence of an unstable attention calculation. This does not
make numerical validation optional; it means the hard error bound must be
calibrated against the unavoidable output rounding error.

## Frozen numeric contract

The pre-observation contract is
`quality/fused_prefill_numeric_adjudication.v1.json`, SHA-256
`131e2ed8e0b34cc28a45486b9a9096d66c556759677b8bbd31024a33933d86b1`.

For the exact same real Q, K, V, paged-cache and block-table activations, the
shadow computes:

1. the candidate fused FP16 output;
2. the PyTorch online-softmax output before its final cast, accumulated in
   FP32;
3. the normal rounded baseline produced by casting that FP32 reference to
   FP16.

Every observation must be finite and satisfy all of:

- candidate versus rounded-reference relative L2 `<= 1e-5`;
- candidate versus FP32 relative L2 no more than twice the normal
  FP16-rounding relative L2, plus the frozen denominator floor;
- candidate versus FP32 max absolute error no more than twice the normal
  FP16-rounding max absolute error, plus the same floor.

The old fixed max-absolute threshold remains reported as a diagnostic but no
longer decides this calibrated gate. Semantic or task scores cannot waive a
failure. Conversely, passing this gate decides only the covered operator
surface and cannot establish task capability or production readiness.

## Sampling and failure behavior

The fixed TP4 matrix requires ranks 0 through 3, two observations in each
disjoint context bucket beginning at 49,152 and 114,688 tokens, FP16, head
dimension 256, block size 16, GQA 4:1, and query segments of 17 through 8192
tokens.

Finite threshold failures are recorded so both context buckets can be
collected. They still make qualification fail. Candidate non-finite output,
reference failure, an invalid native result, or malformed evidence remains
fail-fast. The qualifier independently recomputes every ratio, report status,
observation count and scalar maximum from the records; malformed evidence
takes precedence over a numeric-failure classification.

The rank is obtained from an initialized distributed process group before
falling back to environment or current-device identity. The shared process
helper already waits for the service parent. M1-138 additionally gives any
zombies already adopted by PID 1 a bounded 20-second interval to disappear
before the recorded-session audit. A surviving live process or zombie still
fails the run; the runner never signals an unverified process group.

## TP4 command

After installing an immutable runtime overlay from the exact clean revision:

```bash
BI100_RUNTIME_SITE_PACKAGES=/path/to/runtime/site-packages \
BI100_RUNTIME_INSTALL_REPORT=/path/to/runtime/install.json \
scripts/run_m1_138_fused_prefill_calibrated_shadow.sh \
  ssh-73ca29ba /tmp/m1-138-calibrated-shadow-SOURCE
```

The runner retains the M1-136 fixed 65K/131K requests, runtime identity,
dispatch proof, scoped cleanup, postflight, four-card preflight comparison,
and fatal/timeout scans. A passing result permits the candidate to proceed to
the powered task-capability gates. It does not authorize performance claims,
formal YAML changes, a `main` merge, or production promotion.
