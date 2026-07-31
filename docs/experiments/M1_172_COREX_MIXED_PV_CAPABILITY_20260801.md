# M1-172 CoreX mixed-PV capability result

## Status

The bounded mixed-value PV capability probe is closed. The candidate kept V
in FP16, probabilities and online state in FP32, and requested FP32 output and
accumulation from `cublasGemmStridedBatchedEx`. It did not change model dtype,
request semantics, context capacity, or any production default.

## Result

The first compile attempt used an incorrect Torch include path and is excluded
as harness error. The only permitted environment correction used the verified
CoreX Torch root at
`/usr/local/corex-3.2.3/lib64/python3/dist-packages/torch`; compilation then
succeeded and produced candidate SHA-256
`6ee08812eab724a6acb7098de75981ec96335af332fd76ce95e1917f20b0ea20`.

The fixed 16K calibrated cell reached the mixed PV GEMM and failed with cuBLAS
status 15. CoreX 3.2.3 declares status 15 as
`CUBLAS_STATUS_NOT_SUPPORTED`. No finite output, numeric comparison,
repeatability result, or speedup exists, so the candidate cannot enter the
three-cell screen or any service test.

The experiment used physical GPU 1 on `ssh-73ca29ba`. Scoped cleanup removed
the temporary build tree, no experiment process remained, and GPU1 postflight
passed. No TP4 process was started.

## Decision

The FP16-V times FP32-probability GemmEx contract is closed under the frozen
stop rule. There will be no tile, chunk, algorithm, or YAML scan for this
contract. A distinct FP16-probability PV design would be a separate numerical
and distribution-risk experiment; this result does not authorize it.

The local M1-164 drafts remain untracked and are not included in this result
commit. Privacy-safe evidence is in
`docs/experiments/evidence/M1_172_COREX_MIXED_PV_CAPABILITY_20260801`.
