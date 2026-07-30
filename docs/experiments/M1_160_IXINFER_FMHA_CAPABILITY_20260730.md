# M1-160 ixinfer FMHA capability matrix

Date: 2026-07-30

Status: the installed CoreX 3.2.3 `cuinferFMHAForwardEx` entry point rejected
all four bounded capability cells with `CUINFER_STATUS_BAD_PARAM`. This route
is closed for the current environment. It is not authorized for runtime
overlay, TP4 service, quality, default, YAML, `main`, or official-score
testing.

## Motivation

The M1-156 profile attributes 64%-70% of the fixed 16K, 32K, and 64K
prefill operator time to QK and PV. M1-159 showed that changing only the
normalization schedule improves that operator by about 1.1%, so a fused
attention implementation is the remaining structural operator direction.

CoreX ships `libcuinfer.so` and a public `ixinfer.h` declaration for
`cuinferFMHAForwardEx`. The library contains FP16 FMHA kernel symbols for
head dimension 256. M1-160 tests whether that supported-looking vendor path
can replace a custom implementation without changing the production
runtime.

## Bounded matrix

The isolated extension uses contiguous FP16 tensors, exact tensor
descriptors and strides, the current CUDA stream, and the documented
configuration fields. No M1-160 file is installed by `patch_ops.sh`.

| Case | Layout | D | Heads | Causal | Result |
|---|---|---:|---:|---|---|
| MHA control | BSHD | 128 | 4:4 | no | `BAD_PARAM` |
| MHA production D | BSHD | 256 | 4:4 | no | `BAD_PARAM` |
| Layout control | BHSD | 128 | 4:4 | no | `BAD_PARAM` |
| Production GQA | BSHD | 256 | 4:1 | yes | `BAD_PARAM` |

Every cell returned normally from its isolated child process with nonzero
status; none timed out or wrote a numerical result. The privacy-safe fatal
scan found zero CUDA fatal errors, Gloo resets, OOMs, segmentation faults,
timeouts, or worker losses. Scoped postflight found no server, worker, or
GPU process residue. A subsequent parallel matmul preflight passed on GPUs
1, 2, and 3.

## Identity

- source revision:
  `9a2a87b88f8d450d587b82c073a055fb5742eccc`;
- instance: `ssh-73ca29ba`;
- CoreX: `3.2.3`;
- `libcuinfer.so` SHA-256:
  `465678e6811a0fc3eaf8923c62852e59a19a69f500b9af16e6529c39c36b92a2`;
- isolated extension SHA-256:
  `3c83ae9c0bb35096bd41c9bc9be4481710ce234335bcdc3b0ff80589d4b39b5a`;
- probe source SHA-256:
  `c0bb78ee90d75037a2269e6a8ba525dca51bc59b39ed0379831789455cacc6da`.

## Decision

The failure is at vendor dispatch rather than the PyTorch reference path.
Both layouts, both advertised head sizes, noncausal MHA, and causal GQA fail
under the header contract. Continuing to guess undocumented configuration
values would violate the two-implementation stop rule and would not be a
controlled TTFT experiment.

The next TTFT-P90 work returns to the two measured levers:

1. reduce 16K-64K residual prefill through correct admission64 branch-state
   reuse; and
2. implement one bounded custom FP32-preserving QK/online-softmax/PV data
   flow that does not materialize the complete score tensor.

Formal defaults, `computility-run.yaml`, and `main` remain unchanged.
Privacy-safe evidence is under
`docs/experiments/evidence/M1_160_IXINFER_FMHA_CAPABILITY_20260730`.
