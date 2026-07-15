# Fused MoE architecture notes for BI100 decode

## References

- vLLM fused MoE implementation:
  <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py>
- NVIDIA CUTLASS grouped GEMM example:
  <https://github.com/NVIDIA/cutlass/blob/main/examples/24_gemm_grouped/gemm_grouped.cu>
- ROCm AITER operator repository:
  <https://github.com/ROCm/aiter>

## Transferable design

Current vLLM maps each output tile to an expert id and adds the expert stride
inside the MoE kernel. Its small-batch `naive_block_assignment` bypasses token
sorting when the assignment metadata would cost more than it saves. Routed
weights are consumed by the second expert GEMM kernel instead of a separate
Python reduction. These are the relevant principles for the competition's
single-token decode path:

- expert selection belongs in weight address calculation;
- `M=1` should not pay general token-sort or grouped-GEMM setup costs;
- routed weighting should share the W2 output boundary;
- W13 and W2 remain separate stages because activation is a real dependency.

E-MOE-20 applies those principles to the fixed rank-local shape without first
copying selected weights. It also keeps a staged activation variant so the
value of direct addressing and W2 reduction can be measured independently of
activation approximation.

## Non-transferable implementation

The referenced production kernels depend on Triton, CUTLASS, CUDA tensor-core
contracts, or ROCm CK/FlyDSL backends. None is a supported BI100/CoreX backend
in the evaluator image. Porting their source or tuning tables would not create
a valid implementation. The reusable part is the scheduling and data-flow
model; the executable kernel must use CoreX clang and ivcore10 primitives that
have already passed local probes.

CUTLASS grouped GEMM is also a poor direct match for this exact boundary. Its
general grouped scheduler and host/device problem metadata target many GEMM
shapes, while this path has one token, eight fixed-size experts, and two GEMV-
like projections. The earlier pointer-batched cuBLAS experiment measured that
general dispatch overhead and was slower than the current gather path.

## Decision rule

Do not tune block sizes before the direct two-stage algorithm passes the
boundary gates. If it passes, the next work is endpoint qualification and
production dispatch guards. If it fails, return to the current production
path and select a different full-layer boundary from a valid TP4 profile.
