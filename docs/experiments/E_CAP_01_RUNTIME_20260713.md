# E-CAP-01 Runtime Capability - 2026-07-13

## Environment

- PyTorch `2.1.0+corex.3.2.3`
- vLLM `0.6.3+corex.3.2.3`
- transformers `4.55.3`
- ixformer `0.4.0+corex.3.2.3`

Importing ixformer requires the CoreX/OpenMPI `LD_LIBRARY_PATH`; without it the
loader fails on `libmpi.so.40`. The Dockerfile already provides the required
paths.

## MoE capability

The vendor vLLM source contains wrappers for:

- `vllm_moe_topk_softmax`;
- `vllm_moe_align_block_size`;
- `vllm_invoke_fused_moe_kernel`.

However, all three target attributes are absent from `ixformer.functions` in
the installed package. `torch.ops._moe_C` also lacks `topk_softmax`, align,
fused MoE, and marlin MoE operators. A string scan of all three ixformer shared
objects found no matching MoE symbols.

`vllm._custom_ops.supports_moe_ops == true` is therefore not a sufficient
capability signal on this image: calling its wrappers would reach missing
ixformer attributes.

Decision: `E-MOE-01` fails at the level-1 symbol gate. Do not execute GPU MoE
probes through these wrappers and do not enable vendor FusedMoE. Continue with
the documented custom/combination fallback design unless the organizer provides
a different ixformer build.

## Attention capability

- ixformer `vllm_single_query_cached_kv_attention`: present;
- vLLM `paged_attention_v1`: implemented through that ixformer function;
- vLLM `paged_attention_v2`: explicit `NotImplementedError`.

The existing long-context fallback remains required. Forcing v2 is not a valid
optimization on this runtime.

## Build capability

- GCC/G++: present;
- CMake: present;
- Ninja: absent;
- ixcc: absent from PATH;
- nvcc: absent from PATH.

Custom extension work must first identify the supported CoreX compiler entry
point and headers. Do not assume a CUDA/NVCC build flow.
