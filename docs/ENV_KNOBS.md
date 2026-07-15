# BI100 Environment Knobs

| Name | Default | Range | Purpose | Task |
| --- | --- | --- | --- | --- |
| `BI100_ALLOW_PREFIX_GUARD_CAP` | `0` | boolean | Debug-only cap for undersized prefix block tables; default raises because truncation corrupts attention. | T3 |
| `BI100_ATTN_COREX_PAGED_GATHER` | `1` | boolean | Enables the exact fused CoreX K/V gather in the long-context PyTorch decode fallback; set to `0` for native tensor indexing and layout copies. | E-ATTN-04 |
| `BI100_DNN_CHUNK` | `4096` | `64..65536` | Caps GatedDeltaNet prefill sub-sequence chunk size to balance memory and launch overhead. | T3 |
| `BI100_EXECUTOR_STARTUP_DEBUG` | `0` | boolean | Adds executor startup logs for TP=4 init/load stalls. | T1 |
| `BI100_FORCE_PAGED_ATTN_V2` | `0` | boolean | Explicit opt-in to route paged attention to V2 instead of the stable BI100 V1 default. | T3 |
| `BI100_GDN_ALLOW_NAN_ZERO` | `0` | boolean | Diagnostic-only replacement of non-finite GDN values with zero; invalid for final scoring. | T3 |
| `BI100_GDN_FINITE_CHECK` | `0` | boolean | Enables synchronous per-layer GDN non-finite checks for qualification/debug runs. `BI100_GDN_ALLOW_NAN_ZERO=1` also forces this check on. | E-SYNC-01 |
| `BI100_GDN_COREX_GATED_NORM` | `1` | boolean | Enables the CoreX decode gated-norm output kernel while retaining the PyTorch FP32 inverse reduction; set to `0` for the reference path. | E-GDN-05 |
| `BI100_MOE_COREX_EXACT_REDUCE` | `1` | boolean | Enables the exact CoreX T=1 MoE weighted reduction for FP16 top-8 outputs; set to `0` for the PyTorch reference path. | E-MOE-10 |
| `BI100_MOE_COREX_WEIGHT_GATHER` | `1` | boolean | Enables the exact 128-bit segmented CoreX gather of selected FP16 top-8 routed-expert weights in the T=1 decode path; set to `0` for native advanced indexing. | E-MOE-12/13 |
| `BI100_MOE_FUSED_ACTIVATION` | `1` | boolean | Reuses vLLM's bit-exact `SiluAndMul` for the T=1 routed-expert activation; set to `0` for the native `F.silu(gate) * up` path. | E-MOE-11 |
| `BI100_PAGED_ATTN_DIAGNOSTICS` | `0` | boolean | Enables physical slot/block-ID checks, device synchronization after `reshape_and_cache`, and sparse 8,192-token decode snapshots. Diagnostic only; invalid for performance runs. | E-CTX-01 |
| `BI100_PREFIX_BLOCKS_PER_TILE` | `32` | `1..1024` | Prefix attention K/V block tile count for the PyTorch online-softmax fallback. | T3 |
| `BI100_PROFILE` | `0` | boolean | Enables lightweight CUDA-synchronized timers for BI100 hotspot profiling. | T4 |
| `BI100_GDN_COREX_CAUSAL_CONV` | `1` | boolean | Enables the fused CoreX decode causal-convolution/state-update kernel when the extension is installed; set to `0` for the PyTorch reference path. | E-GDN-03 |
| `BI100_PROFILE_INCLUDE_STARTUP` | `0` | boolean | Includes vLLM synthetic startup `profile_run()` in BI100 timers; default skips it so profiling focuses on real requests and avoids perturbing startup dummy runs. | T5 |
| `BI100_PYTORCH_DECODE_THRESHOLD` | `32768` | `1..262144` | Routes long-context decode to the pure PyTorch paged attention fallback. | T3 |
| `BI100_UNSET_CUDA_VISIBLE_DEVICES` | `1` | boolean shell flag | Lets the contest container expose all four GPUs by default while allowing debug runs to preserve a caller-specified visibility mask. | T1 |
| `ENABLE_CUSTOM_IPC` | `1` | boolean | Enables IxFormer CUDA-IPC all-reduce for same-node TP. Set `0` to restore the IxFormer NCCL path. E-COLL-01 measured +44.6% decode TPS P10 with 328 fewer GPU KV blocks; reduction order is quality-equivalent but not bit-exact. | E-COLL-01 |

`NUM_GPU_BLOCKS_OVERRIDE` was used in an earlier diagnostic run but is no longer exposed by `launch_service`. Skipping vLLM synthetic profiling violates the fixed competition launch contract and is invalid for official comparison.
