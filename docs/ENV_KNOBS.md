# BI100 Environment Knobs

| Name | Default | Range | Purpose | Task |
| --- | --- | --- | --- | --- |
| `BI100_ALLOW_PREFIX_GUARD_CAP` | `0` | boolean | Debug-only cap for undersized prefix block tables; default raises because truncation corrupts attention. | T3 |
| `BI100_ATTN_COREX_PAGED_GATHER` | `1` | boolean | Enables the exact fused CoreX K/V gather in the long-context PyTorch decode fallback; set to `0` for native tensor indexing and layout copies. | E-ATTN-04 |
| `BI100_ATTN_COREX_HEAD_RMS_NORM` | `1` | boolean | Enables the exact decode-only CoreX elementwise path for 256-wide full-attention q/k head RMSNorm while retaining PyTorch mean/rsqrt; set to `0` for GemmaRMSNorm. | E-NORM-02 |
| `BI100_DNN_CHUNK` | `4096` | `64..65536` | Caps GatedDeltaNet prefill sub-sequence chunk size to balance memory and launch overhead. | T3 |
| `BI100_EXECUTOR_STARTUP_DEBUG` | `1` | boolean | Adds executor startup logs for TP=4 init/load stalls; enabled in the submission image after the 2026-07-15 evaluator Gloo reset. | T1 |
| `BI100_FORCE_PAGED_ATTN_V2` | `0` | boolean | Explicit opt-in to route paged attention to V2 instead of the stable BI100 V1 default. | T3 |
| `BI100_GDN_ALLOW_NAN_ZERO` | `0` | boolean | Diagnostic-only replacement of non-finite GDN values with zero; invalid for final scoring. | T3 |
| `BI100_GDN_FINITE_CHECK` | `0` | boolean | Enables synchronous per-layer GDN non-finite checks for qualification/debug runs. `BI100_GDN_ALLOW_NAN_ZERO=1` also forces this check on. | E-SYNC-01 |
| `BI100_GDN_COREX_GATED_NORM` | `1` | boolean | Enables the CoreX decode gated-norm output kernel while retaining the PyTorch FP32 inverse reduction; set to `0` for the reference path. | E-GDN-05 |
| `BI100_GDN_COREX_BETA_DECAY` | `1` | boolean | Enables the exact fused CoreX decode beta-sigmoid and recurrent decay-factor kernel for contiguous FP16 rank-local inputs; set to `0` for the PyTorch reference path. | E-GDN-10 |
| `BI100_GDN_COREX_QK_MAP` | `1` | boolean | Normalizes four FP16 q/k heads before expansion, then fuses exact 4-to-8 head mapping, FP32 conversion, and query scaling; set to `0` for the PyTorch reference path. | E-GDN-12 |
| `BI100_GDN_COREX_PACKED_DECODE` | code fallback `0`; submission `1` | boolean | Enables the TP4-qualified single-token packed q/k/v, beta/decay, recurrent update, and output kernel for the exact local `(1,4,8,128)` GDN shape; unsupported inputs use the existing path. | E-GDN-14 |
| `BI100_KV_EVICTION_POLICY` | `lru` | `lru`, `frequency` | Private M1-41 experiment selector for SHA-256 content-frequency-aware KV eviction. Invalid values fail at allocator construction; it must remain absent from submission YAML until the complete 881-request trace qualifies. | M1-41 |
| `BI100_GDN_CACHE_POLICY` | `fine32` | `off`, `fine32`, `admission64` | Selects the scheduler-owned recurrent prefix-state policy. `fine32` keeps 32 chunk checkpoints; `admission64` keeps up to 64 branch/final states; `off` disables recurrent-state reuse while leaving raw KV prefix caching enabled. | M1-31 |
| `BI100_GDN_RESTORE_MODE` | `direct` | `direct`, `chunk64`, `aligned` | `direct` restores any complete KV block and remains the submission default. Experimental `chunk64` restricts states to the native 64-token DeltaNet recurrence boundary; `aligned` restricts states to `max_num_batched_tokens`. Neither experimental mode is enabled in submission YAML before runtime qualification. | M1-31/M1-33 |
| `BI100_HYBRID_KV_ACCOUNTING` | `legacy40` | `legacy40`, `full_attention` | Private M1-49 same-image selector. `full_attention` allocates KV only for Qwen's ten full-attention layers and aligns startup profiling to that count; it remains absent from submission YAML until TP4 long-context and 881 gates qualify. | M1-49 |
| `BI100_MOE_COREX_EXACT_REDUCE` | `1` | boolean | Enables the exact CoreX T=1 MoE weighted reduction for FP16 top-8 outputs; set to `0` for the PyTorch reference path. | E-MOE-10 |
| `BI100_MOE_COREX_WEIGHT_GATHER` | `1` | boolean | Enables the exact 128-bit segmented CoreX gather of selected FP16 top-8 routed-expert weights in the T=1 decode path; set to `0` for native advanced indexing. | E-MOE-12/13 |
| `BI100_MOE_COREX_DIRECT_ROUTED` | code fallback `0`; submission `1` | boolean | Enables the TP4-qualified staged direct-addressing W13 and fused W2/routed-reduction path only for the exact FP16 `(M=1,E=256,K=8,H=2048,I=128)` decode shape. Unsupported shapes fail closed to the reference path. | E-MOE-20 |
| `BI100_MOE_FUSED_ACTIVATION` | `1` | boolean | Reuses vLLM's bit-exact `SiluAndMul` for the T=1 routed-expert activation; set to `0` for the native `F.silu(gate) * up` path. | E-MOE-11 |
| `BI100_PAGED_ATTN_DIAGNOSTICS` | `0` | boolean | Enables physical slot/block-ID checks, device synchronization after `reshape_and_cache`, and sparse 8,192-token decode snapshots. Diagnostic only; invalid for performance runs. | E-CTX-01 |
| `BI100_CACHE_TRACE` | `0` | boolean | Emits privacy-redacted version-4 allocator lifecycle records with chained SHA-256 block hashes for offline KV/GDN policy simulation. Diagnostic only; never enable for scored latency runs. | M1-31 |
| `BI100_CPU_KV_OFFLOAD` | `0` | exactly `0` or `1` | Enables the private M1-45 scheduler-owned, content-addressed inclusive CPU KV tier. It reuses the existing CPU swap allocation, disables request-level swap, and must remain absent from submission YAML until all TP4 long-context and 881-request performance gates qualify. | M1-45 |
| `BI100_PREFIX_BLOCKS_PER_TILE` | `32` | `1..1024` | Prefix attention K/V block tile count for the PyTorch online-softmax fallback. | T3 |
| `BI100_PROFILE` | `0` | boolean | Enables lightweight CUDA-synchronized timers for BI100 hotspot profiling. | T4 |
| `BI100_GDN_COREX_CAUSAL_CONV` | `1` | boolean | Enables the fused CoreX decode causal-convolution/state-update kernel when the extension is installed; set to `0` for the PyTorch reference path. | E-GDN-03 |
| `BI100_PROFILE_INCLUDE_STARTUP` | `0` | boolean | Includes vLLM synthetic startup `profile_run()` in BI100 timers; default skips it so profiling focuses on real requests and avoids perturbing startup dummy runs. | T5 |
| `BI100_PROFILE_MODE` | `sync` | `sync`, `event` | Diagnostic timer backend. M1-48 uses `event`; this variable is forbidden in submission YAML. | M1-48 |
| `BI100_PROFILE_FILTER` | empty | comma-separated bounded globs | Restricts privacy-safe M1-48 timing regions. It is diagnostic-only and forbidden in submission YAML. | M1-48 |
| `BI100_PYTORCH_DECODE_THRESHOLD` | `32768` | `1..262144` | Routes long-context decode to the pure PyTorch paged attention fallback. | T3 |
| `BI100_UNSET_CUDA_VISIBLE_DEVICES` | `1` | boolean shell flag | Lets the contest container expose all four GPUs by default while allowing debug runs to preserve a caller-specified visibility mask. | T1 |
| `ENABLE_CUSTOM_IPC` | `1` | boolean | Enables IxFormer CUDA-IPC all-reduce for same-node TP. Set `0` to restore the IxFormer NCCL path. E-COLL-01 measured +44.6% decode TPS P10 with 328 fewer GPU KV blocks; reduction order is quality-equivalent but not bit-exact. | E-COLL-01 |

`NUM_GPU_BLOCKS_OVERRIDE` was used in an earlier diagnostic run but is no longer exposed by `launch_service`. Skipping vLLM synthetic profiling violates the fixed competition launch contract and is invalid for official comparison.
