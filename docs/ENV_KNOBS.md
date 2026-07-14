# BI100 Environment Knobs

| Name | Default | Range | Purpose | Task |
| --- | --- | --- | --- | --- |
| `BI100_ALLOW_PREFIX_GUARD_CAP` | `0` | boolean | Debug-only cap for undersized prefix block tables; default raises because truncation corrupts attention. | T3 |
| `BI100_DNN_CHUNK` | `4096` | `64..65536` | Caps GatedDeltaNet prefill sub-sequence chunk size to balance memory and launch overhead. | T3 |
| `BI100_EXECUTOR_STARTUP_DEBUG` | `0` | boolean | Adds executor startup logs for TP=4 init/load stalls. | T1 |
| `BI100_FORCE_PAGED_ATTN_V2` | `0` | boolean | Explicit opt-in to route paged attention to V2 instead of the stable BI100 V1 default. | T3 |
| `BI100_GDN_ALLOW_NAN_ZERO` | `0` | boolean | Diagnostic-only replacement of non-finite GDN values with zero; invalid for final scoring. | T3 |
| `BI100_GDN_FINITE_CHECK` | `0` | boolean | Enables synchronous per-layer GDN non-finite checks for qualification/debug runs. `BI100_GDN_ALLOW_NAN_ZERO=1` also forces this check on. | E-SYNC-01 |
| `BI100_GDN_COREX_RECURRENT` | `1` | boolean | Uses the build-time CoreX fused GDN decode recurrent kernel. Set to `0` only for same-image reference A/B and rollback. | E-GDN-07 |
| `BI100_PAGED_ATTN_DIAGNOSTICS` | `0` | boolean | Enables physical slot/block-ID checks, device synchronization after `reshape_and_cache`, and sparse 8,192-token decode snapshots. Diagnostic only; invalid for performance runs. | E-CTX-01 |
| `BI100_PREFIX_BLOCKS_PER_TILE` | `32` | `1..1024` | Prefix attention K/V block tile count for the PyTorch online-softmax fallback. | T3 |
| `BI100_PROFILE` | `0` | boolean | Enables lightweight CUDA-synchronized timers for BI100 hotspot profiling. | T4 |
| `BI100_PROFILE_INCLUDE_STARTUP` | `0` | boolean | Includes vLLM synthetic startup `profile_run()` in BI100 timers; default skips it so profiling focuses on real requests and avoids perturbing startup dummy runs. | T5 |
| `BI100_PYTORCH_DECODE_THRESHOLD` | `32768` | `1..262144` | Routes long-context decode to the pure PyTorch paged attention fallback. | T3 |
| `BI100_UNSET_CUDA_VISIBLE_DEVICES` | `1` | boolean shell flag | Lets the contest container expose all four GPUs by default while allowing debug runs to preserve a caller-specified visibility mask. | T1 |
| `ENABLE_CUSTOM_IPC` | `1` | boolean | Enables IxFormer CUDA-IPC all-reduce for same-node TP. Set `0` to restore the IxFormer NCCL path. E-COLL-01 measured +44.6% decode TPS P10 with 328 fewer GPU KV blocks; reduction order is quality-equivalent but not bit-exact. | E-COLL-01 |

`NUM_GPU_BLOCKS_OVERRIDE` was used in an earlier diagnostic run but is no longer exposed by `launch_service`. Skipping vLLM synthetic profiling violates the fixed competition launch contract and is invalid for official comparison.
