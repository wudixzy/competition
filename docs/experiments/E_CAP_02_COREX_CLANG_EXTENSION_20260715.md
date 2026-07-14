# E-CAP-02: CoreX Clang extension capability

## Scope

The existing E-CAP-01 probe found no vendor FusedMoE symbols and no `ixcc` or
usable `nvcc` in the original PATH. E-CAP-02 repeats the capability audit on
`ssh-a2d0a302.default.gpu.phanthy.com` and tests the installed CoreX Clang
device backend directly.

## Runtime capability

The new instance reproduces the vendor MoE result:

```text
ixformer.functions.vllm_moe_topk_softmax       absent
ixformer.functions.vllm_moe_align_block_size   absent
ixformer.functions.vllm_invoke_fused_moe_kernel absent
torch.ops._moe_C target operators              absent
vllm._custom_ops.supports_moe_ops              true (false-positive wrapper signal)
```

`/usr/local/corex/bin/nvcc` exists but is a 194-byte shell script that only
prints a CUDA 10.2 version banner. It is not a device compiler.

The same image contains CoreX Clang 16 under
`/usr/local/corex-3.2.3/bin/clang++`. Its registered targets include `bi`, and
the device backend exposes `ivcore10`, `ivcore11`, and `ivcore20`. BI-V100
reports CUDA capability 7.0 and CoreX's installed compile command selects
`--cuda-gpu-arch=ivcore10`.

## Compile and runtime gates

`tests/run_corex_clang_smoke.sh` builds and executes two probes:

1. A standalone host/device CUDA program using `cudaMalloc`, kernel launch,
   synchronization through D2H copy, and full result validation.
2. A Torch Python extension linked against the installed ABI-0 Torch 2.1.0
   libraries. It accepts a CUDA tensor, launches on the current Torch stream,
   returns a new tensor, and verifies exact output and input immutability.

Both gates passed on physical GPU1:

```text
COREX_EXTENSION_SMOKE_OK count=256 last=256.0
COREX_TORCH_EXTENSION_SMOKE_OK 256.0
build/status exit code: 0
```

Remote evidence:

```text
/root/competition/bench_runs/20260715_E_CAP_02/build.log
/root/competition/bench_runs/20260715_E_CAP_02/torch_build.log
/root/competition/bench_runs/20260715_E_CAP_02/torch_build.status
```

## Decision

`CUSTOM COREX EXTENSIONS ARE AVAILABLE`. Vendor FusedMoE remains unavailable,
but custom MoE/GDN kernels can be compiled during Docker build with CoreX
Clang. Do not invoke the placeholder `nvcc`; use the documented Clang command,
Torch ABI 0, `ivcore10`, and the image's Torch/CoreX rpaths.
