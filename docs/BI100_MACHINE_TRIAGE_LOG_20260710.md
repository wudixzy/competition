# BI100 Machine Triage Log - 2026-07-10

This log records the BI100 instance screening done after the T5 optimization
work was blocked by CUDA timeouts. It is intended for platform/runtime triage.

## Summary

No tested replacement instance has passed the minimum hardware gate:

1. SSH/TLS access works.
2. CoreX runtime is visible.
3. `torch==2.1.0` imports and sees 4 CUDA devices.
4. Each GPU can run a minimal single-card BF16 matmul.
5. Only if all 4 single-card checks pass, run TP=4 NCCL all-reduce.

All failures below happened before vLLM service startup and before any project
model code, patch code, or benchmark code was required. The repeated symptom is
that a minimal `torch` CUDA tensor/matmul operation hangs until timeout on one
or more GPUs.

This does not prove physical GPU damage. It is more accurately described as a
platform/runtime/device-state failure at the CUDA/CoreX execution layer. It is
unlikely to be caused by this repository's vLLM changes because the failing
probe is a standalone `torch` script.

## Probe Used

The strict quick check used this shape:

```bash
BI100_SSH=root@<host> \
BI100_SSH_KEY=/home/coolboy/projects/enginex-vllm-bi100-qwen36/Iluvatar-BI-V100-50c-200G-4GPU-Qwen3.6-35B-A3B.pem \
BI100_SSH_TLS_PORT=32222 \
REMOTE_ROOT=/root \
timeout 220 scripts/remote-run.sh '<remote script>'
```

`scripts/remote-run.sh` injects the CoreX environment:

```bash
PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages
LD_LIBRARY_PATH=/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib
PATH=/usr/local/corex/bin:/usr/local/corex-3.2.3/bin:/usr/local/openmpi/bin:...
```

Per-GPU CUDA check:

```python
import torch
torch.cuda.init()
a = torch.ones((512, 512), device="cuda", dtype=torch.bfloat16)
b = a @ a
torch.cuda.synchronize()
free, total = torch.cuda.mem_get_info()
print("checksum=%s free=%s total=%s" % (float(b.sum().item()), free, total))
```

Each GPU was run as:

```bash
CUDA_VISIBLE_DEVICES=<gpu> timeout 15 python3 /tmp/quick_gpu_check.py
```

NCCL was intentionally skipped unless all 4 single-card CUDA checks passed.

## Important Interpretation Notes

- `ixsmi` visibility alone is not enough. Several machines showed all four
  BI100 cards in `ixsmi` while CUDA kernel/tensor execution still timed out.
- A `pynvml` deprecation warning appeared frequently; it is not the failure.
- Some early subagent runs reported DNS or `Connection closed by UNKNOWN port
  65535`. Later non-sandbox checks showed the involved hosts could resolve,
  reach the TLS tunnel, and SSH login successfully. Those earlier connection
  errors were treated as network-path/subagent-environment instability, not as
  final machine health evidence.
- One strict-check command printed `NameError: name 'TORCH' is not defined`
  from a diagnostic `python3 -c` label quoting issue. That only affected the
  version-print line. The per-GPU CUDA checks ran independently and remain valid.

## Host Results

| Host | SSH/CoreX/Torch/Model Dir | Single-GPU CUDA Result | NCCL | Decision |
| --- | --- | --- | --- | --- |
| `ssh-7f994958.default.gpu.phanthy.com` | Earlier baseline initially worked; after BI100_PROFILE failure, SSH and `ixsmi` still visible | GPU0 timeout, GPU1 timeout, GPU2 PASS, GPU3 PASS | Later NCCL timed out after unhealthy state | Not ready |
| `ssh-0fa8ec7c.default.gpu.phanthy.com` | SSH OK, `ixsmi` OK, model dir OK | GPU0/1/2/3 timeout | Skipped | Not ready |
| `ssh-ea17e3be.default.gpu.phanthy.com` | SSH OK, `torch 2.1.0`, `device_count=4`, `ixsmi` OK after corrected env, model dir OK | GPU0 PASS, GPU1 timeout, GPU2 PASS, GPU3 PASS | Skipped | Not ready |
| `ssh-a37d238f.default.gpu.phanthy.com` | SSH OK, `torch 2.1.0`, `device_count=4`, `ixsmi` OK after corrected env, model dir OK | GPU0/1/2/3 timeout | Skipped | Not ready |
| `ssh-ded5dcef.default.gpu.phanthy.com` | SSH OK, `torch 2.1.0`, `device_count=4`, `ixsmi` OK, model dir OK | GPU0 timeout, GPU1 timeout, GPU2 PASS, GPU3 PASS | Skipped | Not ready |
| `ssh-9ff4e457.default.gpu.phanthy.com` | SSH/CoreX/Torch/model dir OK in quick check | GPU0 hung after `init_ok`; GPU1/2/3 not completed inside quick window | Skipped | Not ready |
| `ssh-3a52c74e.default.gpu.phanthy.com` | SSH OK, `torch 2.1.0`, `device_count=4`, `ixsmi` OK, model dir OK | GPU0 PASS, GPU1 timeout, GPU2 PASS, GPU3 PASS | Skipped | Not ready |
| `ssh-4f83aab0.default.gpu.phanthy.com` | SSH OK, `torch 2.1.0`, `device_count=4`, `ixsmi` OK, model dir OK | GPU0/1/2/3 timeout | Skipped | Not ready |
| `ssh-97df07c5.default.gpu.phanthy.com` | Non-sandbox DNS/TLS/SSH OK, `ixsmi` OK, model dir OK | GPU0/1/2/3 timeout | Skipped | Not ready |
| `ssh-29feca15.default.gpu.phanthy.com` | Non-sandbox DNS/TLS/SSH OK, `ixsmi` OK, model dir OK | GPU0 PASS, GPU1 timeout, GPU2 PASS, GPU3 PASS | Skipped | Not ready |
| `ssh-0ef3419e.default.gpu.phanthy.com` | Non-sandbox DNS/TLS/SSH OK, `ixsmi` OK, model dir OK | GPU0/1/2/3 timeout | Skipped | Not ready |

## Raw Evidence Snippets

### `ssh-97df07c5`

Non-sandbox TLS/SSH checks:

```text
Connecting to 82.157.12.233
depth=0 CN=TRAEFIK DEFAULT CERT
SSH-2.0-OpenSSH_8.2p1 Ubuntu-4ubuntu0.13

cc-97df07c5-3f09-43fd-ab1b-521b9852bdbc-0
2026-07-10T07:54:07+00:00
```

Strict CUDA quick check:

```text
HOST=cc-97df07c5-3f09-43fd-ab1b-521b9852bdbc-0
/usr/local/bin/python3
/usr/local/corex/bin/ixsmi
drwxr-xr-x ... /root/public-storage/models/Qwen/Qwen3.6-35B-A3B
== GPU0 ==
GPU0_RC=124
== GPU1 ==
GPU1_RC=124
== GPU2 ==
GPU2_RC=124
== GPU3 ==
GPU3_RC=124
QUICK_MACHINE_NOT_READY
```

### `ssh-29feca15`

Non-sandbox SSH:

```text
cc-29feca15-4c71-46dd-8798-7c9be8bf30bc-0
2026-07-10T07:54:02+00:00
```

Strict CUDA quick check:

```text
HOST=cc-29feca15-4c71-46dd-8798-7c9be8bf30bc-0
/usr/local/bin/python3
/usr/local/corex/bin/ixsmi
drwxr-xr-x ... /root/public-storage/models/Qwen/Qwen3.6-35B-A3B
== GPU0 ==
checksum=134217728.0 free=34055647232 total=34057748480
GPU0_RC=0
== GPU1 ==
GPU1_RC=124
== GPU2 ==
checksum=134217728.0 free=34055647232 total=34057748480
GPU2_RC=0
== GPU3 ==
checksum=134217728.0 free=34055647232 total=34057748480
GPU3_RC=0
QUICK_MACHINE_NOT_READY
```

### `ssh-0ef3419e`

Non-sandbox SSH:

```text
cc-0ef3419e-c93c-45ea-9219-fca0b2ef06de-0
2026-07-10T07:53:55+00:00
```

Strict CUDA quick check:

```text
HOST=cc-0ef3419e-c93c-45ea-9219-fca0b2ef06de-0
/usr/local/bin/python3
/usr/local/corex/bin/ixsmi
drwxr-xr-x ... /root/public-storage/models/Qwen/Qwen3.6-35B-A3B
== GPU0 ==
GPU0_RC=124
== GPU1 ==
GPU1_RC=124
== GPU2 ==
GPU2_RC=124
== GPU3 ==
GPU3_RC=124
QUICK_MACHINE_NOT_READY
```

### `ssh-3a52c74e`

```text
cc-3a52c74e-8d82-40c9-8901-6728a3c526fd-0
torch: 2.1.0
torch.cuda.device_count()=4
GPU0: PASS checksum=134217728.0 free=34055647232 total=34057748480
GPU1: FAIL timeout rc=124
GPU2: PASS checksum=134217728.0 free=34055647232 total=34057748480
GPU3: PASS checksum=134217728.0 free=34055647232 total=34057748480
NCCL: SKIPPED
```

### `ssh-4f83aab0`

```text
cc-4f83aab0-6bb7-45fa-b6b6-f20b295c5d38-0
torch: 2.1.0
torch.cuda.device_count()=4
GPU0..GPU3 all TimeoutExpired after 15s
NCCL: SKIPPED
```

### `ssh-ded5dcef`

```text
cc-ded5dcef-7aa0-449e-a445-c01d98731084-0
torch: 2.1.0
torch.cuda.device_count()=4
GPU0: FAIL_TIMEOUT rc=124
GPU1: FAIL_TIMEOUT rc=124
GPU2: PASS checksum=134217728.0 free=34055647232 total=34057748480
GPU3: PASS checksum=134217728.0 free=34055647232 total=34057748480
NCCL: SKIPPED
```

### `ssh-ea17e3be`

After correcting environment injection via `remote-run.sh`:

```text
cc-ea17e3be-f5d9-401d-99a5-225a7fbf3ed8-0
torch: 2.1.0
torch.cuda.device_count()=4
ixsmi: runnable
model dir: exists
GPU0: PASS checksum=-12.1875 free=34036772864 total=34057748480
GPU1: FAIL timeout rc=124
GPU2: PASS checksum=-3.09375 free=34036772864 total=34057748480
GPU3: PASS checksum=-22.625 free=34036772864 total=34057748480
NCCL: SKIPPED
```

### `ssh-a37d238f`

After correcting environment injection via `remote-run.sh`:

```text
cc-a37d238f-2aad-47a2-84b3-b3eff573425b-0
torch: 2.1.0
torch.cuda.device_count()=4
ixsmi: runnable
model dir: exists
GPU0: TIMEOUT
GPU1: TIMEOUT
GPU2: TIMEOUT
GPU3: TIMEOUT
NCCL: SKIPPED
```

### `ssh-0fa8ec7c`

```text
cc-0fa8ec7c-d2c6-4937-83cd-df0b15582388-0
SSH: OK
ixsmi: 4 BI100 visible
model dir: exists
GPU0: timeout
GPU1: timeout
GPU2: timeout
GPU3: timeout
NCCL: SKIPPED
```

### Original `ssh-7f994958`

This was the original work instance. It was initially healthy and ran T1-T5
baseline work. After a profiling/startup failure, repeated preflight showed:

```text
GPU0: timeout
GPU1: timeout
GPU2: PASS checksum=1073741824.0
GPU3: PASS checksum=1073741824.0
NCCL: all ranks timeout after GPU0/GPU1 became unhealthy
```

## Why This Is Probably Not Our vLLM Code

The failing check does not import or execute:

- `vllm`
- `qwen3_6_scripts`
- patched GDN/MoE/attention files
- benchmark client
- model weights
- OpenAI API server

It only imports `torch`, selects a GPU using `CUDA_VISIBLE_DEVICES`, creates a
small BF16 tensor, runs one matrix multiply, and synchronizes. Failures at this
level point below the application layer.

## What To Ask The Platform To Check

For a platform support ticket, the useful problem statement is:

```text
Multiple BI100 instances expose 4 GPUs via ixsmi and torch.cuda.device_count(),
but a minimal torch CUDA BF16 matmul hangs until timeout on one or more devices.
This happens before vLLM or model code is involved.
```

Ask them to verify:

- whether the container is launched with the correct BI100/CoreX device runtime;
- whether all 4 device nodes are usable from inside the container, not just
  visible to `ixsmi`;
- whether there are stale GPU contexts or wedged device states on affected hosts;
- whether `torch==2.1.0` with the bundled CoreX runtime is expected to run the
  minimal BF16 matmul on all cards in this image;
- whether an instance should be considered healthy only after a per-GPU CUDA
  kernel execution test, not just `ixsmi`.

## Current Gate For Continuing Work

Do not sync code, start TP=4 vLLM, run smoke, or run benchmark until an instance
passes:

```text
GPU0: PASS
GPU1: PASS
GPU2: PASS
GPU3: PASS
TP=4 NCCL all_reduce: PASS
```

