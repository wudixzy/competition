# ModelHub TP4 Gloo peer reset on 2026-07-15

## Symptom

The ModelHub evaluator reached vLLM multiprocessing initialization with the
expected fixed command, TP=4, and `max_seq_len=262144`. Three remote workers
reported ready at `07:26:19`, but no safetensors loading or cache sizing log
followed. At `07:56:45`, about 30 minutes later, the remaining process logged:

```text
[E ProcessGroupGloo.cpp:138] Gloo connectFullMesh failed with Connection reset by peer
```

The preceding `pynvml` deprecation, missing Triton notice, TensorFlow oneDNN
message, and XFormers backend selection are informational. They also appear in
successful runs and do not explain a peer reset.

## Interpretation

The Gloo line is a secondary error: one process lost its peer while vLLM was
inside `init_device`. It does not show that model configuration, weights, or a
Gloo API call was the original failure. The 30-minute silent interval is
consistent with a GPU/rank stall followed by evaluator cleanup.

vLLM first creates the NCCL world and then creates additional NCCL device and
Gloo CPU groups for the world, tensor-parallel, and pipeline-parallel
coordinators. The older NCCL-only preflight did not cover this complete group
sequence, so `tests/bi100_vllm_group_preflight.py` now reproduces it with a
hard timeout and per-rank last-stage reporting.

## Healthy-host reproduction

The exact submitted candidate was reproduced on
`ssh-a163074c.default.gpu.phanthy.com`:

```text
branch: integration/tp4-candidates-20260715
commit: 6e0e66c0427cacb1c65e3364d3d3c4bc23056a15
model:  /root/public-storage/models/Qwen/Qwen3.6-35B-A3B
```

Evidence:

- all four independent CUDA allocation/synchronize/256x256 matmul probes
  passed with 34,057,748,480 bytes free per GPU;
- four-rank NCCL all-reduce passed with value `10.0` on every rank;
- the vLLM-equivalent NCCL+Gloo group probe passed on ranks 0-3, including all
  world, TP, and singleton PP groups, with no timed-out rank;
- the Docker-equivalent `patch_ops.sh` build completed and produced all seven
  CoreX extensions;
- the full TP4 service started at 262,144 context with 16,878 GPU cache blocks
  and 6,553 CPU cache blocks, then returned HTTP 200 from `/health`;
- quick smoke passed 8/8 and full smoke passed 15/15;
- forced 1,000-token decode completed in 88.332 seconds and matched the
  qualified SHA256 exactly:
  `1766c3c44bfb672e32b2e35419c5e06490e539e54250ab2fc1012c539e68835f`;
- the service log contained no traceback, Gloo failure, OOM, non-finite value,
  worker loss, or native crash.

Remote artifacts:

```text
/root/competition-candidate/build_a163.log
/root/competition-candidate/vllm_group_preflight_a163.json
/root/competition-candidate/service_a163.log
/root/competition-candidate/smoke_quick_a163.json
/root/competition-candidate/smoke_full_a163_v2.json
/root/competition-candidate/decode1000_a163.json
```

## Decision

Do not change Triton, TensorFlow, model loading, 262K context, or the TP4
implementation in response to this log. The same candidate builds, forms the
same groups, loads, and serves correctly on a healthy four-card host.

Re-run the candidate on a fresh evaluator allocation. If the stall repeats,
request per-GPU CUDA preflight and per-rank group-stage logs from the organizer.
`BI100_EXECUTOR_STARTUP_DEBUG=1` is enabled in the submission image so the next
run records whether each rank entered and completed `init_device`,
`load_model`, cache sizing, and cache initialization.
