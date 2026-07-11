# EngineX vLLM BI100 Qwen3.6-35B-A3B 交接总结

更新时间：2026-07-12

## 2026-07-12 固定评测契约更正

主办方评测命令已经确认固定：`concurrency=1`、TP=4、
`max_num_seqs=1`、`max_num_batched_tokens=8192`、
`gpu_memory_utilization=0.9`、`max_model_len=100000`，完整参数以
`computility-run.yaml` 为准。此后 benchmark 必须使用 `workers=1`，不得用
`NUM_GPU_BLOCKS_OVERRIDE` 跳过 synthetic profile。

因此下文所有 `max_num_seqs` 扫描、`workers=4` 和 block override 结果仅保留为
历史诊断，不是有效竞赛基线。T5 当前任务改为修复 exact-contract 启动时 GDN
synthetic profile 的 non-finite failure，再建立无 override 的单并发基线。

数据集报告要求 235K+ 输入/256K 上下文，但固定命令仅允许 100K。模型配置原生
支持 262144；实测每卡权重 16.2303 GB，18,000 个 cache blocks 曾报告约 288K
token 容量，因此四卡显存并非明显障碍。100K 限制是否与数据集冲突需主办方确认。

## 2026-07-12 干净实例 T1-T4 重建

当前唯一远端目标为 `ssh-987372d0.default.gpu.phanthy.com`，远端代码目录为
`/root/qwen36-bi100-submission`，模型目录为
`/root/public-storage/models/Qwen/Qwen3.6-35B-A3B`。该实例的可写层在断线后
曾被重置，本轮因此从干净 vendor 环境重建，不依赖旧补丁残留。

同步代码包（不含本地 wheel）：

```text
Archive: /tmp/qwen36-submission-t1-t4-code-fixed.tgz
Local/remote size: 159529 bytes
Local/remote sha256: f66d98595ca5b9db388b2d2e34a42897e76c6ea51c5361467dedd02f9b1a1107
```

`transformers 4.55.3` wheel 按官方仓库使用的清华 PyPI 源在远端一次性暂存，
然后由本项目 `patch_ops.sh` 以 `--no-index --no-deps --find-links=./wheels`
离线安装。远端制品与本地制品一致：

```text
Size: 11269669 bytes
sha256: c85e7feace634541e23b3e34d28aa9492d67974b733237ade9eba7c57c0fd1bd
```

本轮首次在干净 vendor `worker.py` 上暴露
`patch_worker_profile_override.py` 只接受旧 guard 布局的 anchor 问题。修复后它同时
支持干净布局、已有 `BI100_IN_STARTUP_PROFILE` guard 的布局和二次幂等执行，
未知布局仍 fail-fast。

最新门禁：

```text
Hardware: CUDA GPU0..3 PASS; NCCL rank0..3 PASS, all_reduce=10.0
T1: bash/py_compile PASS; worker patch unit 2/2; P0 static 32/32; serving chat 3/3
T2: patch_ops.sh first run PASS; second idempotent run PASS; patched imports PASS
T3: paged-attn/env tests 5/5; default accepted; out-of-range value rejected
T4: profiler default-off PASS; enabled GPU probe=1024.0, timer=2.308 ms
T4: GDN parity 1/1 PASS; MoE parity 1/1 PASS; no skips
```

本轮 T1-T4 已完成。旧 T5 批处理扫描方案已因固定评测契约作废；历史 T5 实验与
旧实例故障记录仅保留作诊断依据。

## 当前结论

当前提交目录是 `localCode4ref/enginex-vllm-bi100-qwen36`。本轮工作已经把
P0 正确性补丁、轻量单测、远端同步方式和 BI100 预检查流程整理到可交接状态。

2026-07-06 T1-T4 已完成并验证：T1 代码卫生、T2 离线构建门禁、T3 env 强校验
与 knob 登记、T4 profiler/parity harness 已上线。按当前任务要求，T4 完成后
旧流程曾要求在此暂停；当前 T5 已改为 exact-contract 启动修复和单并发基线。

本轮 T1-T4 远端验证使用的小代码包（不含本地 wheel 大文件）：

```text
Archive: /tmp/qwen36-submission-t1-t4-code.tgz
Local sha256: bbd363a148bfce5dd9c1d53307bd1c73f1e2f0ea6020f775350f3cbf4e5f7f11
Remote sha256: bbd363a148bfce5dd9c1d53307bd1c73f1e2f0ea6020f775350f3cbf4e5f7f11
Remote size: 151149 bytes
Remote test dir: /root/qwen36-bi100-submission
```

说明：本地完整提交树包含 `qwen3_6_scripts/wheels/transformers-4.55.3-py3-none-any.whl`
作为离线构建资产。当前 SSH 通道无法稳定传输 11 MB 完整 archive，本轮远端验证是在
解压小代码包后，将同一 wheel 放入远端 `qwen3_6_scripts/wheels/`，再执行
`patch_ops.sh`。`patch_ops.sh` 本身通过 `--no-index --no-deps --find-links=./wheels`
离线安装，没有在 patch/build 阶段访问公网。

T1-T4 最新远端 gate：

```text
python3 -m py_compile tests/*.py qwen3_6_scripts/*.py qwen3_6_scripts/qwen3_5/*.py qwen3_6_scripts/qwen3_5_moe/*.py: PASS
bash -n launch_service && bash -n qwen3_6_scripts/patch_ops.sh: PASS
cd qwen3_6_scripts && bash ./patch_ops.sh: PASS, [ok] patched vllm imports
tests/test_p0_static.py: 31 tests OK
tests/test_clients_unit.py: 5 tests OK
tests/test_paged_attn_unit.py: 5 tests OK
tests/test_patch_utils_unit.py: 9 tests OK
tests/test_patch_transformers_unit.py: 2 tests OK
tests/test_patch_registry_unit.py: 2 tests OK
tests/test_patch_tool_parser_unit.py: 2 tests OK
tests/test_protocol_unit.py: 6 tests OK
tests/test_serving_chat_unit.py: 3 tests OK
tests/test_tool_parser_unit.py: 1 test OK
tests/test_preflight_unit.py: 6 tests OK
tests/test_gdn_parity.py: 1 test OK
tests/test_moe_parity.py: 1 test OK
```

2026-07-06 T5 批处理扫描已开始，但在 S2 发现问题并按规则暂停：

```text
Hardware gate before T5:
tests/bi100_preflight.py --timeout-s 25 --matmul-size 1024: PASS on GPU0..3
tests/bi100_nccl_preflight.py --timeout-s 45: PASS, all_reduce value=10.0 on rank0..3

S1 max_num_seqs=1:
RUN_DIR=bench_runs/20260706_145437_T5_S1_maxseq1
quick smoke: PASS
full smoke: PASS
benchmark:
  success_rate=1.0
  ttft_p90_s=29.444989765249193
  output_tps_p10=1.6708191586237287
  input_tps=188.95996048448603
  output_tps_total=6.63984305591319
  cache_tps=164.60512113315227
  cache_hit_rate=0.8711111111111111
  weighted_score=649.1408758188858

S2 max_num_seqs=2:
RUN_DIR=bench_runs/20260706_150132_T5_S2_maxseq2
startup failed during determine_num_available_blocks/profile_run:
  RuntimeError: non-finite values in prefill-norm GatedDeltaNet layer 0 (frac=0.6251)
  observed on driver and worker ranks.
Result: S2 not benchmarkable; S3/S4 not run yet.

T5 mitigation added after discussion/resume:
  qwen3_6_scripts/patch_worker_profile_override.py patches vllm/worker/worker.py
  so an explicit --num-gpu-blocks-override skips only synthetic profile_run.
  launch_service exposes NUM_GPU_BLOCKS_OVERRIDE. Default remains empty, so
  normal startup profiling and real inference fail-fast behavior are unchanged.
  Remote validation:
    python3 tests/test_p0_static.py: 32 tests OK
    cd qwen3_6_scripts && bash ./patch_ops.sh: PASS

S2 retry with max_num_seqs=2 and NUM_GPU_BLOCKS_OVERRIDE=18000:
RUN_DIR=bench_runs/20260706_150821_T5_S2_maxseq2_override18000
startup: PASS, logs include "[BI100] skipping worker.profile_run..."
quick smoke: PASS
benchmark:
  success_rate=1.0
  ttft_p90_s=35.53355162981897
  output_tps_p10=1.0554623468258624
  input_tps=125.25225656307974
  output_tps_total=4.448470821330246
  cache_tps=108.98753512259103
  cache_hit_rate=0.8701442841287459
  weighted_score=429.34163136599835
Result: S2 is functionally runnable but materially worse than S1, so S3/S4 were
not run. This suggests current batched prefill/decode path has extra overhead or
less favorable scheduling under the pure-PyTorch GDN fallback. Next work should
investigate with BI100_PROFILE on S1/S2 or move into T6 GDN prefill work before
resuming larger max_num_seqs scans.

Follow-up profile attempt:
  RUN_DIR=bench_runs/20260706_151539_T5_PROFILE_S1_maxseq1
  MAX_NUM_SEQS=1 BI100_PROFILE=1 without NUM_GPU_BLOCKS_OVERRIDE failed during
  startup synthetic profile_run:
    RuntimeError: non-finite values in prefill-norm GatedDeltaNet layer 0
    observed with fractions 0.3750 / 0.1250 after first L0.gdn.prefill timings.
  After killing the failed service, BI100 preflight became unhealthy:
    GPU0: timeout
    GPU1: timeout
    GPU2: PASS
    GPU3: PASS
  Current state: do not start TP=4 service or benchmark until GPU0/GPU1 pass
  `tests/bi100_preflight.py` again. Future BI100_PROFILE runs should either use
  `NUM_GPU_BLOCKS_OVERRIDE` to skip synthetic profile_run or adjust the profiler
  so startup dummy profiling is excluded from timing.

Retry / safety patch:
  Re-ran `tests/bi100_preflight.py --timeout-s 25 --matmul-size 1024`:
    GPU0: timeout
    GPU1: timeout
    GPU2: PASS
    GPU3: PASS
  Added BI100_PROFILE startup guard:
    qwen3_6_scripts/patch_worker_profile_override.py marks synthetic
    worker.profile_run with BI100_IN_STARTUP_PROFILE=1.
    qwen3_6_scripts/bi100_profile.py skips timers during that phase by default.
    BI100_PROFILE_INCLUDE_STARTUP=1 opts back in for debugging.
  Validation:
    local py_compile + bash -n: PASS
    local tests/test_p0_static.py: 32 tests OK
    remote archive /tmp/qwen36-profile-safety.tgz
      sha256=a8469bec1bcfa7bf6e5cdefd9db3ecabb1b87733bbbbd582b9224ef515cce27c
    remote tests/test_p0_static.py: 32 tests OK
    remote `cd qwen3_6_scripts && bash ./patch_ops.sh`: PASS, [ok] patched vllm imports
  Still blocked for performance work until GPU0/GPU1 preflight recovers.

Third preflight retry:
  Command:
    python3 tests/bi100_preflight.py --timeout-s 25 --matmul-size 1024 \
      --json-out /tmp/bi100_preflight_retry3.json
  Result:
    GPU0: timeout
    GPU1: timeout
    GPU2: PASS
    GPU3: PASS
  No active vLLM/api_server processes were present before the retry; `ixsmi`
  showed only baseline 257 MiB per GPU and no listed compute processes. This
  confirms the current blocker is external GPU/runtime state, not a live service
  consuming memory. TP=4 service, benchmark, T5 profile, and T6 performance work
  remain blocked until all four GPUs pass preflight.
```

Interpretation: S1 reproduces the strict A-fp32norm baseline. S2 failure is not
OOM and not a hardware/NCCL gate failure; GPU memory returned to idle after the
crash. The current fail-fast GDN non-finite check correctly exposed a numerical
issue in vLLM's startup profile run for batched prefill. Do not enable
`BI100_GDN_ALLOW_NAN_ZERO=1` as a scoring workaround. Next discussion should
decide whether to profile S1/S2 scheduling overhead or move directly into T6 GDN
numerical/tensorization work before resuming S3/S4.

最新远端验证 archive：

```text
Archive: /tmp/qwen36-submission-min.tgz
Local sha256: ff61e8ff1997fc9eda3e56cb77801853c39a8de84bc887d98aeed6ca04389b80
Remote sha256: ff61e8ff1997fc9eda3e56cb77801853c39a8de84bc887d98aeed6ca04389b80
Remote size: 129573 bytes
Remote test dir: /root/qwen36-bi100-submission
```

远端静态/单元测试已通过。2026-07-06 11:35 UTC 重新执行 BI100 CUDA 与 NCCL
preflight 后，4 张卡均已通过，硬件门禁当前全绿。本文档只记录 smoke 准备状态；
本轮未启动 TP=4 vLLM、未 benchmark、未 Docker build。

## 代码状态

主要代码以 `qwen3_6_scripts/` 热补丁方式覆盖基础镜像内 vLLM/transformers：

- `patch_ops.sh` 动态定位 `VLLM_ROOT` 和 `TRANSFORMERS_ROOT`，最终 Docker build
  日志应能看到这两个路径。
- `patch_utils.py` 提供 `package_root`、`replace_once`、`replace_one_of` 等
  fail-fast patch helper。
- `patch_vllm_qwen3_5.py` 注册 Qwen3/Qwen3_5/Qwen3_6 registry aliases，
  不需要手动修改 `/model/config.json`。
- `patch_transformers_qwen3_5.py` 注册 qwen3_5/qwen3_5_moe transformers config。
- `protocol.py` 支持顶层 `thinking` 关闭格式和 `tool_choice="none"`。
- `serving_chat.py` 修复 streaming tool arguments double-json 问题。
- `qwen3coder_tool_parser.py` 修复 streaming argument key JSON 转义，并让参数解析
  fallback 可观测。
- `paged_attn.py` 默认在 block table 不足时 raise，只有
  `BI100_ALLOW_PREFIX_GUARD_CAP=1` 才允许 debug cap；同时支持
  `BI100_PYTORCH_DECODE_THRESHOLD`、`BI100_PREFIX_BLOCKS_PER_TILE` 和
  `BI100_FORCE_PAGED_ATTN_V2=1` opt-in 调参。
- `qwen3_5.py` 中 GatedDeltaNet 默认遇到非有限值直接 raise；只有
  `BI100_GDN_ALLOW_NAN_ZERO=1` 才会风险日志后置零。
- `launch_service` 已与 `computility-run.yaml` 的竞赛入口参数对齐，并补齐 CoreX
  `PATH`/`LD_LIBRARY_PATH`/`PYTHONPATH`。

## 当前测试覆盖

本地与远端静态/单元 gate 均已通过：

```text
python3 tests/test_p0_static.py: 28 tests OK
python3 tests/test_clients_unit.py: 5 tests OK
python3 tests/test_paged_attn_unit.py: 4 tests OK
python3 tests/test_patch_utils_unit.py: 9 tests OK
python3 tests/test_patch_transformers_unit.py: 2 tests OK
python3 tests/test_patch_registry_unit.py: 2 tests OK
python3 tests/test_patch_tool_parser_unit.py: 2 tests OK
python3 tests/test_protocol_unit.py: 6 tests OK
python3 tests/test_serving_chat_unit.py: 3 tests OK
python3 tests/test_tool_parser_unit.py: 1 test OK
python3 tests/test_preflight_unit.py: 6 tests OK
python3 -m py_compile tests/*.py qwen3_6_scripts/*.py \
  qwen3_6_scripts/qwen3_5/*.py qwen3_6_scripts/qwen3_5_moe/*.py: PASS
bash -n qwen3_6_scripts/patch_ops.sh launch_service: PASS
```

这些测试覆盖：

- P0 静态合约、patch fail-fast、动态路径、registry alias。
- protocol thinking、`tool_choice="none"`、guided decoding/tool conflict。
- serving chat tool argument serialization。
- Qwen3 coder parser streaming key escaping。
- patch scripts 对 fake vLLM/transformers package 的可执行验证。
- BI100 CUDA/NCCL preflight helper 逻辑。
- smoke/benchmark client 对 usage-only SSE chunk、HTTP error body、分位数和加权分公式的处理。
- paged attention env 默认值/覆盖值、V2 opt-in、prefix block table guard。

## 远端环境与阻塞点

远端目录：

```text
/root/qwen36-bi100-submission
```

远端模型：

```text
/root/public-storage/models/Qwen/Qwen3.6-35B-A3B
```

注意：远端 `/model` 当前不存在，真实服务启动前需要按竞赛入口或容器挂载提供
`/model`。

最新 BI100 CUDA 预检查结果（2026-07-06 11:35 UTC）：

```text
bash scripts/remote-run.sh 'python3 tests/bi100_preflight.py --timeout-s 25 --matmul-size 1024 --json-out /tmp/bi100_preflight.json'

GPU0: PASS, checksum=1073741824.0
GPU1: PASS, checksum=1073741824.0
GPU2: PASS, checksum=1073741824.0
GPU3: PASS, checksum=1073741824.0
bi100_preflight_rc=0
```

最新 BI100 NCCL 预检查结果（2026-07-06 11:36 UTC）：

```text
bash scripts/remote-run.sh 'python3 tests/bi100_nccl_preflight.py --timeout-s 45 --json-out /tmp/bi100_nccl_preflight.json'

rank0/gpu0: PASS, all_reduce value=10.0
rank1/gpu1: PASS, all_reduce value=10.0
rank2/gpu2: PASS, all_reduce value=10.0
rank3/gpu3: PASS, all_reduce value=10.0
bi100_nccl_preflight_rc=0
```

因此当前已满足 TP=4 smoke 的硬件前置门禁，但尚未启动服务。下一步若进入 smoke，
仍需先确认远端目录与本地代码同步，记录 archive sha256/size，然后再启动服务。

## 历史一手报错与处理记录

### 1. 初始可用基线与模型识别

`patch_ops.sh` 在远端首次完成动态路径定位，未出现 anchor 缺失：

```text
VLLM_ROOT=/usr/local/corex-3.2.3/lib64/python3/dist-packages/vllm
TRANSFORMERS_ROOT=/usr/local/lib/python3.10/site-packages/transformers
No anchor-not-found warnings.
Model config architectures: ["Qwen3_5MoeForCausalLM"]
```

Config A 是当前最有价值的早期性能参考，但它发生在后续 GPU1 wedge 之前，且当时
GatedDeltaNet 仍会静默 zero-fill NaN：

```text
max_num_batched_tokens=8192
gpu_memory_utilization=0.90
BI100_PREFIX_BLOCKS_PER_TILE=32
API smoke: basic/tool_choice none/json_object/streaming/bad request/prefix cache PASS
prefix cache cached_tokens=2416
success_rate=100%
TTFT P90=17.97s
Output TPS P10=8.68
Input TPS=324.29
Output TPS total=5.43
Cache TPS=279.78
Cache hit rate=86.27%
Weighted smoke score=1155.52
```

Config B 只提高 `max_num_batched_tokens` 到 12288，结果比 A 差，不建议作为优先
基线：

```text
success_rate=100%
TTFT P90=18.62s
Output TPS P10=7.89
Input TPS=311.72
Output TPS total=5.22
Cache TPS=268.94
Cache hit rate=86.27%
Weighted smoke score=1110.74
```

### 2. GatedDeltaNet NaN 与数值修复实验

早期默认 dtype 路径多次记录以下风险日志，API smoke 虽然通过，但这是效果正确性
风险，不应当被当作最终可接受状态：

```text
NaN in prefill GatedDeltaNet layer 32/33/34/36/37/38 (A) and 36/37/38 (B),
replacing with zeros.
```

`--dtype bfloat16` 能作为诊断手段绕开该 NaN，但性能明显退化，不能默认开启：

```text
Config A-bf16
GatedDeltaNet NaN warnings: none observed
success_rate=100%
TTFT P90=49.83s
Output TPS P10=0.37
Input TPS=132.90
Output TPS total=3.82
Cache TPS=116.10
Cache hit rate=87.36%
Weighted smoke score=443.27
```

当前代码采用 fp32-through-norm 修复，并把非有限值从静默置零改为默认 fail-fast。
这是后续优化应继续沿用的正确性基线：

```text
Config A-fp32norm
GatedDeltaNet NaN/non-finite warnings: none observed
success_rate=100%
TTFT P90=29.16s
Output TPS P10=1.73
Input TPS=189.01
Output TPS total=6.68
Cache TPS=165.02
Cache hit rate=87.31%
Weighted smoke score=650.47
```

严格检查下的 Config E 失败，说明 `gpu_memory_utilization=0.94` 与
`max_num_batched_tokens=12288` 当前不可用：

```text
RuntimeError: non-finite values in prefill-norm GatedDeltaNet layer 36 (frac=0.8750)
```

结论：不要把 `DTYPE=bfloat16`、`gpu_memory_utilization=0.94` 或更大的 batch/token
配置直接作为默认方案；先在健康 4 卡环境里从 A-fp32norm 继续做小步性能实验。

### 3. GPU1 / NCCL / vLLM 启动阻塞

vLLM TP=4 后续启动卡在 `init_device`，尚未进入 safetensors 加载：

```text
[BI100 startup] starting init_device
[BI100 startup] enqueue remote method=init_device workers=3
[BI100 startup] remote enqueued method=init_device
[BI100 startup] driver start method=init_device
[BI100 worker] start method=init_device
```

`CUDA_VISIBLE_DEVICES=1,2,3,0` 轮换物理 GPU 后仍卡在同一阶段。独立 probe 显示不是
模型加载问题，而是当前分配的 GPU/runtime 状态问题：

```text
4-rank torch.distributed NCCL probe:
  NCCL all_reduce failed because ranks timed out retrieving the NCCL unique ID
  from rank 0 via TCPStore.

4-rank probe with NCCL_P2P_DISABLE=1 and NCCL_IB_DISABLE=1:
  same timeout pattern; P2P/IB isolation did not recover collectives.

Per-GPU single-process CUDA matmul probe:
  GPU0: PASS
  GPU1: timeout after torch.cuda.set_device(1), before mem_get_info
  GPU2: PASS
  GPU3: PASS
```

reset 也无法在容器内完成：

```text
ixsmi -i 1 --gpu-reset failed with rc=255.
ixsmi reported hidden host PID 13269 using GPU 00000000:4C:00.0.
/proc/13269 was not visible inside the container.
ixsmi suggested using --pid=host or running reset outside the container.
```

结论：GPU1 修复或重新分配前，不要继续启动 TP=4 vLLM、NCCL preflight、benchmark
或 Docker build。先跑 `tests/bi100_preflight.py`，四张卡全绿后再继续。

### 4. SSH/同步断流与 archive 处理

远端文件同步过程中多次出现连接中断，不要把半同步目录误判为代码失败：

```text
Connection to ... closed by remote host
client_loop: send disconnect: Broken pipe
```

chunked uploader 曾观察到固定 offset 上传失败后重试成功：

```text
chunk 20 offset 40960 attempt 1 failed: Connection closed by UNKNOWN port 65535
chunk 62 offset 126976 failed on attempts 1-4 with "Connection closed by UNKNOWN port 65535"
```

处理方式：使用 `scripts/remote-upload-chunked.sh`，并只在 remote SHA256 和 size 与
local 完全一致后解压测试。失败或不确定时先删除 `/root/qwen36-bi100-submission`，
重新完整上传。

### 5. Shell CRLF、CoreX 环境和远端工具差异

脚本曾因 CRLF 在远端失败：

```text
patch_ops.sh: line 11: $'\r': command not found
patch_ops.sh: line 50: $'\r': command not found
```

这属于打包/换行问题，不是 patch 逻辑问题。当前静态测试已覆盖 `patch_ops.sh` 与
`launch_service` LF-only，并且最终 gate 包含 `bash -n`。

非交互 SSH 中必须显式导入 CoreX 环境；否则可能找不到 `ixsmi`，或 Python 能导入
`pydantic` 但找不到其传递依赖：

```text
ModuleNotFoundError: No module named 'typing_extensions'
```

推荐统一使用 `launch_service` 或 README 中的 CoreX env block。远端测试时还观察到
`curl` 不存在，HTTP smoke 请使用 `tests/smoke_api.py` 或 Python `urllib`，不要把
`curl: command not found` 误判为服务不可用。

### 6. Smoke 中已知的非阻塞日志

`tests/smoke_api.py --mode quick` 的 bad-request case 会主动发送 `messages=[]`，当前
服务会在返回 4xx 前打印 serving_chat/tokenizer traceback。只要 HTTP 结果为预期 4xx，
该 traceback 不代表模型加载或推理失败。

## 远端同步经验

SSH/stdin streaming 和 rsync 在该环境中可能出现 `Broken pipe` 或远端只同步部分文件。
当前稳定方式是小 archive + chunked fixed-offset upload：

```bash
tar --exclude='__pycache__' --exclude='.pytest_cache' --exclude='*.pyc' \
  -czf /tmp/qwen36-submission-min.tgz \
  -C localCode4ref/enginex-vllm-bi100-qwen36 \
  qwen3_6_scripts tests Dockerfile computility-run.yaml launch_service README.md docs

scripts/remote-upload-chunked.sh \
  /tmp/qwen36-submission-min.tgz \
  /tmp/qwen36-submission-min.tgz \
  2048

scripts/remote-shell.sh \
  'rm -rf /root/qwen36-bi100-submission &&
   mkdir -p /root/qwen36-bi100-submission &&
   tar -xzf /tmp/qwen36-submission-min.tgz -C /root/qwen36-bi100-submission'
```

观察到过 final chunk 多次 `Connection closed by UNKNOWN port 65535`，但 helper 会在同一
offset 重试，最终 SHA 和 size 匹配后再允许测试。

## 健康 4 卡环境到位后的下一步

先运行：

```bash
cd /root/qwen36-bi100-submission
python3 tests/bi100_preflight.py --timeout-s 25 --matmul-size 1024
python3 tests/bi100_nccl_preflight.py --timeout-s 45
```

只有两者都通过后，再按 `computility-run.yaml` 或 `launch_service` 启动 TP=4 服务。
服务可用后先跑 smoke，再跑 benchmark：

```bash
python3 tests/smoke_api.py --base http://127.0.0.1:8000 --mode quick
python3 tests/smoke_api.py --base http://127.0.0.1:8000 --mode full
python3 tests/bench_perf.py --base http://127.0.0.1:8000 \
  --label fixed-contract --requests 8 --workers 1 --stagger-s 0 \
  --max-tokens 64 --prompt-salt A-clean --out bench-A.json
```

性能矩阵建议按一次只改一个变量推进：

```text
A: max_num_batched_tokens=8192,  gpu_memory_util=0.90, tile=32
B: max_num_batched_tokens=12288, gpu_memory_util=0.90, tile=32
C: max_num_batched_tokens=16384, gpu_memory_util=0.90, tile=32
D: max_num_batched_tokens=12288, gpu_memory_util=0.92, tile=32
E: max_num_batched_tokens=12288, gpu_memory_util=0.94, tile=32
F: max_num_batched_tokens=12288, gpu_memory_util=0.92, tile=64
```

每轮记录：

- 请求成功率
- TTFT P90
- Output TPS P10
- Input TPS
- Cache TPS
- 缓存命中率
- Token 吞吐加权值

不要默认开启 FP8 KV cache 或 speculative decoding。当前 `paged_attn.py` fallback 没有
完整 FP8 KV dequant 覆盖，贸然启用会带来效果正确性风险。

## Docker 状态

当前阶段按项目要求未执行 Docker build。Docker build 是最终收尾步骤，应在远端
smoke、benchmark 和必要调参完成后再运行：

```bash
docker build -t qwen36-bi100-dev . 2>&1 | tee build.log
grep -E "VLLM_ROOT|TRANSFORMERS_ROOT" build.log
```

最终 Docker build 不应出现未解释的 `anchor not found`。
