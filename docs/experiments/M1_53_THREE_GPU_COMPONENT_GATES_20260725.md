# M1-53 三卡组件门禁与运行时可复现性

## 结论

`ssh-73ca29ba` 当前只有物理 GPU1、GPU2、GPU3 健康。模型有 16 个全局
query heads，不能被 TP3 整除，因此本轮没有启动模型服务，也没有把三卡
结果外推成 TP4 性能、质量或竞赛分数。

本轮完成了两个可独立成立的结论：

1. commit `fbb37b2` 修复了 bare-host overlay 的非确定身份。同一源码连续
   安装两次，runtime tree SHA 均为
   `75b3c665a9767968628aecf1d15d488a89884327c73eacdfded5a55a1bdb42f1`，
   `diff -qr --exclude=__pycache__` 无差异。
2. 既有 packed QGKV 候选在四个 TP4 逻辑 rank 上权重和输出逐 bit 一致；
   三张物理卡的固定单卡基准也一致，但仍需 TP4 服务 A/B 和全部质量门禁。

结构化证据见
`docs/experiments/evidence/M1_53_THREE_GPU_COMPONENT_GATES_20260725.json`。

## 修复

`pip --target` 会把随机的 `/tmp/bi100-patch-stage.*` wheel 路径写入
`transformers-4.55.3.dist-info/direct_url.json`，随后改变 `RECORD`，使相同
源码的 runtime tree SHA 不稳定。

`scripts/normalize_offline_distribution.py` 现在：

- 用固定的 `file:///offline/<wheel-name>` URL 记录离线来源；
- 绑定 wheel SHA-256；
- 按 Wheel `RECORD` 规范重算 URL-safe base64 SHA-256 和大小；
- 对缺失或重复 distribution/RECORD 项 fail-fast；
- 保证重复执行字节级幂等。

安装器在计算 runtime tree SHA 前调用该脚本，并把规范化器自身 SHA 写入安装
报告。`tests/verify_bare_host_runtime_identity.py` 会验证该身份。安装器还会在
做任何工作前处理 `--help` 和拒绝未知选项，避免把选项误当成报告路径。

与 parent `5bb8536` 的 runtime 比较时，排除 `direct_url.json` 和 `RECORD`
后实现文件差异为零。因此本提交的 runtime 变化只属于离线发行包来源元数据，
不是模型计算或请求语义变化。

## 三卡结果

### TP4 rank-local QGKV

逻辑 rank 0/1/2 分别在物理 GPU1/2/3 上并行测试，rank 3 随后在 GPU1 上
测试。每个 rank 的 Q、K、V 加载、三个权重段和三个输出段全部 exact，
`max_abs=0`。

固定基准未扫描参数：

| tokens | separate median 范围 | packed median 范围 | speedup 范围 |
|---:|---:|---:|---:|
| 1 | 0.051657-0.054018 ms | 0.027249-0.027782 ms | 1.8932-1.9443x |
| 64 | 0.093423-0.096777 ms | 0.066025-0.068032 ms | 1.4112-1.4225x |

该测试使用固定 seed 的合成 rank-local 权重，不加载模型 checkpoint。它只证明
loader 数学和单卡 GEMM，不证明 TP4 checkpoint 加载、collective、端到端收益
或模型能力。

### 通信与清理

- 三 rank NCCL all-reduce 均为预期值 `6.0`。
- vLLM group preflight 的三个 rank 均到达 `collectives_done`，结果为 `6.0`。
- 标准 IxFormer NCCL 路径 `parity_max_abs=0`，明确
  `ipc_initiated=false`。
- 最终 postflight 中三卡均有 `34,057,748,480` 字节可用，矩阵 checksum
  均为 `1,073,741,824`，无残留 benchmark、worker 或 API server 进程。

## 复现边界

运行时双安装使用：

```bash
BI100_BARE_HOST_RUNTIME_ROOT=/root/competition-three-gpu-runtime-fbb37b2-a \
  bash scripts/install_bi100_bare_host_runtime.sh /tmp/install-a.json
BI100_BARE_HOST_RUNTIME_ROOT=/root/competition-three-gpu-runtime-fbb37b2-b \
  bash scripts/install_bi100_bare_host_runtime.sh /tmp/install-b.json
diff -qr --exclude=__pycache__ \
  /root/competition-three-gpu-runtime-fbb37b2-a/site-packages \
  /root/competition-three-gpu-runtime-fbb37b2-b/site-packages
```

QGKV 固定网格为 `tokens=1,64`、warmup 30、iterations 200、repeats 9。
通信固定使用 world size 3 和 FP16 元素数 2048/8192/65536，未启用 IPC。
原始 JSON 未提交，只在结构化证据中保存其 SHA-256 和必要统计。

## 决策

运行时可复现性修复通过，可作为后续 TP4 实验基础。packed QGKV 只通过了
rank-local 精确性和单卡 microbenchmark，不能据此修改 `main`、
`computility-run.yaml` 或默认开关。

待四卡恢复后必须继续执行：

1. TP4 checkpoint 加载和服务启动；
2. 同一 runtime 的 control/candidate A/B；
3. cold/warm token、tool call、reasoning、content 和多模态一致性；
4. 完整功能、长上下文、性能和 fatal/OOM/worker-loss 门禁。

这些门禁通过前，模型能力不回退和端到端收益均未建立。
