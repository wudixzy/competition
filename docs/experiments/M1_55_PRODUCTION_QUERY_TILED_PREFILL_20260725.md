# M1-55 生产形状分页 Prefill 内核门禁

## 结论

M1-55 按预设停止规则关闭，不进入默认运行时。旧的 full-query split4 候选在
Q=256 的旧 microbenchmark 上达到 `2.341x`，但在真实生产 query 长度上只有
`1.175-1.191x`，低于 `1.5x` 门槛，并额外占用约 `345-500 MiB` 工作区。

新的 query-tiled 主设计消除了 full-query 全局 scores 和 split output，但在
固定小型分页输入 `context=240, query=16` 上未通过数值门禁：

- 最佳主实现相对 L2 为 `1.3592e-5`；
- 唯一允许的固定 split-reduction 备选为 `1.4863e-5`；
- 固定门槛为 `1e-5`，没有放宽；
- LSE 相对误差均约 `4.96e-8`，偏差集中在 PV 累加顺序；
- 小型分页路径自身也只有 `0.407-0.414x`，没有性能晋升依据。

因此没有运行 query-tiled 的 65K/128K/235K 生产形状，也没有启动模型服务。
该候选没有写入 `patch_ops.sh`，没有修改 `main`、`computility-run.yaml` 或
任何默认开关。结构化证据见
`docs/experiments/evidence/M1_55_PRODUCTION_QUERY_TILED_PREFILL_20260725.json`。

## 固定实验合同

实验分支为 `exp/M1-55-production-q-prefill-20260725`，冻结基线为
`a594c55`。生产门禁 harness 从 `9131623` 开始固定：

- FP16 输入和输出、FP32 QK/PV/online-softmax；
- head dim 256、4 个 query heads、1 个 KV head、block size 16；
- 512-token K 方向在线 softmax 分区；
- seed `20260725`、warmup 1、trial 3；
- 相对 L2 `<=1e-5`、max abs `<=1e-3`；
- 生产形状核心路径 speedup `>=1.5x`；
- 不扫描 query tile、split 数、阈值或 YAML 参数。

实例为 `ssh-73ca29ba`，只使用物理 GPU1、GPU2、GPU3。运行时 tree SHA-256
为 `75b3c665...42f1`。输入由固定 seed 合成，不加载模型权重，因此本实验只能
建立算子数值和性能结论，不能建立模型能力或端到端吞吐结论。

## 旧 Split4 生产形状

旧内核保留完整 query 的 FP32 score、split output 和其他全局中间量。其
Q=256 探针与生产形状差异很大：

| case | reference | candidate | speedup | rel L2 | 增量显存 |
|---|---:|---:|---:|---:|---:|
| 74K / Q256 | 49.475 ms | 21.130 ms | 2.341x | 4.92e-6 | 19.5 MiB |
| 65K / Q8176 | 672.627 ms | 572.558 ms | 1.175x | 5.07e-6 | 500.4 MiB |
| 128K / Q8176 | 1195.496 ms | 1011.538 ms | 1.182x | 5.28e-6 | 500.4 MiB |
| 235K / Q5616 | 1515.243 ms | 1272.078 ms | 1.191x | 5.67e-6 | 345.2 MiB |

数值通过但三个生产形状均未达到 `1.5x`。这证明 Q256 microbenchmark 不能
代表真实 chunked-prefill query 长度，也关闭了继续调旧 split4 参数的路径。

## Query-Tiled 主设计

主设计使用一个 64-thread BI warp/CTA 处理一个 query head 和 16-token query
tile，直接通过 block table 读取分页 K/V。QK、PV 使用 CoreX FP32 WMMA，
在线 softmax 状态保留在 shared memory，不生成 full-query logits。

实现经历三次有因果依据的修正：

1. 初版每 16 token 合并一次 softmax，分页门禁 rel L2 为 `1.8509e-5`。
2. 对齐参考实现的 512-token context/current 分区后降到 `1.4913e-5`。
3. 在单个 FP32 WMMA accumulator 中完成每组 PV 后降到 `1.3592e-5`。

第三版的固定结果为：

| case | rel L2 | LSE rel L2 | speedup | 结果 |
|---|---:|---:|---:|---|
| dense Q1 | 0 | 0 | 2.158x | 通过 |
| dense Q256 | 7.447e-6 | 4.625e-8 | 0.304x | 数值通过 |
| paged 240 + Q16 | 1.359e-5 | 4.955e-8 | 0.407x | 失败 |

LSE 几乎一致而输出超限，将剩余问题定位到 PV reduction order，不是页表、
causal mask 或在线 softmax状态错误。

## 唯一 Split-Reduction 备选

按照方案只实现一次替代：把每个 512-token PV reduction 固定拆成四个连续的
128-token FP32 WMMA partial，再按固定二叉树合并。没有扫描 split 数。

该版本 dense Q256 仍通过，paged 240 + Q16 却回退到 `1.4863e-5`，且速度仅
`0.414x`。继续改变 split 或 tile 将违反停止规则，因此不再尝试 Kahan、不同
split 数、近似 exp 或放宽容差。

## 能力与发布边界

M1-55 未启动 API 服务，以下项目均没有新增通过声明：

- tool calling、reasoning、thinking 和多模态；
- cold/warm token 完全一致；
- 65K、131K、235K 和 262144 长上下文；
- Output TPS、TTFT、缓存命中和 weighted score；
- 完整 `指标集合` 与质量数据集。

候选从未安装到运行时，也未改变请求语义，因此现有默认方案的模型能力没有被
本轮代码改变。但这不等于候选本身通过模型能力门禁。

实验结束后 GPU1/2/3 均回到 257 MiB，无残留 benchmark、worker 或 API
server 进程，未观察到 fatal、OOM、Gloo reset 或 worker loss。

## 决策

- M1-55 query-tiled fused prefill：停止。
- 旧 full-query split4：停止。
- 生产 query-tiled 测试：因小型分页数值门禁失败而未运行。
- `main`、正式 YAML、默认开关：保持不变。
- 后续优化必须回到端到端 profile，选择新的精确语义方向，不能继续扫描本内核
  的 tile、split 或阈值。

## 后续方向复核

M1-55 关闭后又对已有证据做了固定收益上限复核，没有直接开始新的参数或内核
扫描。

CPU KV block-major 传输曾在裸连续布局探针上达到较高带宽，但 M1-49 已把每
rank GPU KV 容量提高到约 `1,080,192` token。仓库仍没有完整同会话、固定顺序
的 `1..881` v4 trace。13 请求选取集只有 `6,089` prompt tokens，无法触发该
容量下的 CPU offload，所以它只能继续作为功能和 prefix smoke，不能解锁
M1-46 生产实现或默认路径。

`norm_rope` 也不具备足够上限。M1-48 的 235K profile 中：

- `layer.full_attn` 为 `356,622.656 ms`；
- `paged_attn.prefix_pytorch` 为 `348,123.654 ms`；
- `full_attn.output_proj` 为 `6,459.109 ms`。

三者相减只剩约 `2,039.893 ms`，其中还包含 QGKV projection、norm/RoPE、
gate、attention 和计时残差。相对 `526,192.057 ms` 模型时间，其全部理论上限
不足 `0.39%`；单独 `norm_rope` 必然更低，即使无限加速也达不到预设 `1%`
端到端最低收益，因此不实现融合版本。

在三卡条件下，剩余高价值工作需要以下至少一项外部条件：

1. 一份与当前源码、overlay 和指标同轮绑定的完整 881 privacy-safe trace，用于
   判断缓存准入及 CPU tier 是否值得真实 A/B；
2. 四张健康卡，用固定 TP4 服务重新测 control/candidate、完整质量与 262144
   容量门禁。

条件满足前不应通过继续调 attention tile、CPU transfer chunk、YAML 或低占比
算子制造无依据的进展。

## 本地验证

- M1-55 定向单测：21 项，失败 0；
- 完整 `tests/` 单测：681 项，跳过 25，失败 0；
- submission preflight：9/9；
- Python 语法：254 个源文件通过；
- shell 语法：33 个脚本通过；
- 质量指标与数据 provenance manifest：通过；
- JSON、diff 和敏感制品扫描：通过。
