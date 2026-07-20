# EngineX vLLM BI100 Qwen3.6-35B-A3B 交接总结

## 2026-07-21 M1-33 原生 64-token GDN 恢复候选

- M1-32 `admission64/aligned` 已通过 17 会话压力和修正后的
  235K/1,000-token exact 门槛；cold/warm 均生成 1,000 tokens，warm 命中
  229,376，消息 hash 同为 `a7d5a63c...81799`，两个最终 rc 均为 0。
- aligned 仅证明 8,192 边界正确，其固定矩阵命中上限仍只有 14.68%，不能晋级
  性能方案。`admission64/direct` 的错误边界 10,592 对齐 KV block 16，但落在
  DeltaNet 原生 64-token chunk 中间。
- 新增隔离的 `BI100_GDN_RESTORE_MODE=chunk64`，不修改
  `computility-run.yaml` 或默认 `fine32/direct`。CoreX 实测 64/128/2368/4096
  分割的输出与状态均逐位相等、最大误差 0；非对齐 2400 对照立即出现
  `2.682e-7` 状态误差，确认了根因方向。
- `scripts/run_m1_33_chunk64_gates.sh` 先做 GPU 精确性检查，只有通过才启动
  TP4 smoke、17 会话压力和 235K/1,000 exact；当前 TP4 门槛仍在运行，候选不得
  视为 qualified。完整证据见
  `docs/experiments/M1_33_GDN_CHUNK64_RESTORE_20260721.md`。

## 2026-07-21 M1-32 固定内核绝对基线

- `fine32/direct` 的生产等价 18 请求矩阵已完成，服务日志确认固定启用
  `BI100_MOE_COREX_DIRECT_ROUTED=1` 和 `BI100_GDN_COREX_PACKED_DECODE=1`。
- 成功率 `100%`、Output TPS P10 `21.6563`、有效命中 `49.9301%`、Input TPS
  `741.4479`、Cache TPS `7607.9233`、代理加权值 `6699.4888`。
- warm TTFT P90 仅 `1.4438s`，但 cold/全体 TTFT P90 为
  `20.9465s/20.8748s`。因此当前硬差距是冷 prefill：距 8000 仍差
  `1300.5112` 分，TTFT 距 5s 差 `15.8748s`；命中率只差 `0.0699` 个百分点。
- 2026-07-21 实例和 ModelHub 连接恢复；私有实验分支已推送到 `c156cc3`，并在
  `/root/competition-m1-32-latest` 启动剩余 direct/aligned 正确性门禁。

## 2026-07-20 M1-32 内容键 GDN 准入资格结果

- 数据集形状矩阵已固定为 18 个严格同请求样本，并校验 client/server token 数、
  16-token 目标误差、cold/warm salt 及跨策略完整请求合同；比较器拒绝非同请求 A/B。
- 同请求 TP4 相对结果中，`admission64/direct` 相比 `fine32/direct` 将有效命中率
  `49.9301% -> 61.0671%`、Input TPS `732.366 -> 824.430`、代理加权值
  `6281.291 -> 6846.025`（`+8.99%`）。该轮漏带评测固定的 MoE/GDN 内核环境，
  因此绝对 Output TPS 和得分只作诊断，不能作为正式基线。
- `admission64/direct` 在 17 会话压力后报告完整恢复 10592 tokens，但同一确定性请求
  的输出 hash 从 `bc4f55...` 变为 `eba366...`；服务健康且无 OOM/worker loss。
  `fine32` 压力复测实际已淘汰目标状态并全量重算，不能证明该 direct 边界正确。
  这是正确性硬失败，该策略不得设为默认、不得合入 `main`。
- `launch_service` 已补齐评测固定的
  `BI100_MOE_COREX_DIRECT_ROUTED=1`、`BI100_GDN_COREX_PACKED_DECODE=1`，修复后的
  `fine32/direct` 绝对基线见上方 2026-07-21 更新；此前网关中断现已恢复。
- `scripts/run_m1_32_remaining_gates.sh` 已固化恢复流程：验证固定矩阵后重启干净的
  fine/direct 服务执行 131K exact 和 235K warm-repeat，再重启 aligned 服务执行
  17 会话压力及 235K/1000 exact；每步均有超时、退出码和失败即停。
- 下一步只执行预设的 `admission64/aligned` 正确性回退；若压力测试和 235K/1000
  exact replay 未通过，不运行完整性能矩阵。即使通过，当前 4K/7.8K/16K 矩阵下
  aligned 理论命中上限也只有 `14.68%`，无法达到 50% 门槛。缓存阶段未通过前停止
  第三阶段融合分页注意力，不扫描 YAML、容量或 tile 参数。完整证据见
  `docs/experiments/M1_32_CONTENT_GDN_ADMISSION_20260720.md`。

## 2026-07-20 天垓100基础镜像地址迁移（强制）

- ModelHub 集群调整后，后续本地验证和性能榜单提交统一使用：
  `harbor.4pd.io/modelhubxc/enginex-iluvatar/bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3`。
- 旧地址
  `git.modelhub.org.cn:9443/enginex-iluvatar/bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3`
  已由平台迁移，不再作为任何分支或提交的构建基线。
- `Dockerfile` 和 `tests/submission_preflight.py` 已同步；提交前必须运行
  `python tests/submission_preflight.py`，防止回退到旧 registry。

## 2026-07-19 M1-31 稳定 GDN 前缀状态候选

- 已删除按可回收物理 block id 识别 GDN 状态的旧方案，改为
  `(完整 block 数, 链式 SHA-256)`；多模态请求把规范化后的图片/张量内容摘要
  纳入首块 namespace，不支持的类型按 request-local namespace 隔离，禁止跨请求
  误复用。
- scheduler 是 GDN 状态目录的唯一真相源，向 TP4 worker 明确广播
  `restore/capture/evict`。worker 不再猜测命中或自行 LRU；scheduler 要求恢复但
  worker 缺状态时直接报错，避免已经跳过输入后静默使用错误状态。
- 候选策略为 `fine32`（默认、容量 32）、`admission64`（容量 64，只保存重复
  branch 和 final）与 `off`。`direct` 保持默认；`aligned` 仅是 8192 边界回退。
  `computility-run.yaml` 的固定 262144/TP4/8192 参数未修改。
- trace 协议升级到 version 4，使用 `sha256_base64`，只在
  `BI100_CACHE_TRACE=1` 时输出。离线 simulator 同时报告 raw KV 连续命中和
  KV/GDN 状态交集实际可跳过 token，禁止二者相加造成重复计数。
- 本地 `unittest discover` 为 191 项通过、23 项可选依赖跳过；submission
  preflight 8/8。当前文档指定实例 `ssh-a2d0a302` 已失效，2026-07-19 快速门禁
  返回 `Connection closed by UNKNOWN port 65535`，因此 TP4 correctness、235K
  cold/warm 与性能 A/B 仍未执行。本候选不得标记为 qualified，也不得仅凭离线
  模拟宣称得分提升。完整设计见
  `docs/experiments/M1_31_STABLE_GDN_PREFIX_STATE_20260719.md`。

## 2026-07-17 当前 RC 状态

- 私有 RC `rc/submission-preflight-20260717@215ca46` 已通过本地 174 项和
  远端 CoreX 172 项单测，submission preflight 7/7；远端真实退出码均为 0。
  RC 不改变模型 runtime 或 YAML 语义，证据见
  `docs/experiments/M1_23_SUBMISSION_RC_AND_LONG_STABILITY_20260717.md`。
- 三轮 TP4 长稳定性矩阵 18/18 通过：32K/65K/131K/235K 均为零缓存 cold、
  完整 warm 命中且 8-token hash 一致；三次 1,000-token 短上下文 decode hash
  一致，服务日志无 fatal/OOM/Gloo/shape/worker-loss。
- 更强的 235K+1,000-token cold/warm 均完成且服务健康，但输出在生成 token 97
  分叉；32K/65K/131K 的 256-token 对照均一致。下一 runtime 门禁 M1-24 必须
  保留 cold 最后 scheduler chunk 的 replay 边界并复测，不得把当前结果误报为
  完整长生成 bit-exact。
- `ssh-913ffbfe` 未向容器透传 GPU PCI/字符设备，不能作为第二实验机；该问题
  不是安装 Torch 可以修复的。
- 当前 RC 由私有 ModelHub `main@46e8a12` 演进，提交时以通过本节门禁的 main
  最新 head 为准；M1-12 direct prefix
  fast-forward、M1-14 MRoPE chunk/prefix 对齐、E-MOE-20 direct routed decode
  和 E-GDN-14 packed decode 均已完成 TP4 资格并进入 main。
- 当前生产实例使用 262144 context，GPU/CPU KV blocks 为 `16878/6553`，
  health/models HTTP 200，短 decode Output TPS P10 为 `21.506`。本地评分回放
  aggregate proxy 为 `6696.0`，仍需新的官方 881 结果验证真实得分。
- M1-22 three-bucket cuBLAS MoE 因启动后 GPU KV blocks 降至 `9751`、只能容纳
  `156016` tokens 而关闭；该实验不进入 main/YAML，也不得继续 bucket/workspace
  参数扫描。
- M1-21 cache trace 仅位于私有诊断分支，必须先取得真实 881 block 到达序列并让
  离线模拟超过预设收益门槛，才允许修改 cache retention。正式性能提交不能包含
  trace 开关。
- 提交前运行 `python tests/submission_preflight.py`。它验证固定 YAML 合同、离线
  wheel 哈希、Docker patch 入口、诊断开关隔离、LF 换行及源码语法。

下方按时间保留早期实验和当时的“下一步”记录；发生冲突时以本节和日期更新较新的
结论为准。

## 2026-07-16 真实评测与 MRoPE 致命错误

最新 881 请求评测只有 269 成功、612 失败，Output TPS P10 `4.03`、TTFT P90
`29.706s`、cache hit `42%`。Docker traceback 显示
`Qwen3_5InterleavedMRotaryEmbedding.forward` 收到 26540 个 positions，但当前
Q/K 仅对应一个小物理 chunk，最终触发 invalid view 并终止 async engine。这不是
普通图片 4xx，也不能用后续 612 个失败请求反推每个 API 类型的兼容率。

M1-14 修复 vendor vLLM 0.6.3 的 chunk/prefix MRoPE 对齐：完整 MRoPE map 保留
request-level delta，再精确切到 `[context_len:seq_len]`；partial/full prefix hit
同步裁剪三个 position axes，并在进入 GPU 前检查长度等于物理 input token 数。
本地 160 tests 通过、22 项环境跳过；第二实例真实 vendor 源码副本补丁和幂等
`py_compile` 通过。TP4 长图片 cold/warm 回归随后通过并已合入 main。详情见
`docs/experiments/M1_14_MROPE_CHUNK_ALIGNMENT_20260716.md`。

## 2026-07-16 当前主线与下一步

ModelHub 私有仓库的 E-MOE-20 资格基线为 `main@2d3a0e5`，当前生产主线已继续
合入 E-GDN-14 packed decode、Agent 请求兼容修复和资格文档。评测配置只在精确
`T=1/FP16/top_k=8/E=256/H=2048/I=128` 条件下启用
`BI100_MOE_COREX_DIRECT_ROUTED=1`，其他形状完整回退；GitHub 仍可匿名读取，
在仓库所有者手动改为 private 前禁止推送 GitHub。

E-MOE-20 的三组同 binary、同请求、仅切换开关的严格 TP4 配对结果如下：基线
Output TPS P10 为 `15.1293/15.0863/15.1491`，候选为
`20.0563/19.0185/20.0312`，相对提升 `32.57%/26.07%/32.23%`。候选 TTFT P90
为 `2.1337/2.2143/2.1733s`，请求成功率均为 100%，缓存命中均为 86.87% 左右。
这证明算法有显著端到端收益，但只有 2/3 次达到 20，暂时只能称为接近硬门槛，
不能称为稳定达标。

质量门禁已通过：full smoke `15/15`，Agent workload matrix `9/9`，持续 1,000
token 全部 finite 且输出 hash 与资格基线一致。99.5K cold/warm 为
`157.252/16.583s`、命中 `99,296` tokens；235K 为 `562.368/55.489s`、命中
`234,544` tokens，两组 cold/warm 输出完全一致。API 层同时修复 assistant
`tool_calls` 搭配 `content:null`，以及多条文本 system message 的顺序合并问题。

E-GDN-14 packed decode 随后完成生产资格。V2 删除 decayed-state 的冗余全量
写回，并把 normalized q/k 提升为 block 共享数据；GPU1-3 对当前生产边界达到
`4.71x-5.24x`，候选绝对延迟 `0.0367-0.0378 ms/layer`，1,000/1,000 步 finite
且误差分布未扩大。三组同 binary TP4 配对 Output TPS P10 从
`20.28395/20.22108/20.34747` 提升到 `21.95640/21.75182/21.93097`，即
`+8.25%/+7.57%/+7.78%`；success 100%，full smoke `15/15`、Agent `9/9`、
两次 1,000-token 历史 hash、99.5K/235K cold-warm 全部通过。提交配置已启用
`BI100_GDN_COREX_PACKED_DECODE=1`，代码默认仍为 `0` 并保留精确 shape guard。
资格历史保留在私有分支 `exp/E-GDN-14-production-integration`，winner 已合入
生产 `main`；禁止推送仍公开的 GitHub。

## 2026-07-15 TP4 候选栈更新

E-ATTN-04 以单个 CoreX kernel 将 paged FP16 K/V 精确 gather 到现有 FP32
attention 布局，后续 matmul/softmax 不变。GPU1-3 的 64K 完整 attention 提升
`1.365x-1.370x`，100K 提升 `2.018x-2.024x`，K/V/输出均逐位一致。按十个
full-attention 层投影，64K/100K 每 token 分别约节省 28.9/93.8 ms；该收益仅在
32K 以上长上下文 fallback 生效，尚未完成 TP4 服务 A/B。证据见
`docs/experiments/E_ATTN_04_COREX_PAGED_KV_GATHER_20260715.md`。

E-ATTN-05 进一步按上下文长度调整同一内核的 grid：96K 及以下使用 256 blocks，
99.5K/100K 保留原大 grid。三卡交叉验证对 64K/90K/96K 再降低约 5%-8%，生产
分派探针达到 64K `1.503x`、96K `2.021x`、100K `2.016x`，全程逐位一致。
E-ATTN-05 已取代 E-ATTN-04 成为长上下文候选，证据见
`docs/experiments/E_ATTN_05_PAGED_GATHER_GRID_20260715.md`。

E-ATTN-06 direct split-K 原型比 E-ATTN-05 在 64K/100K 分别再快
`1.635x/1.362x`，但改变了归约顺序。100K 的 100 组压力测试只有 83 组通过
`1e-3` 容差，最差绝对误差 0.05937，因此按 exact-contract 拒绝，未进入生产
代码或 TP4 候选栈。证据见
`docs/experiments/E_ATTN_06_DIRECT_PAGED_DECODE_20260715.md`。

E-ATTN-07 将广播 `matmul` 改写为 stride-zero GQA `bmm`，64K/100K 各 100
组输出均逐位一致，但完整路径仅提升 `0.25%/0.00%`，低于 5% 门槛；物理 repeat
版本更慢。因此拒绝，E-ATTN-05 保持不变。

E-MOE-11 将逐位一致的 routed `SiluAndMul` 与 E-MOE-10 CoreX 精确归约组合。
GPU1-3 真实 TP4 rank-local shape 的完整 routed decode 路径分别提升
`1.0993x/1.0993x/1.0998x`，每卡 1,000 组随机输入均逐位一致。组合预计每 token
节省约 1.83 ms，使当前 TP4 候选栈的未资格化投影从约 4.4 ms/token 增至约
5.1 ms/token，对应 13.3-13.5 TPS 基线约 14.3-14.5 TPS。服务 A/B 尚未完成。

E-MOE-12 将 T=1 top-8 的 W13/W2 两次高级索引替换为一个 `__half2` CoreX
gather。GPU1-3 完整路由链路稳定提升 `1.2606x-1.2615x`，固定路由全路径约
`1.329x`，每卡 1,000 组随机路由均逐位一致；生产模型方法探针为
`0.4570 -> 0.3619 ms`（`1.2630x`）。按 40 个 MoE 层新增约 3.76 ms/token，
使短上下文候选栈未资格化投影增至约 8.9 ms/token，即约 15.1-15.3 TPS。
证据见 `docs/experiments/E_MOE_12_FUSED_WEIGHT_GATHER_20260715.md`。

E-MOE-13 进一步把 E-MOE-12 的 half2 一维 copy 改为按 16 个专家权重段调度的
128-bit 二维 copy，消除了内层逐元素整数除法。相对 E-MOE-12，GPU1-3 完整路由
链路再提升 `1.090x/1.094x/1.095x`，每卡 1,000 组仍逐位一致；生产方法相对
原生索引达到 `1.396x`。该增量约 1.25 ms/token，短上下文候选栈保守投影更新为
约 10.1 ms/token，即约 15.4-15.6 TPS。E-MOE-13 取代 E-MOE-12 的 copy kernel，
但沿用同一环境开关和回退。证据见
`docs/experiments/E_MOE_13_VECTOR_WEIGHT_GATHER_20260715.md`。

E-MOE-14 尝试先以 FP16 top-k、再只转换选中的 8 个 logits。3,003 组随机和
并列边界用例的专家索引、权重、输出均逐位一致，但完整 E-MOE-13 路径仅提升
`1.0104x`（约 0.14 ms/token），低于 5% 门槛，因此拒绝且未改生产代码。

E-MOE-15 测试 W13/W2 单次 packed 分配和 1/2/4/8-way copy 展开。所有结果仍
逐位一致，但最佳完整路由结果仅为 E-MOE-13 的 `0.9998x`，因此拒绝且未改
生产代码。

E-MOE-16 重新分解 E-MOE-13 后的 `0.3304 ms/layer`：route 17.3%、gather
19.2%、W13 linear 39.4%、W2 bmm 16.3%，activation/reduce 合计仅 5.4%。
组合调度开销只有约 0.0075 ms，因此后续必须直接优化 W13 或同时覆盖两次
expert GEMM，不能继续包装现有算子。

E-MOE-17 扫描连续 `2048x2048` W13 的 CoreX cuBLAS `GemmEx` 算法。裸 W13
最好 `1.044x`，但完整路由只有 `1.0011x`；`Hgemm` 更慢且不精确。该方向
拒绝，后续 W13 必须是实际融合或 shape-specific kernel，不能继续包装 cuBLAS。

E-MOE-18 的 shape-specific W13 matvec 裸算子达到 `5.84x`、完整固定路径
`1.746x`，但所有 FP32/Kahan 归约都产生至少 `3.05e-5` 的最终输出差异；CoreX
FP64 版本结果无效。该误差与已导致 1,000-token hash 分叉的 E-MOE-04 同量级，
因此按正确性门槛拒绝，不能放宽容差。

E-GDN-08 按 checkpoint 真实 TP4 八 value-head 形状测得完整 rank-local decode
为 `0.5646 ms/layer`；q/k prep 和输入投影各占约 27%/26%。该画像同时发现旧
E-GDN-02/06/07 沿用了 12 local heads 的过期注释。E-GDN-09 在真实八头下重审
后，normalize-before-expand 与 exact recurrent 分别只有 `0.9850x/0.9855x`，
仍为负收益并保持拒绝。

E-GDN-10 将 decode beta sigmoid 和最终 recurrent decay factor 合为一个 CoreX
kernel。按 checkpoint 的 BF16 权重经运行时下采样为 FP16 的真实 dtype 重跑后，
生产 merged-view 在 1,000 组随机输入上 beta/decay 均逐位一致，
`0.06270 -> 0.00712 ms`（`8.80x`），约投影节省 `1.67 ms/token`。短上下文候选栈累计投影更新为约
`11.8 ms/token`，即约 `15.8-16.1 TPS`；仍需 TP4 服务 A/B。

E-GDN-11 扫描两个真实 rank-local GDN 投影的全部可用 cuBLAS 模式。最佳 exact
输入/输出模式仅 `1.0034x/1.0085x`，唯一不精确 Hgemm 还明显更慢，因此按 5%
门槛拒绝，生产代码和候选栈不变。

E-GDN-12 保留 PyTorch FP16 q/k L2 归约，只融合归一化后 4→8 head 映射、
FP32 转换和 query scale。真实 convolution split-view 下 q/k 都连续，1,000 步
q/k、输出和 recurrent state 全部逐位一致；完整 prep+recurrent 为
`0.19362 -> 0.16757 ms`（`1.155x`），约投影节省 `0.78 ms/token`。短上下文
候选栈累计投影更新为约 `12.6 ms/token`，即约 `16.0-16.3 TPS`。

E-GDN-13 将 E-GDN-10/12 放回同一个完整 rank-local 层复测，E-GDN-03/05-only
参考为 `0.54915 ms`，当前 E-GDN-03/05/10/12 为 `0.44950 ms`（`1.222x`），
输出、conv state 和 temporal state 均逐位一致。组合净省约 `2.99 ms/token`，
因此短上下文候选栈统一改用约 `13.1 ms/token`、`16.1-16.4 TPS` 的未资格化投影。

ModelHub 的 `6e0e66c` 提交曾在 `init_device` 静默约 30 分钟后报告 Gloo peer
reset。相同 commit 已在四卡全绿的 `a163074c` 完成 Docker 等价构建、vLLM 等价
NCCL+Gloo 建组、TP4/262144 服务启动、full smoke 15/15 和 1,000-token hash
门禁；hash 与资格基线完全一致，日志无 Gloo/OOM/worker loss。该评测错误应视为
某个 evaluator rank/GPU 卡住并被平台清理后的次生错误，不回退模型代码或 256K。
下一提交镜像启用 `BI100_EXECUTOR_STARTUP_DEBUG=1` 记录逐 rank 启动阶段。详见
`docs/incidents/MODELHUB_GLOO_RESET_20260715.md`。

健康实例进一步完成完整候选资格：99.5K cold/warm 为 `154.536/18.247s`、命中
`99,296` tokens；235K 为 `554.632/56.838s`、命中 `234,544` tokens，两组
cold/warm hash 均与历史资格值完全一致。8 请求固定单并发样本为 Output TPS P10
`15.5445`、TTFT P90 `1.795s`、成功率 100%、缓存命中 87.45%。因此正确性、
长上下文、TTFT、成功率和缓存门槛已通过，但到 20 TPS 仍需约 28.7% 相对提升。
短 prompt 本地 weighted `1417.9` 不可与官方长输入加权 8000 直接比较。完整证据见
`docs/experiments/TP4_CANDIDATE_24C75C7_QUALIFICATION_20260715.md`。

新实例 `ssh-a2d0a302.default.gpu.phanthy.com` 的 GPU0 仍为 257 MiB、100% 利用率且
无容器内可见进程；GPU1-3 CUDA 探针正常。TP4 服务资格验证仍需宿主侧复位或健康
四卡实例。证据见 `docs/experiments/E_MOE_11_COMBINED_EXACT_TAIL_20260715.md`。

13:55 再次执行独立四卡 preflight（每卡 12 秒硬超时、256 方阵乘）：GPU0 返回
`124 timeout`，没有完成设备/显存读取；GPU1-3 均通过，free/total 各为
`34057748480/34057748480` bytes，matmul checksum 均为 `16777216.0`。最新远端
证据为 `/root/competition/preflight_a2_20260715.json`，TP4 阻塞未解除。

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

T5 fixed-contract qualification 随后连续两次无 override 启动成功。正式验证目录：

```text
/root/qwen36-bi100-submission/bench_runs/20260711_194713_e00de43_T5_fixed_qualification
startup: 80 s, GPU blocks=18275, num_gpu_blocks_override=None
full smoke: 14/14 PASS
benchmark: requests=8, workers=1, success_rate=1.0
TTFT P90=2.5836 s, Output TPS P10=5.7213
Input TPS=231.4230, Cache TPS=201.3813, Cache hit=0.870187
Weighted proxy score=856.6219
```

此前 layer 6 synthetic-profile non-finite 在两次复测中均未重现，暂列稳定性观察项；
不得因此恢复 GPU-block override。T5 已完成，当前进入 T6 长 prefill/GDN 剖析。

T6 profiling 的 8K/16K 请求显示 routed MoE 为最大热点，GDN prefill 次之。
`3e6df10` 尝试以 Neumann doubling 向量化 GDN chunk inverse，小规模 parity 通过，
但真实 startup profile 在 layer 0 出现 14.7%-46.9% non-finite。该变化已由
`865ec8a` 撤回并按重大问题门禁暂停；详见
`docs/experiments/T6_GDN_INVERSE_20260712.md`。

T7 随后转向 profile 中占比最高的 routed MoE。`9cb31f3` 将每 expert 的全路由
mask/nonzero 扫描改为一次 stable sort+bincount，保持 expert GEMM 和 decode
快路径不变。MoE/GDN parity、硬件、固定启动和 full smoke 全绿。三组严格 seeded
A/B 的 weighted 平均提升 7.67%，TTFT P90 平均改善 16.01%，Input/Cache TPS
平均提升 7.96%，因此保留。详见 `docs/experiments/T7_MOE_GROUPING_20260712.md`。

T8 审计发现现有 GDN prefix-state cache 的重大边界错误：3678-token prompt 的
state 在处理完整 prompt 后保存，却使用只覆盖 3664 tokens（229×16）的 physical
block key。缓存命中后恢复了超前 14 tokens 的 state 并再次处理这 14 tokens。初始
实验由 `42fc9b7` 撤回，避免把错误实现留在稳定基线。

后续 `c05fd52`/`9f95cb5` 已实现 scheduler checkpoint mirror 与 GDN 精确边界
capture，小规模和 3663-token case 通过，但原始 8712-token payload 在 cached=8176
时仍由 18/stop 变为 32/length。四 rank DEBUG 均确认 GDN state 命中，因此剩余差异
来自 full attention：首次请求在 8192 current chunk 内计算末尾 16 tokens，缓存请求
按 8176 context + 16 current 计算，online-softmax 归约分区不同。两提交已由
`d837caa`/`4161d3f` 撤回，恢复 T7 winner 后重新设计完整方案。

最终 T8 由 `0e52374`/`0ec0607` 保证 scheduler 与 GDN state 使用同一严格边界，
`b22fd8f` 将 PyTorch full attention 在该边界分段，`a63a1ef` 修复 fp32 query 原地
缩放。原始 8712-token payload 在 `cached=0/8176` 时响应完全一致，耗时从
19.53s 降至 6.62s。Full smoke 14/14、attention/GDN/MoE parity、四卡 CUDA/NCCL、
对齐/未对齐交错前缀和 17 项 LRU 驱逐均通过；GPU 显存不增长，后续三轮基准 RSS
仅增加 14.9MiB。T8 保留，证据见
`docs/experiments/T8_GDN_PREFIX_BOUNDARY_ISSUE_20260712.md`。

T9 最终提交 `3f3f021` 修复分阶段 GDN checkpoint 的 usage 低报：旧实现只记录
首次 8176 tokens，现按每轮 checkpoint 相对已计算位置累计真实跳过量，不改变模型
张量路径。8712-token 样本报告 `cached=8688`，耗时 19.69s→6.66s；99500-token
样本报告 `cached=99296`，耗时 158.63s→19.67s，且两次完整响应一致。

最终固定命令重启、冷启动确定性、四卡 CUDA/NCCL、100/100 package tests 和 full
smoke 14/14 全部通过，日志无 non-finite/OOM/CUDA error。`computility-run.yaml`
SHA256 仍为 `5f07f437...e517c0f`。宿主机没有 Docker CLI，无法执行 image build，
已验证离线 wheel hash、patch 输入、py_compile 和导入门禁。详见
`docs/experiments/T9_QUALIFICATION_20260712.md`。

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

以下性能矩阵是早期草案，已因主办方固定命令和后续 256K 资格要求作废，禁止执行：

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

当前本地主机没有 Docker CLI，不能在此重建镜像；ModelHub 构建和同基础镜像的远端
`patch_ops.sh`/TP4 服务资格已经通过。最终提交仍应由平台重新构建，并检查完整日志：

```bash
docker build -t qwen36-bi100-dev . 2>&1 | tee build.log
grep -E "VLLM_ROOT|TRANSFORMERS_ROOT" build.log
```

最终 Docker build 不应出现未解释的 `anchor not found`。

## 2026-07-15 E-MOE-02 进展

- 新 4 卡实例 `ssh-a2d0a302.default.gpu.phanthy.com` 已通过 Torch、TP4 与
  262,144-token 服务启动，最终候选 PID/PGID 为 `18909/18909`。
- `f11c6f9` 将 MoE 路由从 256 路完整 softmax 后取 top-8，改为先取
  top-8 logits 再仅对 8 路 softmax；四卡真实形状微基准均 bit-exact。
- 三组交错固定请求配对的 Output TPS P10 均提升，中位提升 `4.33%`；ITL
  P90 中位下降 `4.99%`，固定短测加权值中位提升 `5.00%`。
- full smoke 为 `15/15`。235K 冷/热请求耗时 `519.855s/48.385s`，热命中
  `234,544` tokens，输出 SHA256 与既有 256K 资格化结果一致。
- `min_tokens=max_tokens=1000` 的持续解码门槛耗时 `77.831s`，返回 1,000
  completion tokens、`finish_reason=length`，完成后服务 health 仍为 200。
- GitHub 分支 `exp/E-MOE-02-decode-primitives` 已推送。ModelHub token 仍被
  Gitea 拒绝，在认证恢复前不要反复猜用户名或覆盖官方远端。
- 完整证据见 `docs/experiments/E_MOE_02_DECODE_ROUTING_20260715.md`。当前
  Output TPS 约 `13.45-13.73`，仍低于竞赛目标 20，下一步继续优化 MoE
  专家计算或 TP collective，不改固定 evaluator 参数。

## 2026-07-15 E-MOE-03 进展

- `7a68a94` 将 256-output router 和 1-output shared-expert gate 合并为
  257-row replicated linear，并用显式分片 loader 保持 checkpoint 兼容。
- 四卡 T=1 微基准 bit-exact，融合速度为 `1.744x-1.812x`；T=64 为
  `1.829x-1.837x`。
- 三组交错固定请求配对的 Output TPS P10 均提升，中位 `+5.78%`；固定短测
  加权值中位 `+3.17%`。full smoke 为 `15/15`。
- 235K 冷/热耗时 `503.270s/43.172s`，热命中 `234,544` tokens，输出哈希
  保持不变。强制 1,000-token 解码耗时 `76.278s`，哈希也与 E-MOE-02 一致。
- 最终候选服务 PID/PGID `35306/35306`，262,144 context，GPU/CPU blocks
  `16878/6553`，health 200，无 loader、fatal、OOM 或 worker loss。
- 完整证据见 `docs/experiments/E_MOE_03_ROUTER_SHARED_GATE_20260715.md`。
  当前 Output TPS 约 `13.26-13.54`，仍需继续优化 routed expert 和 TP
  collective 才能接近竞赛门槛 20。

## 2026-07-15 E-MOE-04 结论

- 尝试将 T=1 routed expert 的逐元素权重乘法和 top-k sum 合并为一次
  `torch.matmul`。四卡真实 shape 微基准中，归约本身提升 `2.85x-3.24x`，完整
  routed path 提升 `1.053x-1.087x`。
- 该实现不是 bit-exact，最大绝对差 `6.1035e-5`。CoreX 单测/静态测试通过，
  256K 服务正常启动，full smoke `15/15`，日志无 fatal/OOM/non-finite/worker loss。
- 强制 1,000-token 请求耗时 `76.693s`，但输出哈希从 E-MOE-03 的
  `1766c3...` 变为 `be4ee3...`；请求 token 数和 reasoning 全部一致，正文发生
  token 分叉。因此按正确性门禁拒绝，不运行性能 A/B 或长上下文门禁。
- 远端运行时已恢复 E-MOE-03 `2103876`；`integration/perf-winners` 不变。
- 证据见 `docs/experiments/E_MOE_04_WEIGHTED_REDUCE_20260715.md`。

## 2026-07-15 E-MOE-05 结论

- 复用 shared expert 已使用的 CoreX/vLLM `SiluAndMul` 测试 routed expert
  激活。GPU1-3 激活结果和完整 routed output 均 bit-exact，max abs 为 0。
- 融合激活单项提升 `1.64x-1.69x`，但完整 routed path 仅提升
  `1.030x-1.032x`，低于 5% 集成门槛，因此不修改模型、不做服务 A/B。
- GPU0 在测试初始化时卡住。清理仍可见的服务 PGID `42435` 后，显存从
  18,164 MiB 降至 257 MiB，但 GPU0 仍为 100% util，最小 Torch preflight
  继续超时。设备 reset 被剩余宿主 PID `7093` 拒绝，而容器内无对应进程；
  继续四卡测试前需要平台实例级重启或宿主侧 reset。
- 证据见 `docs/experiments/E_MOE_05_FUSED_ACTIVATION_20260715.md`；E-MOE-03
  仍是当前 qualified model winner。

## 2026-07-15 E-GRAPH-01 结论

- vendor `vllm/engine/arg_utils.py` 将 `enforce_eager=True` 硬编码，导致固定
  命令的 `--max-seq-len-to-capture 32768` 实际无效，同时关闭 async output。
- 实验分支 `exp/E-GRAPH-01-cudagraph-probe`、提交 `97440b0` 已加入 fail-closed
  幂等 patch，以及单卡 MoE/GDN state graph 和 TP=4 IPC collective graph 门禁。
- 本地 patch 单测 `2/2`、Python/shell/diff 检查通过；GPU1 上 MoE、GDN output
  和 mutating state 的 graph replay 均 finite、bit-exact、max abs 0。
- 性能门禁失败：单个 MoE 子图 `0.8686x`，40 层重复子图 `0.9197x`，graph
  分别慢 13.1% 和 8.0%。因此跳过 TP collective 和完整服务，patch 不合入
  integration/qualified runtime。
- 详细流程见 `docs/experiments/E_GRAPH_01_CUDAGRAPH_PROBE_20260715.md`。

## 2026-07-15 后续微基准与 CoreX 扩展结论

- E-MOE-06 routed/shared 双流重叠保持 bit-exact，但完整 MoE block 只有
  `0.711x-0.721x`，因 stream 同步和资源争用淘汰。
- E-GDN-02 将 q/k normalization 前移到 3 倍 head 展开之前，output/state
  bit-exact，但完整 recurrent step 仅 `1.0235x`，低于 5% 集成门槛。
- E-CAP-02 重新确认 vendor FusedMoE 三个 ixformer 符号和 `_moe_C` op 均缺失；
  `/usr/local/corex/bin/nvcc` 只是版本占位脚本。CoreX Clang 16 的 `bi/ivcore10`
  后端可以编译、加载并执行 ABI-0 Torch CUDA extension，standalone 和 Python
  extension smoke 均通过。这一能力结论保留在 integration。
- E-GDN-07 自定义 recurrent kernel 的单步、固定 1,000 步和随机 1,000 步
  数值门槛通过，candidate 绝对延迟稳定约 `0.049-0.052 ms`；但独立串行复测
  仅 `1.280x/1.314x`，低于计划要求的 `1.5x`，生产接入提交不合入。
- E-GDN-06+07 进一步融合 q/k normalization、head expansion 和 recurrent。
  FP32/half normalization 版本达到 `2.179x/2.318x`，但随机序列 state/output
  close 失败；保留完整 PyTorch FP16 normalization 的版本数值通过但为
  `0.992x`。不得放宽容差或重复尝试这些归约路径。
- E-MOE-07 将 T=1 路径分解为 route `0.057 ms`、selected-weight gather
  `0.181 ms`、预取权重后的专家计算 `0.241 ms`，完整路径为 `0.507 ms`；
  gather 单项占 35.75%，因此进入无拷贝门禁。CoreX 扩展直接对原始专家权重
  执行 cuBLAS pointer-batched GEMV，FP32 累加保持 bit-exact，但完整路径为
  `0.718 ms`、仅 `0.704x`；half 累加为 `0.628x` 且不精确。该方向淘汰，
  不得用 cuBLAS 小批量 GEMV 替换当前 flattened W13 + gathered W2 路径。
- E-ATTN-01 因沿用源码陈旧注释中的 `5120/24/4` 形状而撤回；真实 checkpoint
  是 hidden `2048`、Q heads `16`、KV heads `2`，固定 TP4 使用 sharded QG 和
  replicated K/V。纠正提交 `d0dded9` 明确禁止合入旧候选。
- E-ATTN-02 在真实 TP4 shape 下合并 replicated K/V，实际 vLLM layer 输出
  bit-exact；E-ATTN-03 进一步把本 rank QG 分片与全量 K/V 打包，T=1/T=64
  实际层速度为 `1.905x/1.423x`，rank=2 checkpoint 切片和三段输出均 exact。
  候选在 `exp/E-ATTN-03-packed-qgkv` 的 `5bebe8c`，预计全模型仅节省约
  `0.24 ms/token`，TP4 服务收益待 GPU0 恢复后验证，未合入 integration。
- E-MOE-08/09 分别把 shared gate/up 和 shared down 当作第 257 个 expert。
  裸 W13 probe 曾显示 `1.186x` 假收益，但实际 vLLM layer oracle 为 `0.994x`；
  W2 完整尾部为 `0.903x` 且 shared intermediate 非 exact。两条路径均淘汰，
  后续 shared fusion 必须直接使用实际模型层作为基线。
- 当前 `integration/perf-winners` 只包含 benchmark、能力证据和拒绝 decision；
  模型实现仍为 qualified E-MOE-03/E-GDN-01。新实例 GPU0 仍无可见进程却
  100% util，GPU1-3 可做单卡实验；恢复 TP4 前仍需平台侧处理 GPU0。
- E-GDN-03 将 decode 的 state 拼接/回写、4-tap depthwise conv 和 SiLU
  融为一个 CoreX kernel。纠正 checkpoint 真实 TP4 rank shape 为
  `B=1,C=2048,K=4` 后，GPU1-3 均保持 output/state bit-exact，完整边界
  提升 `7.30x-7.39x`，预计 30 个 GDN 层合计节省约 `1.35 ms/token`。
  TP2 原参数和 `cpu_offload_gb=8` 均在权重加载时 OOM，故该候选保留在
  `exp/E-GDN-03-fused-causal-conv`，等待健康 TP4 的服务哈希与性能门禁，
  暂不合入 integration。
- E-GDN-04 复用 ixformer RMSNorm 的探针中，FP32 state 被 operator 明确拒绝；
  预先降为 FP16 可让完整 `norm+gate+out_proj` 提升 `1.785x`，但 tail
  `max_abs=9.77e-4` 且不满足 close，因此按正确性门禁淘汰。不得通过降低
  GDN state dtype 接入该 operator。
- E-GDN-05 保留原 PyTorch FP32 inverse reduction，仅融合后续 scale、weight、
  SiLU gate、乘法与 FP16 输出。GPU1-3 的 1,000 组随机 norm/实际 linear tail
  均 bit-exact，完整 tail 提升 `1.979x-2.024x`，预计 30 层节省约
  `1.63 ms/token`。生产扩展已独立编译和调用通过，但与 E-GDN-03 一样等待
  健康 TP4 服务门禁，暂不合入 integration。
- E-MOE-10 显式复现 FP16 product 舍入后 FP32 累加，修复 E-MOE-04 GEMV
  的输出漂移。GPU1-3 均通过 1,000/1,000 随机 exact，完整 routed path
  稳定提升约 `1.059x-1.064x`，预计 40 层节省约 `1.17 ms/token`。生产编译
  门禁还发现精简 kernel 会被编译器重排而变成 0/1,000 exact；最终保留
  runtime Mode dispatch 后恢复 1,000/1,000。候选等待 TP4，不合入 integration。
- E-MOE-19 将 shared-expert sigmoid gate、FP16 乘法和 routed add 融为一个
  CoreX kernel。以 `volatile half` 强制保留 PyTorch 的中间 FP16 舍入后，
  1,000/1,000 随机、全部 63,488 个有限 FP16 gate 位型及 100/100 完整 MoE
  边界均逐位一致。裸尾部为 `2.502x`，但真实 TP4 rank-local 完整 MoE 仅
  `1.0164x`，40 层预计只省 `0.394 ms/token`，低于 5% 集成门槛，因此拒绝，
  不修改当前候选运行时。证据见
  `docs/experiments/E_MOE_19_SHARED_COMBINE_20260715.md`。

## 2026-07-16 M1-14 MRoPE chunk 对齐与长上下文 decode 结论

- 881 请求评测中的 engine-fatal 并非模型路径问题：MRoPE 位置张量保留了完整
  `26,540` token，而本轮物理 query 只有 `64` token，最终在 rotary reshape
  处触发 `shape '[26540, -1, 256]'`。
- `fix/M1-14-mrope-chunk-alignment` 在完整 MRoPE 映射上按当前 prefill chunk
  精确切片，并在 partial/full prefix hit 后同步裁剪。160 个本地测试通过；真实
  CoreX vendor 函数验证三轴从 `26,540` 正确裁到 `64/64/64`。
- 独立生产扫测在服务全程健康时得到 32K/64K/131K/235K 的 64-token decode
  TPS 分别为 `10.188/7.024/5.120/3.698`。这复现了评测 P10 `4.03`，说明
  MRoPE 修复后下一个高收益方向是 `>32768` 直接 paged decode，而非继续调整
  YAML/scheduler 参数。
- TP4 唯一首块 240,132-token 请求完成真正 `cached=0` cold 和
  `cached=240128` warm，输出哈希一致、HTTP/health 均为 200，日志无 shape
  invalid/fatal/OOM/Gloo/worker loss。M1-14 已通过合并门禁。
- 嵌套请求从 180,096-token partial hit 扩展到 240K 时曾与后续 full hit 输出
  分叉，但两个 full hit 相互一致，唯一零缓存 cold/warm 也一致；该问题作为独立
  prefix 分段数值/语义问题保留，不能再归因于 MRoPE shape 修复。

## 2026-07-16 M1-15/M1-16 长上下文 attention 结论

- M1-15 shared-memory tiled gather 在独立 native 对比中保持完全 exact 且看似
  `1.96x-2.51x`，但与当前 E-ATTN-05 scalar CoreX 生产实现严格 A/B 后，
  32K/64K/100K/235K 增量只有 `+0.6%/-0.1%/+0.03%/+0.1%`。该路线按 5%
  门槛拒绝，不合入、不继续调 tile/grid；详见私有分支
  `exp/M1-15-tiled-paged-gather`。
- M1-16 保留 exact K、PyTorch FP32 QK 和全局 softmax，仅让 PV 直接读取
  paged V。K/logits/weights 全 exact，100/100 输出满足 `1e-3`，worst abs
  `1.5259e-5`，证明 E-ATTN-06 的 `0.05937` 主要来自 QK/分区 softmax；但
  64K/100K 只有 `0.800x/0.830x`，故拒绝。详见私有分支
  `exp/M1-16-exact-qk-direct-pv`。
- CoreX WMMA 仅支持 FP16 matrix input + FP32 accumulator，不能保留当前
  `query.float() @ key.float()` 的 FP32 输入语义。下一 direct kernel 不能依赖
  WMMA 冒充 exact QK，也不能物化全局 weights；必须融合直接分页读取与稳定的
  online softmax，并通过明确的模型级质量门禁。

## 2026-07-16 M1-17 warp-reduced direct decode 结论

- M1-17 用 BI100 64-lane warp 将每个 token 的 256 维串行 QK 改为每 lane
  4 维加 shuffle tree。64K/100K 相对 E-ATTN-05 达到 `1.720x/1.547x`，说明
  direct paged decode 存在性能空间。
- 数值门禁未通过：100K worst abs 虽从 E-ATTN-06 的 `0.05937` 降到
  `0.03424`，但 `1e-3` close 只有 65/100；64K 也只有 75/100。改变 reduction
  order 不能可靠逼近 authoritative FP32 matmul，因此拒绝，不做第二 seed 或模型接入。
- 下一步只做 M1-16 组件级 profile，量化 exact K gather、FP32 QK、softmax、
  contiguous PV 与 paged PV。只有“不物化 global weights”的乐观上限仍能比
  E-ATTN-05 提升超过 5%，才实现 M1-18 fused exact score-to-paged-V；否则关闭该线。

## 2026-07-16 M1-18 exact-QK 组件剖析结论

- 第二实例 GPU1 的 7-trial 同步测量显示，65K/100K 的 exact-QK/direct-PV
  组件和为 `8.9086/11.0811 ms`，而 E-ATTN-05 为 `7.0830/9.2203 ms`，即慢
  `25.77%/20.18%`。结果与 M1-16 端到端时间在 0.6% 内一致。
- K bit-exact，contiguous PV exact，direct paged PV worst abs `7.629e-06`；失败
  原因是成本而非正确性。即使取消 global weights 写回，也没有超过 5% 的理论空间。
- M1-18 按前置门禁拒绝，不实现、不调 kernel。当前 long-context direct-decode
  路线关闭，下一步必须回到 M1-11 全模型 profile 选择其他评分热点。

## 2026-07-16 M1-19 effective-tile MoE 结论

- 真实 7,800-token、40-layer route trace 的 16-row expert-tail padding 只有
  `1.02897x-1.03282x`，证明 vLLM/MegaBlocks 风格 tile map 能消除旧 hybrid
  的 1.77x 重尾工作量。
- 固定 BI100 WMMA `16x32x32` 的 W13+activation 原型在三层达到
  `1.702x/1.705x/1.733x`，workspace 约 17.0 MB，padding exact zero。
- 正确 vendor shared-memory 映射及 FP16 gate/up round-trip 后，三层仍稳定为
  worst abs `0.0078125`、mean abs 约 `4.61e-5`，超过 `1e-3/1e-5` 门禁。
  因此拒绝，不实现 W2、不调 WMMA tile、不做模型接入。

## 2026-07-16 M1-20 当前评分组成

- 当前 TP4 main 的 4K/7.8K/16K dataset-shaped 三组 cold/warm 回放 18/18
  成功，Output TPS P10 `21.506`；服务 PID 23178 和 health/models 200 保持不变，
  新日志无 fatal/OOM/Gloo/MRoPE/worker loss。
- Cold Input TPS P10/aggregate 为 `587.5/719.4`，warm Cache TPS P10/aggregate
  为 `3271.7/7716.5`。balanced cold/warm 的 TTFT P90 为 `21.454s`，cache hit
  为 `49.93%`。
- 聚合加权代理 `6696.0`，距 8000 约 1304 分；Output/Input/Cache 贡献约
  `361/2013/4322`。固定其他项时需要约 `+466 Input TPS` 或 `+2329 Cache TPS`。
- 下一项只审计 prefix-cache retention/admission 是否能在固定 KV 容量内保留短且
  高频共享前缀；先做 trace/simulation，不能修改 cache usage 记账或以 YAML 调参替代。

## 2026-07-16 M1-21 cache trace 诊断分支

- 现有公开材料只有理论 token-weighted prefix hit `65.6%`、会话分布和历史官方
  `42%`，没有 881 请求的 block identity/order，不能直接修改 LRU。
- 私有分支 `exp/M1-21-prefix-cache-trace@a84746c` 只输出 allocator 已计算的链式
  block hash 和必要容量元数据，不输出 token/消息/tool/原始 request id，也不改变
  allocator；离线模拟器比较现有 LRU 与一个固定 frequency-aware 策略。
- 8 项行为测试、60 项聚焦/静态门禁及真实 vendor 源副本的两次幂等 patch/compile
  通过。该分支默认打开 trace，只用于 diagnostic evaluator log，不能用于正式分数。
- 正式提交继续使用 production `main`，不要合入 trace patch 或其 Docker ENV。
