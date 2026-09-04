# Coding-agent 提示词：精简 M1-176 TP4 验证

继续当前私有实验分支上的 BI100/CoreX Qwen3.6-35B-A3B 工作。先阅读 M1-176
结果、2026-09-04 指标文档和 roadmap，继承已有成果，不要从头重做。

本轮目标是修复 reviewer 指出的门禁缺陷，并为 M1-162 FP16-QK 候选取得一份有效、
精简的完整模型 TP4 性能结果。正确性硬要求不降低，但只运行与改动影响面有关的
检查。该候选只改变 Attention 算子，不要默认运行 cache、完整协议和完整能力矩阵。

## 必须修复的问题

1. 修复 L3 qualifier 对 distribution evidence 的 fail-open。严格校验允许的状态、
   source/runtime/workload 身份、targets、完整样本总体和 decision 内容。未知状态、
   空 A/A 或未与 candidate 绑定的 evidence 必须是 `invalid`，不得返回 `pass`。
2. generic validity evidence 应校验非空身份、有限 timing sample 和一致的请求计数。
   保持 schema 简单，不要增加哈希、权限检查或新的 attestation 层。
3. capability gate 只有被实际调用时才校验全部 strata；每个 stratum 必须有样本数、
   配对结果和有效统计，空对象不能通过。本轮短算子实验不要运行完整 capability。
4. 负的性能增益是合法观测，应归为 candidate `fail`，不能抛异常后变成 `invalid`。
5. 精简性能实验。默认只跑一个 control 和一个 candidate；不要采集一个 estimator
   完全不用的 control B。如果确实补跑第二 control 或反序 pair，必须把它纳入点估计
   与置信分析。
6. 修正本轮发现的 reporting-only 指标公式，包括根据 first-token 到 last-token 时间
   计算 Output TPS 时的 token 数。通过单测明确 TTFT、TPOT 和 Output TPS 定义。

## 精简开发 runner

新增 `attention_operator` change scope 或等价的 focused runner。同机开发阶段不要求：

- clean Git tree 硬门禁；
- `/tmp` 路径或精确 0600/0700 权限；
- 逐文件 SHA-256、runtime tree hash 或 HMAC token identity；
- 每个 replay cell 前重复 GPU/NCCL preflight；
- 独立 operator replay 前后的完整 postflight；
- partial-prefix/cache branch 矩阵；
- 完整 API、tool、多模态或 capability 套件。

只记录 Git revision 与 dirty 摘要、runtime/compiler 版本、模型路径、启动命令、相关
环境变量、workload 计数、dispatch marker 和原始 timing。机器、runtime 和 GPU 状态
未变化时复用同一份四卡 preflight。完整模型服务仍必须 scoped TERM-first 清理，等待
现有 grace period，wait/reap 子进程，扫描 fatal/OOM/collective/worker-loss，并保证
退出后无服务和 GPU 进程残留。

## M1-176 证据处理

保留已有 L1 和 TP1-derived real-activation 结果，但将其称为 operator development
screen，不得称为真实 TP4 模型通过。head/KV mapping 未修改时，现有 focused unit
proof 足够；不要仅为满足旧 checklist 扩展成九格 capture/reassembly。

之前 L3 的 control 数据不能作为 candidate 证据：它使用了 reduced checkpoint，
control B 总体不完整，并且 candidate 从未启动。

## 完整模型 TP4 实验

四张 BI100 健康时，只使用固定完整模型：

`/root/public-storage/models/Qwen/Qwen3.6-35B-A3B`

运行最小配对筛选，两臂 runtime、请求语义和启动参数一致：

- control：关闭 fused FP16-QK selector；
- candidate：打开 selector，并要求实际 candidate dispatch marker；
- cold prompt 长度：16K、32K、64K；
- 每个长度 2～3 次，同一 arm 只启动一次服务；
- 使用足以测量 TTFT 和发现即时输出/运行错误的小型固定 greedy 输出预算；
- 记录完整 HTTP/SSE/usage/finish 状态和原始 TTFT；
- 不跑 partial-prefix，因为该候选不修改 cache 逻辑。

必须保持生产形状与语义：TP4、FP16、`max_model_len=262144`、block size 16，且不修改
官方请求语义。不得修改 `computility-run.yaml`、tokenizer、chat template、sampling
语义、cache policy 或无关 selector。

结果解释：

- aggregate TTFT gain 至少 5% 且各长度稳定，可进入 131K/235K；
- 2%～5%、明显顺序漂移或高噪声时，最多补一次反序 A/B 或一次可复用 A/A；
- 低于 2% 时停止该增量候选，除非 profile 证明它直接改善最终硬指标；
- 非有限值、错误 dispatch、畸形结果或 candidate 可归因 fatal 均为硬失败。

只有存活候选才运行 focused teacher-forced distribution 和长上下文确认。完整协议、
cache transparency、capability strata 与近 262K 容量留到最终集成门禁。

## 工程纪律

对修改的 validator/runner 运行 focused unit tests、Python/shell 语法检查和
`git diff --check`。不要在每个小修改后运行全仓测试；只在成熟 integration handoff
前运行一次。不要触碰五个 untracked M1-164 文件。所有工作留在私有实验分支，小步
commit，并只推送认证可用的私有远端。不得修改 `main`、正式 YAML、仓库可见性或
ModelHub 状态。

最终报告分别给出 operator numerics、kernel timing 和 TP4 service performance，包含
准确 commit、runtime/model identity、请求计数、各长度原始 TTFT、aggregate gain、
dispatch 证据、清理结果、已知限制和下一步被授权的阶段。
