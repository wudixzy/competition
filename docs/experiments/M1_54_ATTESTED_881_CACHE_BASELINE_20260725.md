# M1-54 881 请求缓存基线证据合同

## 结论

commit `382a28d` 修复了缓存离线模拟的证据绑定漏洞。此前
`scripts/analyze_prefix_cache_trace.py` 只要求 `run_id`、16 位 trace session
标识和少量聚合指标；一份来自其他源码、overlay、模型或请求顺序的旧指标文件，
理论上也可能参与 881 请求资格计算。

现在显式 `--qualification-trace` 必须同时提供
`bi100-prefix-cache-baseline-contract-v1`。合同严格绑定：

- 源码 revision、bare-host overlay SHA-256 和新 Harbor 基础镜像；
- 完整固定 TP4 命令、环境、`max_model_len=262144`、模型与 tokenizer；
- `fine32/direct/lru/fused-prefill=0` 控制策略；
- 881 请求 workload manifest、固定 revision/split 和请求顺序 SHA-256；
- trace v4 session、记录 SHA-256、日志文件顺序及每个日志的 SHA-256；
- metrics 原始制品 SHA-256、请求计数、TTFT、缓存率和公式输入；
- `Output P10 * 16.796 + Input TPS * 2.799 + Cache TPS * 0.56`
  的重新计算结果。

旧弱指标 JSON 仍可用于非资格诊断，但与 `--qualification-trace` 一起使用会直接
失败。历史 M1-41/M1-49 proxy 和未绑定的 ModelHub `main` 结果不能被转换成
资格证据。

## 三卡边界

M1-53 已证明 `ssh-73ca29ba` 仅有三张健康卡时不能启动该模型的 TP3 服务：
16 个全局 query heads 不能被 3 整除。本轮没有规避该约束，没有启动模型服务，
也没有新增性能或模型能力结论。

本轮重新连接时端点可达，但本地 GitHub 专用 SSH key 被实例拒绝，返回
`Permission denied (publickey,password)`。因此当前远端 GPU 状态未重新认证，
三卡组件状态仍只引用 M1-53 的已提交证据，不把认证失败解释成 GPU 故障。

## 新合同

### Workload manifest

`tests/build_prefix_cache_workload_manifest.py` 只接受完整、有序、单 session 的
881 条 trace v4 记录。它保存来源、许可证、固定 revision、split、选择和转换
规则，以及请求顺序和隐私安全 trace 记录摘要；不保存原始请求或模型输出。

示例：

```bash
python3 tests/build_prefix_cache_workload_manifest.py \
  "$RUN/server.log" \
  --name "restricted official 881 workload" \
  --author-or-org "competition operator" \
  --license "not stated; restricted evaluation data" \
  --revision "<fixed platform run or workload revision>" \
  --captured-at-utc "<fixed UTC timestamp>" \
  --split "all 881 requests in platform order" \
  --selection-rule "all requests in fixed platform order" \
  --transformation "privacy-safe BI100_CACHE_TRACE v4 only" \
  --out "$RUN/workload-manifest.json"
```

禁止把 revision 写成 `latest`，也不得把受限原始 881 请求提交到仓库。

### Baseline contract

`tests/build_prefix_cache_baseline_contract.py` 要求同轮生成的
`runtime_contract.json`、workload manifest、trace 日志和 metrics 原始制品。
两个 `--attest-*` 参数是操作员对“同一服务运行、同一请求顺序”的显式声明，
不能用推测值代替。

```bash
python3 tests/build_prefix_cache_baseline_contract.py \
  "$RUN/server.log" \
  --runtime-contract "$RUN/runtime_contract.json" \
  --workload-manifest "$RUN/workload-manifest.json" \
  --metrics-source "$RUN/platform-or-local-metrics.json" \
  --metrics-transformation "<exact field mapping and aggregation>" \
  --run-id "<fixed run id>" \
  --score-kind local_881_proxy \
  --aggregation "<fixed sequential aggregation definition>" \
  --successful-requests 881 \
  --error-requests 0 \
  --output-tps-p10 "<observed value>" \
  --input-tps "<observed value>" \
  --cache-tps "<observed value>" \
  --ttft-p90-s "<observed value>" \
  --cache-hit-rate "<observed value>" \
  --attest-same-run \
  --attest-exact-request-order \
  --out "$RUN/baseline-contract.json"
```

如果指标来自正式平台，应使用 `--score-kind official_platform`；这只描述指标
来源，不自动授权正式分数声明。

### Offline analysis

```bash
python3 scripts/analyze_prefix_cache_trace.py \
  "$RUN/server.log" \
  --qualification-trace \
  --baseline-contract "$RUN/baseline-contract.json" \
  --out "$RUN/cache-policy-analysis.json"
```

分析报告固定包含：

- `qualification_evidence_attested`；
- runtime、workload、request order 和 trace records 摘要；
- baseline 观测指标；
- `offline_phase_gate_passed`；
- 投影假设和固定开销未知风险；
- `main_or_yaml_change_authorized=false`；
- `official_score_claim_authorized=false`。

当前逐请求投影会按残余 prefill token 缩放非排队 TTFT，但尚未分离前端、调度和
其他固定截距，因此可能偏乐观。它只能筛选是否值得进行下一轮真实 TP4 A/B。

## 失败关闭覆盖

新增测试覆盖：

- 旧弱 baseline 与资格模式组合；
- 非 TP4、非 262144 或非新基础镜像的 runtime；
- candidate 策略伪装成 `fine32/direct/lru` 控制组；
- source revision 或 runtime 合同摘要漂移；
- workload request order、trace records 或日志顺序漂移；
- 非 881 请求、跨 session、重复 request ID 和 ordinal 缺口；
- metrics 请求计数、成功率或 weighted score 公式不一致；
- 缺少 same-run/request-order attestation；
- token、SSH 私钥和常见凭据标记。

生成器 round-trip 验证合同不含 `messages`、原始请求或模型输出。

## 验证

- 定向单测：32 项，失败 0；
- 完整非 GPU 单测：660 项，跳过 25，失败 0；
- Python 语法检查：通过；
- submission preflight：9/9；
- 敏感制品扫描：通过。

结构化证据见
`docs/experiments/evidence/M1_54_ATTESTED_881_CACHE_BASELINE_20260725.json`。

## 下一步

四卡恢复后，先用 `fine32/direct/lru/fused-prefill=0` 收集一次同代完整 881
trace 和指标，生成上述合同。只有离线阶段门禁通过，才安排相同镜像、命令、
请求顺序和清缓存流程的 TP4 control/candidate A/B。

真实 A/B 仍必须分别通过功能、cold/warm token 一致性、tool calling、
reasoning、多模态、IFEval、长上下文、Output TPS、TTFT、成功率和
fatal/OOM/Gloo/worker-loss 门禁。M1-54 不修改 `main`、
`computility-run.yaml` 或任何默认优化开关。
