# 天数智芯 天垓100 文本生成引擎（基于 vLLM 优化适配Qwen3.6-35B-A3B）

协作交接、远端阻塞点、最新验证结果和下一步建议见
[`docs/HANDOFF_SUMMARY.md`](docs/HANDOFF_SUMMARY.md)。

自 2026-07-20 起，天垓100 v1.2.3 基础镜像必须使用
`harbor.4pd.io/modelhubxc/enginex-iluvatar/bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3`。
旧的 `git.modelhub.org.cn:9443` 地址已经停用，不得用于后续提交。

```
# 本地构建
docker build -t enginex-iluvatar-vllm:bi100-qwen3.6 -f Dockerfile .
```

Docker 构建会校验并安装仓库内由 CoreX 3.2.3/ivcore10 生成的 10 个扩展，
不在评测构建阶段重新运行 clang。扩展源码和独立构建脚本仍保留用于开发，正式
镜像只接受 `qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10/SHA256SUMS`
中固定的二进制。


启动容器镜像

下载 Qwen3.6-35B-A3B 模型并以只读方式挂载到 `/model`。
镜像构建时会注册 Qwen3/Qwen3_5/Qwen3_6 的 vLLM registry alias，
不需要、也不应该手动修改 `/model/config.json`。

```bash
docker run -dit --network=host --ipc=host \
  -v /usr/src:/usr/src -v /lib/modules:/lib/modules -v /dev:/dev --privileged \
  -v /mnt/disk1/models/Qwen3.6-35B-A3B:/model:ro --entrypoint=python3 \
  -e CUDA_VISIBLE_DEVICES=4,5,6,7 -e VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
  enginex-iluvatar-vllm:bi100-qwen3.6 \
  -m vllm.entrypoints.openai.api_server \
  --model /model --port 1111 --served-model-name llm \
  --max-model-len 262144 --trust-remote-code -tp 4 --gpu-memory-utilization 0.90 \
  --max-num-seqs 1 --disable-log-requests --disable-frontend-multiprocessing \
  --max-num-batched-tokens 8192 --enable-chunked-prefill \
  --max-seq-len-to-capture 32768 --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --enable-prefix-caching
```

请求
```bash
curl http://localhost:1111/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llm",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Can you tell me the story of Snow White?"}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

本地 P0 静态检查

当前轻量环境可能没有安装 `pytest`。不需要为静态检查额外安装依赖，
可以直接使用 Python 标准库运行：

```bash
python tests/submission_preflight.py
python tests/test_p0_static.py
python tests/test_clients_unit.py
python tests/test_paged_attn_unit.py
python tests/test_patch_utils_unit.py
python tests/test_patch_transformers_unit.py
python tests/test_patch_registry_unit.py
python tests/test_patch_tool_parser_unit.py
python tests/test_patch_worker_profile_override_unit.py
python tests/test_protocol_unit.py
python tests/test_serving_chat_unit.py
python tests/test_tool_parser_unit.py
python tests/test_submission_preflight_unit.py
```

`submission_preflight.py` 对根目录制品、固定 YAML argv/env、离线 wheel
size/SHA256、10 个预编译 CoreX 扩展的集合/大小/SHA256、关键文件 LF 换行、
全部 shell 语法和 Python 语法执行统一 RC 门禁；任何检查失败都返回非 0，禁止提交。

M1-31 引入稳定 SHA-256 前缀键和 scheduler-owned GDN 状态动作协议。当前默认是
`BI100_GDN_CACHE_POLICY=fine32`、`BI100_GDN_RESTORE_MODE=direct`；这两个默认值
尚待新的 TP4 实例完成 cold/warm correctness 与 A/B 资格化。离线诊断可显式开启
`BI100_CACHE_TRACE=1`，但正式性能提交必须保持关闭。设计和固定验证矩阵见
[`docs/experiments/M1_31_STABLE_GDN_PREFIX_STATE_20260719.md`](docs/experiments/M1_31_STABLE_GDN_PREFIX_STATE_20260719.md)。

完整本地标准库测试可运行：

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

该检查覆盖 patch fail-fast、动态 vLLM/transformers 路径、registry alias、
顶层 thinking、`tool_choice="none"`、streaming tool arguments 序列化、
streaming tool argument name JSON 转义和 guided decoding/tool_choice 冲突保护、
BI100 attention guard、executor 启动诊断开关、Qwen3 coder tool parser 转换 fallback 必须可观测、
transformers 配置兼容 stub 不允许 broad exception fallback、
`launch_service` 默认参数与 CoreX 环境路径的关键静态约束。
`test_clients_unit.py` 覆盖远端 smoke 和 benchmark 客户端对 SSE usage-only
chunks (`choices=[]`) 的处理、smoke client HTTP error JSON/raw body 保留、
以及 benchmark 分位数/加权分公式。
`test_paged_attn_unit.py` 使用轻量 `torch`/`vllm` stub 加载真实
`paged_attn.py`，覆盖 BI100 attention env 默认值/覆盖值、`BI100_FORCE_PAGED_ATTN_V2`
显式 opt-in 路由，以及 prefix block table guard 默认 raise/debug cap 行为。
`test_patch_utils_unit.py` 覆盖 shared patch helper 的包路径发现、
file/dir guard、`replace_once`、`replace_one_of`、required/optional anchor
行为以及 shell-safe env 输出。
`test_patch_registry_unit.py` 使用临时 fake vLLM package 执行真实
`patch_vllm_qwen3_5.py`，覆盖 Qwen3/Qwen3_5/Qwen3_6 registry alias 安装、
重复执行幂等性以及 anchor 缺失时 fail-fast。
`test_patch_transformers_unit.py` 使用临时 fake transformers package 执行真实
`patch_transformers_qwen3_5.py`，覆盖 qwen3_5/qwen3_5_moe auto config 注册、
models import 注册、重复执行幂等性以及 anchor 缺失时 fail-fast。
`test_patch_tool_parser_unit.py` 使用临时 fake vLLM package 执行真实
`patch_vllm_tool_parser.py`，覆盖 `qwen3_coder` parser import/`__all__`
注册、重复执行幂等性以及 anchor 缺失时 fail-fast。
`test_patch_worker_profile_override_unit.py` 执行真实
`patch_worker_profile_override.py`，覆盖干净 vendor `worker.py`、已有
startup-profile guard 的旧布局、二次执行幂等性，以及未知布局时
fail-fast。
`test_protocol_unit.py` 使用轻量 runtime stub 直接实例化
`ChatCompletionRequest`，覆盖 thinking 归一化、`tool_choice="none"`、
tools 默认 `auto`、named tool 校验以及 forced tool/guided decoding 冲突。
`test_serving_chat_unit.py` 从 `serving_chat.py` AST 中执行真实
`_serialize_tool_arguments` helper，覆盖字符串 arguments 不被 double-json、
dict/list 正常 JSON 编码以及 `None` 返回 `{}`。
`test_tool_parser_unit.py` 使用轻量 vLLM stub 直接加载 Qwen3 coder tool parser，
覆盖 streaming tool call 增量参数名 JSON 转义后仍可组合成合法 arguments。
`test_preflight_unit.py` 覆盖 BI100 CUDA/NCCL 预检查脚本的参数解析、CoreX
环境拼接、timeout 输出清洗和本地端口选择等纯 Python 辅助逻辑。

当前 T5-T9 的固定契约执行方案见
[`docs/OPTIMIZATION_PLAN_T5_T9.md`](docs/OPTIMIZATION_PLAN_T5_T9.md)。旧的并发、
`max_num_seqs` 和 GPU-block override 实验不作为正式评测基线。

远端 smoke 与 benchmark

`tests/smoke_api.py` 和 `tests/bench_perf.py` 只使用 Python 标准库，
适合远端环境没有 `curl` 或 `requests` 时直接运行：

```bash
python tests/smoke_api.py --base http://127.0.0.1:8000 --mode quick
python tests/bench_perf.py --base http://127.0.0.1:8000 \
  --label fixed-contract --requests 8 --workers 1 --stagger-s 0 \
  --max-tokens 64 --prompt-salt fixed-contract --out bench-fixed.json
python tests/prefix_cache_stress.py --base http://127.0.0.1:8000 \
  --eviction-count 17 --json-out prefix-cache-stress.json
python tests/long_context_api.py --base http://127.0.0.1:8000 \
  --target-prompt-tokens 99500 --output-dir long-context-artifacts
```

`smoke_api.py --mode quick` 覆盖基础 chat、thinking=false 三种格式、
`tool_choice="none"`、`response_format={"type":"json_object"}`、
`response_format={"type":"json_schema"}`、streaming SSE usage、异常请求 4xx
和 prefix cache。`--mode full` 额外覆盖 stop 与强制 tool call。
其中 full 模式同时覆盖强制 tool call 的非流式和流式增量参数路径、
多语言/多轮消息、sampling 合法/非法边界以及 seed 确定性。

`bench_perf.py` 会为每轮生成唯一或显式指定的 prompt salt，避免上一轮
benchmark 预热 prefix cache 后污染结果。输出字段包括请求成功率、TTFT P90、
Output TPS P10、Input TPS、Cache TPS、缓存命中率和加权分。

`prefix_cache_stress.py` 使用本地 tokenizer 构造 16-token 对齐和未对齐长前缀，
验证 A/B/A/B 交错会话隔离，并以 17 个不同前缀覆盖 GDN checkpoint LRU 的驱逐、
安全重算和刷新后再次命中。

`long_context_api.py` 精确构造 99,500-token chat prompt，在固定 100K 合同边界内
验证首次 prefill、近全量 prefix hit、API usage 和两次响应等价性。
GDN state 需要按 chunk 分阶段恢复；`cached_tokens` 会累计每阶段实际跳过的 token，
而不是只报告第一个 8,176-token checkpoint。

BI100 预启动检查

如果服务启动停在 executor `init_device` 或迟迟没有 `Loading safetensors`，
先运行每卡 CUDA 预检查，确认 4 张卡都能完成 `mem_get_info` 和小矩阵运算：

```bash
python tests/bi100_preflight.py --timeout-s 25 --matmul-size 1024 \
  --json-out bi100-preflight.json
```

该脚本会自动补齐常见 CoreX `PATH`、`LD_LIBRARY_PATH`、`PYTHONPATH`，任一 GPU
失败或超时都会返回非 0。不要在预检查失败时继续启动 TP=4 vLLM。
只有该检查全绿后，再运行 NCCL all-reduce 预检查：

```bash
python tests/bi100_nccl_preflight.py --timeout-s 45 \
  --json-out bi100-nccl-preflight.json
```

如果 NCCL 检查失败，同样不要启动 TP=4 vLLM，先处理通信或 GPU 状态问题。

数值稳定性排查

远端 BI100 当前环境支持 `torch.bfloat16` 基础矩阵运算。2026-07-06 观察到
`--dtype bfloat16` 可以避免 A 配置启动、smoke、benchmark 期间的
GatedDeltaNet NaN warning，但同一标准 benchmark 下性能明显变差，不作为默认
提交参数。需要复现时可在 `launch_service` 中设置：

```bash
DTYPE=bfloat16 ./launch_service
```

默认仍不传 `--dtype`，保持 `computility-run.yaml` 当前配置；继续调优时应把
NaN warning 作为效果正确性风险单独记录。

当前默认路径已改为在 GatedDeltaNet core output 进入 gated RMSNorm 前保持
float32，避免先转回 fp16 造成溢出后再归一化。同时，GatedDeltaNet 的非有限值
默认直接抛错，不再静默替换为 0。只有排查问题时可以显式打开：

```bash
BI100_GDN_ALLOW_NAN_ZERO=1
```

该开关会记录风险日志并把非有限值替换为 0，不应作为正式提交默认行为。

启动卡住排查

如果服务日志停在 worker ready、迟迟没有出现 `Loading safetensors`，可临时启用：

```bash
BI100_EXECUTOR_STARTUP_DEBUG=1
```

该开关只增加 executor 启动阶段日志，不改变正常推理路径。若日志停在
`init_device`，先用单卡 CUDA 小矩阵探测各 GPU；若某张卡在 `mem_get_info`
前超时，需先从宿主机侧 reset 或更换 BI100 分配，再继续 TP=4 benchmark。
