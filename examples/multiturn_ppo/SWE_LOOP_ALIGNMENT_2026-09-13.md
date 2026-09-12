# SWE 执行循环对照、修复与全量重启

2026-09-13。用户要求恢复原版格式错误终止逻辑，检查其他差异，并在修复验证后自动重启完整 baseline。

## 已修复

对照来源为 `upstream/main` 的 `218f1f7c0bac0800de4d5a4e5e6f61cf7b5038b4`，文件 `examples/swe_smith/agents/smith_agent.py`。复用的解析、错误识别等函数来自本地固定 SHA-256 的该脚本，没有重新实现解析规则。

| 行为 | 原版 | 修复前的 Docker 适配 | 本次处理 |
| --- | --- | --- | --- |
| 连续格式错误 | 默认 3 次结束；成功解析后清零；非正数禁用限制 | 没有计数，继续到轮次/上下文预算 | 恢复相同规则，`SMITH_MAX_FORMAT_ERRORS=3` |
| 权重同步暂停 | 识别 429 gateway paused，原地等待，不消耗轮次 | SDK 重试耗尽后异常退出 | 复用原版分类器，默认最多等 600 秒，每 5 秒重试同一请求 |
| 服务端上下文溢出 | 识别 400 context overflow，结束且不插入空轮次 | 只有本地 tokenizer 预检，服务端仍可能报错 | 保留本地预检，同时复用原版服务端识别；记录终止原因，评分现有补丁 |
| 命令超时 | 默认 120 秒 | Docker 固定 45 秒 | 正式入口显式设 120 秒；其他未设置环境变量的通用 Sandbox 保持原默认 |
| 全量评分超时 | 默认 600 秒 | 全量评分器固定 300 秒 | 全量评分默认/正式入口均设 600 秒，Docker 客户端等待预算相应增加 |

最后一个触发格式终止的模型回复仍保存到轨迹，不能丢掉该次采样。达到上限后不再发请求、不再追加无意义用户反馈；仍按已产生的补丁正常评分，并通过 reward event 记录 `reason=format_errors`。命令执行失败或被策略拒绝不等于格式错误：只要动作可解析就清零，与原版一致。

## 明确保留的差异

本次逐项对照了模型请求、消息追加、动作解析、执行、提交、终止、评分和奖励路径；并非宣称整个 Lightning/CAPO 系统与官方 SWE 示例等价。

| 项目 | 保留差异与原因 |
| --- | --- |
| 一般 API 故障 | 原版 `_query` 返回空内容/error，循环消耗轮次并追加空 assistant。我们保留有限 SDK 重试后明确失败，不制造缺少实际采样 token 的 PPO 轮次；暂停超时也明确失败，不伪装成模型修复失败。SDK 重试仍为 2，原版 main 为 6，避免叠加长请求超时。 |
| 环境与隔离 | 原版 agent 在任务环境内执行；本地适配用宿主模型客户端及独立、无网络、非 root Docker 工具环境，评分另开新容器。保留私有 Git、补丁保护和环境修复。 |
| Prompt/编辑器/提交检查 | 保留原 SYSTEM/INSTANCE prompt，加已验证的编辑器和提交指引；`agl-edit` 检查精确上下文，提交前检查补丁与源码语法。空提交和非法修改会反馈给模型重试，不直接接受 submit marker。 |
| 轨迹与采样 | 保留完整历史、本地 token 预算预检、逐轮轨迹和 proxy token 记录；按现有采样配置进行训练/验证，保留按轮次派生的 seed 支持。 |
| 奖励 | 原版有成功训练样本的长度/上下文惩罚；本 baseline 按此前确定的二值终局修复奖励，不加入 reward shaping。32 轮预算下原版 80 轮阈值本就不触发，但长上下文惩罚可能触发，仍不引入。 |
| 评分细节 | 保留全部 F2P 与同文件 P2P、节点别名、在线依赖任务剔除、Patsy 测试恢复、Tenacity 文件分组、固定资源与兼容性修复。各项证据见已有诊断文档。 |
| 算法与规模 | 官方 SWE 示例是 GRPO；本任务使用 CAPO 移植的 token GAE + vanilla PPO/独立 critic，不是完整 CAPO action-ratio 论文方案。4B、6251/470、32 轮、4096 输出、65536 上下文和二值奖励保持已批准配方。 |

增加超时容忍和恢复原版终止条件会改变任务执行协议，不能称作纯数值等价的加速。用户已明确授权本次调整，因此旧验证与新验证分开保存，重新从原始模型跑完整验证/训练，不混合两套协议的成绩。

## 验证

- 65 项回归测试通过：真实调用 Docker agent 类（I/O 替身）与原版 `run_agent_loop` 比较连续错误、计数清零、禁用限制和上下文终止；另验证暂停重试的请求/seed 不变、等待有上限、普通故障不被吞掉。
- Ruff 和两个启动脚本 `bash -n` 通过。
- 重放实际拖尾任务 `oauthlib__oauthlib.1fd52536.func_pm_class_rm_funcs__s5ul0i6z` 的旧模型回复，在真实 Docker 中执行、导出、评分；恰好在第 9 轮、第 3 次连续格式错误结束，正常发出奖励 0。证据：`logs/swe-format-error-replay-20260913-01`。这是记录回复重放，不是新的模型能力评测。
- 旧运行 `capo-swe-pythonfull-v6-4b-20260913-01` 按用户要求停止，记录 `manual-stop-20260913.json`，保存 PID/start-time 快照及原因。只终止该 screen 的进程树和按其 rollout 标签确认的临时容器；GPU 0–3 已释放，同门 GPU 6/7 未动。
- 停止时没有训练 checkpoint，仍在初始验证阶段；旧结果原样保留，新运行从原始 4B 开始。

## 新运行状态

当前源码/v6 数据重新验收 **124/124 全部通过**，进程 exit 0，证据 `logs/swe-full-python-envs-20260913-08`。正式入口随后校验全部 **6721 条任务分支**并输出 `FULL_PYTHON_READY`。65 个相关本地/服务器源码及配置 SHA-256 一致。

保持 6251 train / 470 validation、4 个完整 epoch / 784 更新、有效 batch32、每卡 microbatch2、16 agents、每 vLLM 实例最多4序列，仅物理 GPU0–3。先完整验证，再训练；每196步验证、每32步保存可恢复状态。算法、学习率、KL、精度、训练/验证采样分布不变。

新运行：

- tag：`capo-swe-pythonfull-v6-4b-20260913-02`
- screen：`agl-full-v6-4b-20260913-02`
- 运行目录：`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-capo-swe-pythonfull-v6-4b-20260913-02`
- 启动日志：`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/launch-capo-swe-pythonfull-v6-4b-20260913-02.log`
- 端口18401。`provenance.json` 已确认 `SMITH_MAX_FORMAT_ERRORS=3`、`SMITH_GATEWAY_WAIT_S=600`、`SMITH_CMD_TIMEOUT=120`、`SMITH_EVAL_TIMEOUT=600` 和 `CUDA_VISIBLE_DEVICES=0,1,2,3`。
- 02:53 检查：模型已加载并开始训练前470条完整验证；首批32条已有18条完成（执行失败0），真实奖励1共2条，18份评分及trace已保存。无 run.exit，尚未首次PPO更新。这是启动快照，不是完整验证成绩。前四卡已实际采样，D1空闲约505.35GiB。
- 每3小时巡检 `3-swe-ppo` 已更新为新运行并保持 ACTIVE，明确旧运行人工停止，恢复时不得退回旧终止规则。

核心启动命令如下（同名运行存在时不要重复执行）：

```bash
cd /media/ubuntu/D1/zsj/agent-lightning-main/agent-lightning-main
export AGL_TRAIN_TAG=capo-swe-pythonfull-v6-4b-20260913-02
export AGL_TRAIN_PORT=18401 AGL_GPUS=0,1,2,3
bash examples/multiturn_ppo/run_full_python_ppo.sh
```

日志查看：

```bash
tail -f /media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-capo-swe-pythonfull-v6-4b-20260913-02/trainer.log
```
