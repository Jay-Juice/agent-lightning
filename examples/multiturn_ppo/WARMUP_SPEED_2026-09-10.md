# Critic 预热对照与 A800 提速验收

## 目的与边界

上一轮同时降低 actor LR、关闭优势标准化、加入 reference KL、增加 critic 预热，
不能据此断言某一个配置项必要。本次首先取消预热，其他学习配置保持一致。
veRL 默认 `critic_warmup=0`；预热是可选措施。仓库默认 SWE GRPO 没有 critic，
不能直接将其有无预热与这里新增独立价值网络的 PPO 相比较。

无预热试验从原始 Qwen3-1.7B 开始，32/6 数据、4 卡、batch=4、8 turns、
768 response tokens、12288 context，actor LR=1e-6、critic LR=1e-5、
不标准化优势、reference KL loss=0.02、gamma=lambda=1、终局二值奖励。
8 个 global steps，完整遍历当前 32 个训练任务。
与之前 12 步试验相比，actor 都有 8 批更新，但之前额外有 4 批 critic-only 预热，
actor 所见任务顺序也不同；因此它是实用短测，不是多种子严格因果消融。

## 无预热命令（已执行）

```bash
screen -dmS agl-ppo-no-warmup-four-1p7b-20260910-01 env \
  AGL_TRAIN_TAG=ppo-no-warmup-four-1p7b-20260910-01 \
  AGL_GPUS=4,5,6,7 AGL_TRAIN_PORT=18282 \
  bash /media/ubuntu/D1/zsj/agent-lightning-main/agent-lightning-main/examples/multiturn_ppo/run_smith_stable.sh \
  --steps 8 trainer.total_epochs=1 trainer.critic_warmup=0
```

运行目录为 D1 runtime `logs/training-ppo-no-warmup-four-1p7b-20260910-01`。

## 执行效率配置

新增 `run_smith_fast.sh`，不改变优化器全局 mini-batch、PPO epoch、学习率、
奖励定义、训练采样次数、token 预算或 bf16 精度：

- actor/critic 参数和优化器、reference 参数留在 GPU，关闭 CPU offload。
- 关闭 actor/critic gradient checkpointing，用显存换取减少重算。
- log-prob/value 推理 micro-batch 从每 GPU 1 增到 2；训练 micro-batch 仍为 1。
- vLLM 启用 CUDA graph（enforce_eager=false）。
- 本地 agent 并发上限从 4 增到 8，使 6 条验证可同时开始；训练 task batch 仍为 4。
- 每 2 秒记录所选 GPU 的利用率、显存、功率到运行目录 `gpu.csv`；运行器退出时
  清理自己启动的采样进程，不建立额外后台监控任务。

这些是执行配置，数学目标不变，但并行执行和 kernel 路径不保证逐位相同的随机轨迹。
必须实测显存、耗时及输出表现后再判断能否采用。不能把 GPU 利用率的提升等同于
训练效果提升，也不能单凭一次 GPU 快照判断整体速度。

## 当前状态

所有本次试验已结束，`run.exit=0`，GPU 0–7 均已释放；没有启动全量训练。

### 无预热结果

8/8 步通过 PPO 张量和四卡检查点验收，actor/critic 每个 rank 均有 61 次 optimizer
更新。237 次训练调用中 233 次可解析命令（98.31%），length 格式错误为 0。
最后一步平均响应 83.47 tokens，截断率 0；最终验证平均每轮 87.75 tokens。
训练和验证奖励仍全为 0，不能以此证明任务质量相等或预热在更长训练中没有作用。

结论：当前小规模配置未显示必须预热，新的 fast 入口默认 `critic_warmup=0`、
8 步 / 1 epoch；历史 `run_smith_stable.sh` 保留预热以供复现。
这里的预热是 critic-only 更新，不能与 optimizer 学习率 warmup 混为一谈。

### 提速命令（已执行）

```bash
screen -dmS agl-ppo-fast-four-1p7b-20260910-01 env \
  AGL_TRAIN_TAG=ppo-fast-four-1p7b-20260910-01 AGL_GPUS=0,1,2,3 AGL_TRAIN_PORT=18281 \
  bash /media/ubuntu/D1/zsj/agent-lightning-main/agent-lightning-main/examples/multiturn_ppo/run_smith_fast.sh \
  --steps 4 trainer.total_epochs=1 trainer.critic_warmup=0 \
  trainer.max_actor_ckpt_to_keep=1 trainer.max_critic_ckpt_to_keep=1
```

4/4 步正常结束，PPO 张量和检查点验收通过，actor/critic 各 rank 均更新 29 次。
其中第 2 步有一批较长输出，截断比例 25.93%；第 3、4 步为 0，最终验证成功率仍为 0。
这只验证了执行和短程行为，不能证明所有任务的模型质量不受任何数值差异影响。

两组都排除首次编译的第 1 步，比较 step 2–4（不含初始化、验证、检查点保存）：

| 指标 | CPU offload / 重算 | GPU 常驻提速 |
| --- | ---: | ---: |
| 处理的真实 input+response tokens | 180366 | 172234 |
| 平均 step 耗时 | 88.96 s | 71.55 s |
| actor+critic 更新耗时 / 1000 tokens | 0.68714 s | 0.58725 s |
| old/ref log-prob + value 推理耗时 / 1000 tokens | 0.22464 s | 0.11928 s |

平均每步耗时减少约 19.6%；按 token 归一化，更新阶段约 1.17 倍、推理计算阶段约
1.88 倍速度。轨迹长度及调用数不同，且样本很少；这是工程短测估计，不是严格固定
输入的吞吐基准，也不是全量训练速度承诺。
每 2 秒 GPU 采样峰值为 56337 MiB（约 55.0 GiB），各训练步内平均利用率依次为
54.8%、58.5%、66.8%、75.6%。这些利用率不包含整段启动/保存时间。
A800 拓扑为两卡一组 NV8，0–3 在同一 NUMA 节点；本次保留四卡 FSDP 分片方式。

### 更充分交互预算评估（已执行）

```bash
screen -dmS agl-ppo-eval-24turn-1p7b-20260910-01 env \
  AGL_TRAIN_TAG=ppo-eval-24turn-1p7b-20260910-01 AGL_GPUS=4,5,6,7 AGL_TRAIN_PORT=18282 \
  SMITH_MAX_TURNS=24 SMITH_CONTEXT=32768 \
  bash /media/ubuntu/D1/zsj/agent-lightning-main/agent-lightning-main/examples/multiturn_ppo/run_smith_fast.sh \
  --steps 1 trainer.val_only=true trainer.critic_warmup=0 agentlightning.rollout_timeout_seconds=2400
```

使用原始 Qwen3-1.7B，只有验证，没有更新模型。实际 config 为 val_only=true、
max_prompt_length=32000、max_response_length=768、max_model_len=32768。
结果仍为 0/6 成功，平均 21.17 轮；5 个任务耗尽 24 轮，1 个任务第 7 轮提前提交。
127 次调用全部可解析，4 个任务最终没有 patch，另 2 个有 patch 但未修复成功。
因此当前样本的瓶颈并非仅是 8 轮上限或命令格式，仍未建立成功修复反馈。
不能从 6 个任务推断整个模型的能力上限，也不能用程序正常退出作为扩量依据。

### 参数与执行检查

SWE 预算现在由环境变量统一配置，并同时传给 agent、数据和 vLLM。
默认值保持 8 turns / 768 response tokens / 12288 context。
0 response budget、context 不大于 response budget、非整数 turns 均已验证提前返回 2。
Bash 语法、Python Ruff/格式和 git diff whitespace 检查通过。
`gpu.csv` 采样进程随各自运行器退出，未留下额外监控任务。
所有源文件通过 WSL rsync 先 `-anizc` 预览，再 `-aizc` 部署，未同步运行产物或使用删除选项。

全量仍需任务效果验收及其余仓库镜像/评分适配准备；本次效果条件未满足，未启动全量。
