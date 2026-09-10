# 多轮 PPO baseline

本入口在 Agent Lightning 的 veRL 集成上运行普通、独立 critic 的 PPO。
Actor 和 critic 都只读取当前轮可见历史；尚未加入 privileged information。
默认使用 A800 的现有环境和本地模型，不下载模型或数据。

按用户要求另提供 [CAPO PPO 直接移植版](CAPO_PPO.md)：复制其 GAE、PPO loss 和
Actor/Critic 更新代码，使用独立 `backend=capo`，保留来源哈希与原始尾批语义。

## 算法约定

- 每次模型调用一条 transition，保留真实 prompt 和 response token IDs。
- 同一 rollout 的调用按 `turn_index_list` 串联；重复 prompt 的调用仍保留。
- 工具观察只出现在下一轮 prompt，不参与 actor loss，也不占折扣步数。
- critic 输出 response 每个 token **生成前**的状态价值，沿用 veRL 原生 causal value 对齐。
- 奖励只放在最终调用最后一个 response token。默认二值成功奖励，γ=1、λ=1。
- 对整个 episode 的 action tokens 反向计算
  `δ_t = r_t + γ V_{t+1} − V_t`、`A_t = δ_t + γλ A_{t+1}`。
  终局 `V_{T+1}=0`；critic target 为未归一化的 `A_t + V_t`。
- actor advantage 在整个已采集 batch 的有效 action tokens 上 whitening；critic target 不 whitening。
- vanilla clipped PPO，actor/critic 学习率均 1e-5，weight decay=0，KL 和 entropy 系数均为 0。
  critic 使用模型默认初始化的 value head，无额外初始化、warmup 或奖励塑形。

这是同步采集完整 episode 的实现。任务的轮数/上下文预算用尽后，对当前 patch 评分并终止，
属于任务终局。尚未支持采样器把**未结束 episode**截断后 bootstrap 的训练方式。
基础设施失败、空响应、缺失奖励、超长训练输入和缺失轮次会使当前 step 明确失败。

所有轮次均保留，不执行原 GRPO 路径的按奖励删行、mini-batch 向下取整或更新次数截断。
当前单卡默认 `rollout.n=1`、actor/critic `ppo_mini_batch_size=1`、micro batch=1，
每条调用构成一次 mini-batch，损失在该调用的有效 token 上取平均。
四卡入口启用 `distributed_padding`：推理时补齐后按标记移除占位行，再计算真实轨迹的 GAE；
更新时补齐的调用使用零 response mask，损失按每次优化中所有 rank 的真实调用数归一化。
每个调用先按有效 token 取平均，因此占位行不产生 actor/critic 梯度，也不改变真实调用权重。
当前四卡配置限定 FSDP legacy workers、每 rank 一个调用、无序列并行、`rollout.n=1`。
比较实验时应保持全局优化 batch 一致；单卡 mini-batch=1 与四卡 mini-batch=4 的更新频率不同。

## 在 A800 运行

本地源代码先按项目外层 `A800/AGENTS.md` 用 rsync 预览、部署。
在服务器项目根目录运行：

```bash
# 两轮算术任务：每个训练 step 采集 2 个完整 episode，默认训练 2 steps。
AGL_TRAIN_TAG=ppo-smoke-my-run AGL_GPU=4 \
  bash examples/multiturn_ppo/run_training.sh --steps 2

# Qwen3-1.7B，缓存的 exceptiongroup SWE-smith 子集，1 training step。
AGL_TRAIN_TAG=ppo-smith-my-run AGL_GPU=4 \
  bash examples/multiturn_ppo/run_smith_training.sh

# 四卡 baseline：缓存子集 32 train / 6 validation，32 steps = 4 epochs。
AGL_TRAIN_TAG=ppo-four-baseline-my-run AGL_GPUS=0,1,2,3 \
  bash examples/multiturn_ppo/run_four_gpu.sh --steps 32 \
  --train-file /media/ubuntu/D1/zsj/agent-lightning-runtime/data/swe-smith-training/exceptiongroup/train.jsonl \
  --val-file /media/ubuntu/D1/zsj/agent-lightning-runtime/data/swe-smith-training/exceptiongroup/val.jsonl \
  data.val_batch_size=6 trainer.total_epochs=4 trainer.test_freq=8 trainer.save_freq=8

# 四卡不整除尾批短测：4 个合成任务共 9 次调用，补齐为 12 行。
AGL_TRAIN_TAG=ppo-four-smoke-my-run AGL_GPUS=0,1,2,3 \
  bash examples/multiturn_ppo/run_four_gpu_smoke.sh
```

运行器检查 GPU 空闲、D1 剩余空间和端口，将每次实验写入全新目录，
并只清理自己启动的 gateway/controller。长任务请放入独立命名的 screen。
模型可用 `AGL_TRAIN_MODEL` 改写，端口可用 `AGL_TRAIN_PORT` 改写。
默认网关端口 18281，GPU 4；不接入或停止已有 Ray 服务。
四卡入口默认使用 GPU 0、1、2、3；vLLM TP=1，共 4 个推理副本，actor/critic 采用 4-rank FSDP。

SWE 入口默认 8 turns、每次最多 768 response tokens、总 context 12288，
训练 prompt 上限 11520。数据是已缓存的 SWE-smith 4 train / 1 validation smoke 子集，
不代表 SWE-bench Verified 成绩。正式 benchmark 仍需独立 held-out 数据及官方评测。
保留之前经过环境验证的隔离 Docker 适配器和评分方式：训练镜像按 digest 固定，
actor 无宿主机挂载、无网络、无 Git history；patch 在另一容器中用测试集判定成功。
现有适配器禁止测试/配置文件修改，结果不能直接当作 unrestricted SWE-bench 分数。

## SWE 稳定性修复试验

原四卡 SWE baseline 出现持续零奖励和输出退化，不能作为全量训练的已验收配置。
`run_smith_stable.sh` 提供独立的保守试验配置：原始 Qwen3-1.7B、四卡、32/6 数据、12 步，
前 4 步仅更新 critic，第 5 步起更新 actor；关闭优势标准化，actor LR=1e-6，
参考模型 KL loss coefficient=0.02，critic LR=1e-5，终局奖励仍是二值，gamma=lambda=1。
该组合用于检验稳定性，不能据此认定每个配置项的独立因果贡献或已获得任务能力提升。

```bash
AGL_TRAIN_TAG=ppo-stability-my-run AGL_GPUS=4,5,6,7 AGL_TRAIN_PORT=18282 \
  bash examples/multiturn_ppo/run_smith_stable.sh
```

新增指标 `ppo/raw_advantage_std`、`ppo/actor_advantage_std`、`ppo/advantage_scale` 和
`ppo/actor_updated`，用于区分 critic 误差、标准化放大与预热。审计同时保存 `ref_log_prob`。
从退化 checkpoint 恢复不能检验原始模型的稳定性，本试验默认从预训练权重重新开始。

### A800 提速和无预热配置

`run_smith_fast.sh` 默认四卡、8 步、无 critic 预热，其余学习配置沿用上述保守配置。
它将参数和优化器留在 GPU，关闭激活重算，推理 micro-batch=2，启用 vLLM CUDA graph，
本地 agent 并发上限=8，并写入 `gpu.csv`。训练 global mini-batch=4、micro-batch=1。
数学目标和精度不变，随机轨迹不保证逐位一致；实测结果见
[预热与速度对照](WARMUP_SPEED_2026-09-10.md)。历史预热配置仍保留用于复现。

```bash
AGL_TRAIN_TAG=ppo-fast-my-run AGL_GPUS=0,1,2,3 \
  bash examples/multiturn_ppo/run_smith_fast.sh
```

SWE 的 `SMITH_MAX_TURNS`、`SMITH_MAX_TOKENS` 和 `SMITH_CONTEXT` 可通过环境变量调整；
入口同步设置 agent、数据和 vLLM 的 token 上限，默认仍为 8 / 768 / 12288。
`AGL_MAX_LOCAL_AGENTS` 控制本地并发；有效值写入 `provenance.json` 的 `runtime_options`。

## 断点恢复

```bash
AGL_TRAIN_TAG=ppo-resume-my-run AGL_GPU=4 \
  bash examples/multiturn_ppo/run_training.sh --steps 3 \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path=/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-ppo-smoke-my-run/checkpoints/global_step_2
```

`--steps` 是恢复后计划达到的总 global step。veRL 恢复 actor、critic、两者 optimizer/scheduler
及 dataloader 状态。恢复时模型、数据及训练配置应与原实验一致；新日志与新检查点进入新目录。

## 记录与验证

每次运行目录：`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-<TAG>/`。

- `resolved-config.json`：解析后配置，网关密钥已脱敏。
- `provenance.json`：软件版本及参与训练的源文件 SHA256（恢复和 SWE 验证开始记录）。
- `metrics.jsonl`：每个 step 的 actor/critic loss、grad norm、价值与奖励统计。
- `traces/`、`agent/`：原始调用、工具观察、模型 patch、独立评分及终止原因。
- `ppo-audit/step-*.pt` 和对应 JSON：更新前 token IDs、mask、log-probs、values、
  token rewards、raw/whitened advantages、returns、rollout ID、轮次和终局标记。
- `checkpoints/global_step_*/actor` 与 `critic`：模型、optimizer 和恢复状态。
- `run.exit`：进程退出码。

任务成功率看 `training/reward`、`val/reward`。
`critic/score/mean` 是调用行的奖励均值，其中中间调用为零，不能当作 episode 成功率。

审计张量可能较大；正式长训练可设置 `agentlightning.multi_turn_ppo.audit_dir=null`，
保留 metrics 和按需求配置的轨迹日志。实验输出只留在 D1，不提交 Git。

```bash
source /media/ubuntu/D1/zsj/agent-lightning-runtime/admin/activate-agent-lightning-d1.sh
python -m pytest -q tests/verl tests/server/test_triplet_preserve.py
python examples/multiturn_ppo/verify_run.py --run /absolute/path/to/training-TAG
```

核心开关为 `agentlightning.multi_turn_ppo.enabled=true`。
公共配置默认关闭，已有 GRPO 路径及默认 triplet 去重接口保持原行为。
部署新训练器时须同时部署服务器的 `triplet-preserve` 事件接口。

实测配置、运行目录与结果见 [2026-09-10 验收记录](VALIDATION_2026-09-10.md)。
四卡尾批处理、短测结果和 baseline 命令见 [四卡验收与运行记录](VALIDATION_FOUR_GPU_2026-09-10.md)。
原 SWE 配置的输出退化诊断及保守配置试验见 [稳定性修复记录](STABILITY_FIX_2026-09-10.md)。
实现来源、与旧 Agent-R1/CAPO PPO 的实际差异、独立 GAE 数值对照及 SWE 失败轨迹见
[PPO 参考实现审计](PPO_REFERENCE_AUDIT_2026-09-10.md)。当前 fast/stable 是实验配方，
不代表上游 SWE 示例或旧 CAPO PPO baseline 的完整复现。
