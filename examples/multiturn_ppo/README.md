# 多轮 PPO baseline

本入口在 Agent Lightning 的 veRL 集成上运行普通、独立 critic 的 PPO。
Actor 和 critic 都只读取当前轮可见历史；尚未加入 privileged information。
默认使用 A800 的现有环境和本地模型，不下载模型或数据。

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
若改用更大 mini-batch 或多卡，完整调用数必须整除相应大小，否则明确报错；
尚未实现任意尾批的无偏分布式优化。运行结果比较时应保持这一损失权重和 batch 配置一致。

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
```

运行器检查 GPU 空闲、D1 剩余空间和端口，将每次实验写入全新目录，
并只清理自己启动的 gateway/controller。长任务请放入独立命名的 screen。
模型可用 `AGL_TRAIN_MODEL` 改写，端口可用 `AGL_TRAIN_PORT` 改写。
默认网关端口 18281，GPU 4；不接入或停止已有 Ray 服务。

SWE 入口默认 8 turns、每次最多 768 response tokens、总 context 12288，
训练 prompt 上限 11520。数据是已缓存的 SWE-smith 4 train / 1 validation smoke 子集，
不代表 SWE-bench Verified 成绩。正式 benchmark 仍需独立 held-out 数据及官方评测。
保留之前经过环境验证的隔离 Docker 适配器和评分方式：训练镜像按 digest 固定，
actor 无宿主机挂载、无网络、无 Git history；patch 在另一容器中用测试集判定成功。
现有适配器禁止测试/配置文件修改，结果不能直接当作 unrestricted SWE-bench 分数。

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
