# 四卡 PPO 支持及 baseline 启动记录

## 四卡短测

运行目录：`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-ppo-four-smoke-20260910-01`。
使用物理 GPU 0、1、2、3，Qwen3-0.6B，4 个算术任务的调用数分别为 1、2、3、3。

结果：`run.exit=0`，`verification.json` 为 `verified`。

| 检查项 | 实测 |
| --- | --- |
| 真实 episodes / 调用 / action tokens | 4 / 9 / 532 |
| 进入 GAE 的调用数 | 9，无占位行 |
| 更新时占位行 / 总物理行 | 3 / 12 |
| Actor grad norm | 54.92640694 |
| Critic grad norm | 283.25506083 |
| Actor / critic optimizer steps | 各 4 个 rank 全部为 3 |
| 核心 policy loss / value loss | 0.05298425 / 13.25762284 |
| 该 training step 耗时 | 44.48 秒，不含初始化和检查点保存 |

所有有效 token 的 values、advantages、returns、log-probs 有限；
所有真实调用均保留；终局奖励正确回传至先前轮次；actor/critic 四份模型、optimizer 和 extra state 均保存。

79 项 veRL/服务器测试通过；新增测试包含乱序后移除推理占位行、更新占位零权重、
不同长度真实调用与 dummy-only rank 的梯度等价、原生 value loss 的零占位梯度。
新增核心代码静态类型检查为 0 errors；Ruff 和格式检查通过。

## 实现约定

- 使用一个 Ray 作业，4-rank FSDP actor/critic 和 4 个 TP=1 的 vLLM 副本。
- 推理阶段补齐到 4 的倍数，用 `ppo_padding` 标记跟随负载均衡重排；计算 GAE 前移除占位行。
- 更新阶段再次补齐，每个 rank 每次优化处理 1 行；占位行 response mask 全零，但保留有效模型输入。
- 每个真实调用先按 action token 平均；各 rank 的损失求和再按**本次优化的全局真实调用数**归一化。
  乘 world size 抵消 FSDP 的梯度平均，避免空 rank 的零分母和占位导致的梯度缩小。
- 保留 veRL 原生 clipped policy/value objective、optimizer、checkpoint 和梯度裁剪。
  仅通过该作业的 worker hook 替换 `seq-mean-token-mean` 聚合；默认单卡/GRPO 路径不安装此 hook。
- 当前限定 legacy FSDP、无序列并行、micro-batch=1、global mini-batch=GPU 数、rollout.n=1。
  必须由入口初始化专属 Ray 作业，以确保每个 worker 安装同一聚合逻辑。

PPO 审计和 trainer 数据统计只包含真实调用；核心 policy/value loss 与梯度按真实调用归一化。
veRL 的部分辅助 worker 统计（如 clip fraction、vpred_mean）仍是 rank 局部均值再聚合，
出现占位时不能将其当作严格的全体真实 token 统计；需要精确分析时使用 `ppo-audit` 的真实行。

## 四卡 SWE baseline

2026-09-10 16:53:56（北京时间）启动，screen：`agl-ppo-baseline-four-1p7b-20260910-01`。
运行目录：`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-ppo-baseline-four-1p7b-20260910-01`。

Qwen3-1.7B 从预训练权重开始，4 × A800 80GB，32 train / 6 validation，task batch=4，
rollout.n=1，32 global steps=4 epochs，最多 128 个训练 episodes。
每个 episode 最多 8 轮、每轮 768 response tokens、context 12288；gamma=lambda=1；
终局二值奖励；actor/critic LR=1e-5；KL=entropy=0；每 8 步评测与保存。
每个角色保留最近 2 份检查点，日志、原始轨迹和 PPO 审计全部保存在该 D1 运行目录。

```bash
screen -dmS agl-ppo-baseline-four-1p7b-20260910-01 env \
  AGL_TRAIN_TAG=ppo-baseline-four-1p7b-20260910-01 AGL_GPUS=0,1,2,3 AGL_TRAIN_PORT=18281 \
  bash /media/ubuntu/D1/zsj/agent-lightning-main/agent-lightning-main/examples/multiturn_ppo/run_four_gpu.sh \
  --steps 32 \
  --train-file /media/ubuntu/D1/zsj/agent-lightning-runtime/data/swe-smith-training/exceptiongroup/train.jsonl \
  --val-file /media/ubuntu/D1/zsj/agent-lightning-runtime/data/swe-smith-training/exceptiongroup/val.jsonl \
  data.val_batch_size=6 trainer.total_epochs=4 trainer.test_freq=8 trainer.save_freq=8
```

原单卡试跑 `training-ppo-baseline-1p7b-20260910-01` 在记录到第 7 步后，
按用户改为四卡的要求主动中断；日志保留，其退出码 1 不代表四卡程序失败。
本次四卡作业尚非全量 SWE-bench 评测，仅使用已缓存的 exceptiongroup 子集。
