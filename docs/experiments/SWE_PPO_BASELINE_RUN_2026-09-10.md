# Qwen3-1.7B SWE-smith PPO baseline 运行记录

**状态更新：用户随后要求改用四卡。单卡运行记录到第 7 步后被主动中断，`run.exit=1`
来自这次中断；GPU 已释放，原始日志保留。以下配置和启动命令记录的是被替代的单卡试跑。**

2026-09-10 16:31:39（北京时间）启动。源码为内层仓库 main，提交
`0ba4a37e33c5dffce3bec0558014f4973225b342`。
启动前用 WSL `rsync -anizc` 校验本地/服务器源文件一致，无需再次传输。
本次未修改算法或训练程序。

## 实验配置

| 参数 | 本次设置 |
| --- | --- |
| GPU | 物理 GPU 4，1 × A800 80GB |
| Actor / critic 初始化模型 | Qwen3-1.7B，各自独立参数 |
| 算法 | 多轮 PPO，GAE，普通 critic |
| 数据 | 缓存 exceptiongroup 子集，32 train / 6 validation，无 data_id 重叠 |
| 训练规模 | 32 global steps，1 epoch |
| Task batch / rollout.n | 1 / 1 |
| Actor / critic mini-batch | 每次模型调用 1 行；micro-batch=1；PPO epochs=1 |
| Agent budget | 每个任务最多 8 次模型调用，单次最多 768 response tokens |
| Context / prompt limit | 12288 / 11520 tokens |
| Actor / critic 学习率 | 均为 1e-5，weight decay=0 |
| PPO clipping | Actor 0.2，critic value clip 0.5，gradient clip 1.0 |
| Reward / GAE | 终局二值成功，gamma=1，lambda=1 |
| KL / entropy | 均为 0 |
| 采样温度 | train=1.0，validation=0.7；validation 每题采样 1 条 |
| 训练实现 | FSDP，CPU 参数/optimizer offload，gradient checkpointing；vLLM TP=1 |
| 评测 | 训练前评测，之后每 8 步及最后一步评测全部 6 条验证样本 |
| 保存 | 每 8 步，actor/critic 各保留最近 2 份检查点 |
| 日志 | 每步 metrics、原始轨迹、工具输出、独立评分、PPO 审计张量 |

本次用于观察完整一轮训练行为。数据仅来自一个缓存仓库，不是正式的完整 SWE-bench。
验证采样有随机性；6 条验证数据的分数不宜作稳定统计结论。
上一轮该模型单步验收的训练/验证任务成功率均为 0。

训练数据 SHA256：`5832d962a81c99ede7aa5583ed47b679e244644527072ad846eca8ed2b6ea86e`。
验证数据 SHA256：`ce5e2c18dd4409676f3be439e88fa58141ad2b2af36a5ae31ebe8681df2f63bc`。

## 运行位置和实际命令

SSH：`ubuntu@A800`。
工作目录：`/media/ubuntu/D1/zsj/agent-lightning-main/agent-lightning-main`。
Screen：`agl-ppo-baseline-1p7b-20260910-01`。
日志目录：`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-ppo-baseline-1p7b-20260910-01`。

```bash
screen -dmS agl-ppo-baseline-1p7b-20260910-01 env \
  AGL_TRAIN_TAG=ppo-baseline-1p7b-20260910-01 AGL_GPU=4 AGL_TRAIN_PORT=18281 \
  bash /media/ubuntu/D1/zsj/agent-lightning-main/agent-lightning-main/examples/multiturn_ppo/run_smith_training.sh \
  --steps 32 \
  --train-file /media/ubuntu/D1/zsj/agent-lightning-runtime/data/swe-smith-training/exceptiongroup/train.jsonl \
  --val-file /media/ubuntu/D1/zsj/agent-lightning-runtime/data/swe-smith-training/exceptiongroup/val.jsonl \
  data.val_batch_size=6 trainer.total_epochs=1 trainer.test_freq=8 trainer.save_freq=8
```

`resolved-config.json`、`datasets.json` 和 `provenance.json` 为运行时实际配置记录。
结束状态看 `run.exit`：0 表示训练程序正常结束。成功率看 `training/reward` 与 `val/reward`。

## 仓库示例的源码默认值

这些是当前 checkout 中示例入口的默认配置，不代表所有配置都已在当前 A800 环境验证。
每行卡数均为单机训练作业总卡数；`batch × n` 是每批任务数 × 每任务采样轨迹数。

| 示例 | 默认模型 | GPU 数 | 算法 | Task batch × n |
| --- | --- | ---: | --- | --- |
| GSM8K | Qwen2.5-1.5B-Instruct | 1 | GRPO | 8 × 4 |
| Calc-X | Qwen2.5-1.5B-Instruct | 1 | GRPO | 32 × 4 |
| Multimodal QA | Qwen3.5-2B | 1 | GRPO | 8 × 4 |
| SWE-smith FSDP | Qwen3.5-9B | 4 | GRPO | 16 × 8 |
| SWE-smith Megatron | Qwen3.5-9B | 4 | GRPO | 32 × 8 |
| LLM-in-sandbox | Qwen3-4B-Instruct-2507 | 4 | RLOO | 8 × 8 |
| Search-R1 | Llama-3.2-3B-Instruct | 8 | GRPO | 512 × 4 |
| ScienceWorld | Qwen2.5-7B-Instruct | 8 | GRPO | 32 × 4 |
| 本次新增 PPO baseline | Qwen3-1.7B | 1 | PPO + critic | 1 × 1 |

原 SWE-smith FSDP 默认 vLLM TP=1、最大 context=81920、学习率=1e-6；Megatron 默认 TP=2。
我们当前 PPO 的普通 critic、上下文预算、采样数和 mini-batch 都不同。
可变轮数的单卡 PPO 已验收；直接把本入口 GPU 数改为 4 并不能自动解决完整轨迹的尾批整除问题。
