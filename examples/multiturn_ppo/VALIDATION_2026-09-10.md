# 多轮 PPO 首版验收 — 2026-09-10

本地权威源码：`D:\ai project\RL学习\agent-lightning-main\agent-lightning-main`。
部署目录：`A800:/media/ubuntu/D1/zsj/agent-lightning-main/agent-lightning-main`，SSH 用户 `ubuntu`。
所有运行日志、审计张量、模型与 optimizer 检查点仅保存在 D1 的 runtime 目录。

## 实测结果

| 实验 | global steps | 训练 episodes / 模型调用 | action tokens | actor / critic optimizer steps | 结果 |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B 两轮算术 | 1、2 | 每步 2 / 4 | 240、242 | 8 / 8 | 退出码 0，验收通过 |
| Qwen3-0.6B 从 step 2 恢复 | 3 | 2 / 4 | 265 | 12 / 12 | 退出码 0，验收通过 |
| Qwen3-1.7B SWE-smith | 1 | 1 / 8 | 464 | 8 / 8 | 退出码 0，验收通过 |

运行目录前缀：`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/`。

- `training-ppo-smoke-20260910-01`
- `training-ppo-resume-20260910-01`
- `training-ppo-smith-20260910-01`

每个目录都有 `verification.json`。实测训练中未删除任何调用，没有训练 token 截断或失败 rollout。
所有有效 token 上的 values、log-probs、advantages 和 returns 都为有限值。
γ=λ=1 且无 KL reward 时，每个 episode 所有 action tokens 的 returns 等于最终奖励；
终局奖励仅在最后一轮最后一个 token 上出现。奖励为 0、1 的轨迹都经过验证。

| 实验 step | actor grad norm | critic grad norm | 训练 episode 奖励总和 |
| --- | ---: | ---: | ---: |
| 算术 1 | 26.8821 | 366.2362 | 2 |
| 算术 2 | 18.6461 | 128.3271 | 1 |
| 恢复 3 | 92.2482 | 120.6849 | 0 |
| SWE 1 | 22.4676 | 139.3712 | 0 |

梯度数值为 veRL 报告的 grad norm，不能用来推断训练已收敛。
额外对比了算术 step 1 和 step 2 的保存参数：actor 和 critic 的第一层 Q/K/V 投影均发生变化。
例如 Q 投影最大绝对变化分别为 `3.712857e-5` 和 `3.978983e-5`。

恢复日志明确加载了两个模型、两个 optimizer、RNG 和 LR scheduler，global step 从 2 继续到 3。
因此恢复验证包含 critic 的完整训练状态。

SWE 的训练前验证、训练 episode、训练后验证均执行了 8 轮，独立 pytest 评分均未解决任务。
训练样本 F2P 为 0/36、P2P 为 53/53；验证样本 F2P 为 0/63、P2P 为 26/26。
训练最大 prompt 为 2495 tokens、最大单轮 response 为 69 tokens。
本次只有缓存的 exceptiongroup 子集，证明流程可运行，**没有证明 SWE 解题能力提升**，
也不是 SWE-bench Lite/Verified 的正式分数。

## 配置与环境

普通 PPO、独立 critic、actor/critic 都只访问本轮可见历史。
`adv_estimator=gae`、transition traces、vanilla clipped PPO、γ=λ=1、终局二值奖励、
全 batch action-token advantage whitening、KL=0、entropy=0、两个学习率均为 1e-5。
每条调用一个 mini-batch，保留全部轮次。没有 privileged critic 或额外稳定性技巧。

实测环境：Python 3.12.14，torch 2.9.0+cu129，veRL 0.7.1，vLLM 0.12.0，
Ray 2.58.0，Transformers 4.57.6，Agent Lightning 1.0.1。
恢复及 SWE 运行的 `provenance.json` 记录了软件版本和当时源文件 SHA256。
GPU 验收后的修改限于文档、版权头、类型检查兼容注释和等价的空值类型收窄，
没有修改成功轨迹的 PPO 数学或训练配置。

## 实际执行命令

服务器工作目录均为部署目录。运行器会自动激活 D1 环境。

```bash
screen -dmS agl-ppo-smoke-20260910-01 env \
  AGL_TRAIN_TAG=ppo-smoke-20260910-01 AGL_GPU=4 \
  bash examples/multiturn_ppo/run_training.sh --steps 2

screen -dmS agl-ppo-resume-20260910-01 env \
  AGL_TRAIN_TAG=ppo-resume-20260910-01 AGL_GPU=4 AGL_TRAIN_PORT=18281 \
  bash examples/multiturn_ppo/run_training.sh --steps 3 \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path=/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-ppo-smoke-20260910-01/checkpoints/global_step_2

screen -dmS agl-ppo-smith-20260910-01 env \
  AGL_TRAIN_TAG=ppo-smith-20260910-01 AGL_GPU=5 AGL_TRAIN_PORT=18381 \
  bash examples/multiturn_ppo/run_smith_training.sh

source /media/ubuntu/D1/zsj/agent-lightning-runtime/admin/activate-agent-lightning-d1.sh
python -m pytest -q tests/verl tests/server
python examples/multiturn_ppo/verify_run.py --run /media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-ppo-smoke-20260910-01
python examples/multiturn_ppo/verify_run.py --run /media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-ppo-resume-20260910-01
python examples/multiturn_ppo/verify_run.py --run /media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-ppo-smith-20260910-01
```

同步使用 WSL `rsync -anizc` 预览，检查后再执行 `rsync -aizc`，SSH transport 为
`/mnt/c/Windows/System32/OpenSSH/ssh.exe -l ubuntu`。源目录为
`/mnt/d/ai project/RL学习/agent-lightning-main/agent-lightning-main/`。
排除了 `.git`、Python/tool caches、venv、data、datasets、checkpoints、outputs、logs、wandb
和模型张量文件，未使用 `--delete`。未编辑远端源文件，未停止任何已有训练任务。

## 验收范围

数学/数据/API 测试包含跨轮 GAE 的独立 oracle、不同 γ/λ、padding mask、乱序行、
缺失轮次与终局、重复 prompt、仅终局奖励、whitening 不改变 critic target，以及已有 GRPO 回归。
首轮专项测试 49 项通过；随后 veRL 与服务器回归测试 71 项通过。
新增代码 Ruff、格式、版权头及静态类型检查一并检查。

首版默认单卡；更大 mini-batch 或多卡要求完整调用数可整除相应 batch 大小。
未实现任意分布式尾批、异步跨策略采样、未结束 episode 的 bootstrap、长时间正式训练、
全量 SWE benchmark 或 PI critic。后续 PI 对照应复用本实现的 token、reward、GAE、batch 和 loss 约定。
