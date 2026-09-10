# SWE PPO 稳定性诊断与修复试验

## 已确定的事实

原运行 `ppo-baseline-four-1p7b-20260910-01` 使用 Qwen3-1.7B、4 卡、32/6 个
exceptiongroup 任务、终局二值奖励、gamma=lambda=1、actor/critic LR=1e-5。
无 critic 预热、无 reference KL loss，启用全批 action token 优势标准化。

1. 日志明确显示 critic 的 `score.weight` 和 `score.bias` 为随机初始化。
2. 检查第 1、5、8、16 步的真实 action token，所有 returns 均为 0，且
   `raw_advantages + values` 的最大绝对误差为 0。这是完整终局 GAE 在该配置下的
   正确望远镜求和结果 `A = R - V`，未发现跨轮串接算错或混入占位行。
3. 原始优势标准差从 2.09325 降到 0.0221513，但标准化后始终约为 1。
   第 16 步把 critic 的剩余误差放大约 45.14 倍，仍驱动 actor 更新。
4. task batch=4 并不表示每个 global step 只更新一次 actor：例如 32 个调用行、
   global mini-batch=4 会做 8 次 optimizer step。PPO clipping 逐 mini-batch 生效，
   不能等同于对原始预训练模型的整体漂移约束。
5. 原模型训练前验证已经为 0；训练后的输出退化是额外问题，修复数值稳定性
   并不自动提供成功修复轨迹。

| 原运行 step | 原始优势标准差 | 标准化后标准差 | 训练奖励 |
| --- | ---: | ---: | ---: |
| 1 | 2.093249 | 1.000000 | 0 |
| 5 | 0.456521 | 1.000000 | 0 |
| 8 | 0.189459 | 1.000000 | 0 |
| 16 | 0.022151 | 0.999990 | 0 |

中期抽查：最早 10 个已完成轨迹的 79 次调用有 77 次可解析 action；最新 10 个
轨迹的 80 次调用全部因 length 截断而无 action。train/val 的已有 reference controls
奖励为 1、bug controls 奖励为 0，暂未看到这些对照的判分异常。

以上能确认优势噪声放大机制及行为退化同时存在，不能单凭日志把每个配置项的
独立因果贡献分离出来；下面是组合稳定化试验，不是严格的单因素消融。

进一步排查：veRL 的 vLLM 服务已将 HF generation config 覆盖为 top_p=1、top_k=-1，
网关训练 temperature=1 与 actor log-prob 计算一致。未发现上述采样参数错配。
对全部 32 train + 6 val 运行独立 reference positive control，38/38 全通过。
正确代码能通过当前判分；对照只用于诊断，从未提供给 actor 或加入训练轨迹。
对照日志：`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/ppo-stability-reference-controls-20260910-01`。

## 修复配置

独立入口 `run_smith_stable.sh`，从原始 Qwen3-1.7B 开始，保留原 PPO clipped
objective、独立 critic、GAE 和终局二值奖励。没有删除零奖励轨迹、伪造正奖励，
也没有把 PPO 改为 GRPO。

- 关闭优势标准化，保留 `R - V` 的原始尺度。
- 前 4 个 global steps 仅更新 critic；veRL 的边界为 `step >= critic_warmup`，
  因此设置 `critic_warmup=5`，第 5 步才更新 actor。
- actor LR 从 1e-5 降为 1e-6；critic LR 保持 1e-5。
- actor 增加相对冻结预训练 reference 的 `low_var_kl` loss，系数 0.02；
  KL 不混入环境二值奖励。
- 同样 32 train / 6 val、8 turns、768 response tokens、12288 context。
- 12 步、最多 2 epochs，每 4 步验证并保存；有效 actor 更新覆盖第 5–12 步。
- 新增 raw/actor advantage std、放大系数、actor 是否更新的指标，并保存 reference
  log-prob 审计。验收脚本同时检查 critic 预热期的审计和非标准化优势。

## 实际启动命令

```bash
screen -dmS agl-ppo-stability-four-1p7b-20260910-01 env \
  AGL_TRAIN_TAG=ppo-stability-four-1p7b-20260910-01 \
  AGL_GPUS=4,5,6,7 AGL_TRAIN_PORT=18282 \
  bash /media/ubuntu/D1/zsj/agent-lightning-main/agent-lightning-main/examples/multiturn_ppo/run_smith_stable.sh
```

运行目录：`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-ppo-stability-four-1p7b-20260910-01`。
原运行保留在 GPU 0–3，本次使用另一组空闲四卡做有界对照。
每次发布源代码均先执行 WSL rsync `-anizc` 预览，再 `-aizc` 同步；排除 `.git`、
环境、缓存、数据、日志、outputs、checkpoints 和权重，未使用 `--delete`。

## 验收状态

回归测试：`tests/verl` 与 `tests/server/test_triplet_preserve.py` 共 58 passed。
核心改动 Pyright 0 errors，Ruff 和格式检查通过。
初始验证：0/6 成功，平均每轮 93.85 response tokens。
12/12 步已正常结束，`run.exit=0`，GPU 4–7 已释放。
`verify_run.py` 对所有 12 步的跨轮张量、有效 token、reference log-probs、全部
actor/critic rank 检查点验收通过。每个 actor rank 的 optimizer step=64，
每个 critic rank=96；第 4 步的 actor optimizer 尚无状态，critic 为 32，确认预热有效。

| 指标 | 修复试验结果 |
| --- | --- |
| 训练 episodes / action calls | 48 / 377 |
| 实际 actor 更新 global steps | 5–12，共 8 批 |
| 第 12 步平均响应长度 / 截断率 | 85.46875 tokens / 0 |
| 第 12 步原始优势标准差 / 放大系数 | 0.0394148 / 1 |
| 最终验证平均每轮响应长度 | 90.77778 tokens |
| 训练成功率 / 各轮验证成功率 | 全部为 0 |
| 第 7–12 步可解析 action | 184 / 186，98.92% |
| 第 7–12 步 length 格式错误 | 0 / 186 |

第 6 步曾有 2 个任务重复生成转义字符，导致 16/31 次调用因 length 截断无法解析；
之后未持续恶化。该失败保留在原始轨迹中，不能把结果描述为所有格式问题都消失。
原运行在同期仍持续 768 tokens / 100% 截断，新配置的持续崩坏明显缓解。

**范围限制：**这是组合稳定化短测，不是训练有效性的验收，也不是单因素因果消融。
验收 JSON 明确记录 `verification_scope` 和 `task_success_observed=false`。
当前 Qwen3-1.7B + 8 轮预算尚未产生成功轨迹；不能断言模型在更充分预算或其他任务上
也无法成功。下一步应先用原始权重做更充分交互预算的能力评估，建立可获得正奖励的
任务设置，再决定是否需要工具使用 SFT / 训练课程。没有启动全量训练。
此前全量镜像及逐仓库评分适配缺口也仍存在。
