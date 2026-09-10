# CAPO PPO 直接移植验收

本次按用户要求以复制为主，源快照为
`Jay-Juice/Agent-R1@e8407baea32fb36191f029f0a4666bf681bdf1ca` 的 CAPO 项目。
复制约 3,750 行 Python 源码（包含其完整 core_algos 文件中的其他算法），
核心兼容层 `capo_ppo.py` 为 97 行，另有已有 trainer 的路由接线、启动脚本、来源生成器及测试。
没有重写 PPO loss、Actor/Critic 更新循环或跨轮 token GAE。

## 已完成的检查

- 61 项 veRL 集成及事件保留回归测试通过。
- 新适配和修改过的 trainer/entrypoint 的 Pyright：0 errors。
- Ruff、格式、shell 语法、版权头及 `git diff --check` 通过。
- 来源 manifest 校验通过；完整复制的 GAE/core loss 文件与源文件哈希相同。
  Actor/Critic 只调整 loss import，并加有明确的修改声明。
- 运行时日志记录 Actor/Critic 来自 `agentlightning.verl.vendor.capo`，Actor 更新直接继承复制函数。
- 当前安装环境与 CAPO 旧目录的 `masked_mean`、`masked_var`、`masked_whiten`、
  `logprobs_from_logits`、两种 entropy 函数、`clip_by_value`，函数 AST 全部相同。
- 保留原项目 Apache-2.0 许可证，未用 Microsoft 版权头替换复制源码的原版权声明。

## 四卡短测及恢复

日志共同前缀：`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-`。

| run | 模型与设备 | 结果 |
| --- | --- | --- |
| `capo-config-20260910-01` | 仅配置解析 | exit 0，token_gae / gamma=.99 / whitening=true / backend=capo |
| `capo-smoke-20260910-01` | Qwen3-0.6B，GPU 0–3 | 两步完成，exit 0，验证器通过 |
| `capo-resume-20260910-01` | 同模型/设备，从 step 2 恢复 | step 3 完成，exit 0，验证器通过 |

每个训练 batch 有 4 个不等长合成算术 episode，共 9 次真实模型调用。
CAPO 复制的 padding 将它补成 12 行，未丢真实调用；每步 3 个优化器 minibatch。
此处奖励属于合成算术，不是 SWE 成功率。

| global step | 真实生成 token | 奖励和 | Actor grad norm | Critic grad norm | 四个 rank 的 Actor/Critic optimizer 累计更新 |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 398 | 1 | 16.2000 | 170.1626 | 两步后统一核查 |
| 2 | 533 | 1 | 24.7988 | 120.5465 | 每个 rank、每个组件均为 6 |
| 恢复后的 3 | 336 | 1 | 16.1831 | 48.2888 | 每个 rank、每个组件均为 9 |

验证器重新调用复制的 GAE，检查 audit 的 advantage/return，确认补齐行标记和完整 episode，
并检查每个 rank 的 model、optimizer、extra-state 文件与计数。
训练计算 step 时间分别为 39.80、19.68、38.14 秒；不含模型初始化、检查点保存等时间，
不能将该时间与不同任务/不同预算的旧运行直接当作速度对照。

恢复命令：

```bash
AGL_TRAIN_TAG=capo-resume-20260910-01 AGL_GPUS=0,1,2,3 AGL_TRAIN_PORT=18381 \
  bash examples/multiturn_ppo/run_capo_smoke.sh --steps 3 trainer.total_epochs=3 \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path=/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-capo-smoke-20260910-01/checkpoints/global_step_2
```

## SWE 单步验收

运行：`capo-swe-1p7b-20260910-01`，Qwen3-1.7B，GPU 4–7。
采集缓存 exceptiongroup 的全部 32 条训练任务，训练前后各验证固定的 6 条独立任务。
任务 batch / 调用 minibatch=32，每卡 microbatch=1；其余算法配方见 `CAPO_PPO.md`。

```bash
AGL_TRAIN_TAG=capo-swe-1p7b-20260910-01 AGL_GPUS=4,5,6,7 AGL_TRAIN_PORT=18382 \
  bash examples/multiturn_ppo/run_capo_ppo.sh --steps 1 trainer.total_epochs=1
```

运行 exit 0，验证器通过，四张 GPU 已释放。32 条训练轨迹产生 250 次真实模型调用、
21,835 个有效生成 token；CAPO 补齐 2 行到 252 行，没有丢弃真实调用。
四个 rank 的 Actor 和 Critic 优化器各更新 8 次，model/optimizer/extra-state 文件齐全。

| 指标 | 结果 |
| --- | --- |
| 训练前独立验证 | 0/6 |
| 训练轨迹获得成功奖励 | 0/32 |
| 训练后同一验证集 | 0/6 |
| Actor / Critic grad norm | 6.5946 / 110.1424，均有限 |
| Actor PPO clip fraction / reference KL loss | 0.00552 / 0.01034 |
| 训练 step 总耗时 | 550.51 秒，不含初始加载、保存和独立验证 |
| 其中 rollout / Critic update / Actor update | 113.45 / 165.38 / 154.93 秒 |

这次工程验收通过，但没有观察到 SWE 成功率提升，因此没有启动全量训练。
全部终局奖励为零仍可能产生非零梯度：初始 value 预测不为零，GAE/whitening 和 reference KL
会形成更新信号；“optimizer 确实更新”不能等同于“学会修复”。单步、6 条验证任务也不足以判断收敛。

## 范围

这证明复制的 PPO 核心可以接入当前 Agent Lightning 环境，并不意味着复现了 CAPO 论文算法，
也不意味着 SWE 修复能力提升。SWE task/scaffold/预算与旧 ALFWorld 实验不同。
CAPO 自带尾批 response mask 不清零、固定 minibatch 尾批缩放等行为原样保留，见 `CAPO_PPO.md`。
本次没有同时调 SWE prompt/奖励、添加特权信息，或启动全量训练。
