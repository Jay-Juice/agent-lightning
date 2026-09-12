# CAPO 项目 PPO 的直接移植版

2026-09-12 更新：SWE 启动器默认使用 `gamma=1, lambda=1`，并打开
`agentlightning.multi_turn_ppo.capo_strict_padding=true`。
GAE、PPO clipping、KL 和 actor/critic 更新循环的复制源码保持原哈希；
新的兼容层将补齐行的损失权重设为零，并按每个分布式 minibatch 中的真实调用数归一化。
这修正了旧版补齐行参与 critic/KL 更新、短尾批权重偏小的问题，因而不再声称这些行为原样保留。
重现旧配方可显式设置 `algorithm.gamma=0.99` 与
`agentlightning.multi_turn_ppo.capo_strict_padding=false`。
实验效果与工程验收须分开判断；详见 [完整排查记录](SWE_DIAGNOSIS_2026-09-12.md)。

本入口按用户要求优先复制原实现：复制的是 CAPO 项目已经运行过的
`token_gae + vanilla PPO` baseline，不是 CAPO 论文中的动作级概率比算法。
普通 critic 无特权信息。新检查入口使用 Qwen3-4B-Instruct-2507。
初版 1.7B 的四卡短测、恢复及 SWE 单步结果见 [历史验收记录](CAPO_PPO_VALIDATION_2026-09-10.md)。

## 复制范围

来源为本地 `Agent-R1/CAPO/CAPO-main`，Git commit
`e8407baea32fb36191f029f0a4666bf681bdf1ca`。
代码保存在 `agentlightning/verl/vendor/capo/`，保留 Apache-2.0 许可证和原始版权头。

| 文件 | 处理方式 |
| --- | --- |
| `arft_core_algos.py` | 完整复制 CAPO `arft/core_algos.py`，不改 GAE 或 whitening |
| `verl_core_algos.py` | 完整复制 CAPO 自带 veRL core_algos，不改 PPO clipping、KL、value loss 或 loss aggregation |
| `dp_actor.py` | 完整复制 Actor 前向和优化器更新代码，仅将 loss 的 import 指向复制文件 |
| `dp_critic.py` | 完整复制 Critic 前向和优化器更新代码，仅将 core_algos import 指向复制文件 |
| `trajectory.py` | 从 CAPO trainer 原样提取六个函数：valid-data 选择、GAE 路由、critic mask、诊断散射和 world-size padding；仅调整本地 import |
| `reference_run_ppo.sh` | 完整保存原 ALFWorld PPO 配方作为参考，不直接执行旧环境启动器 |

`manifest.json` 记录源文件与复制文件的 SHA256（统一 LF）、每处 import 替换及修改声明。
调整过 import 的两个文件加有明确的修改注释，原版权和许可证完整保留。
启动时会校验复制内容。可以用 `scripts/vendor_capo_ppo.py --source /path/to/CAPO-main`
从同一来源重新生成，不应手工改 vendor 里的算法。

## 必要的适配

`capo_ppo.py` 与 `capo_padding.py` 负责接线、检查和补齐权重：

- 将 Agent Lightning 的 rollout ID、turn index 映射为 CAPO 的 trajectory UID、step index。
- 调用复制的 padding、GAE 路由和 Actor/Critic 类。
- CAPO Actor 的 log-prob 推理返回 `(log_probs, entropys)`，当前 veRL FSDP 外层要求字典；
  兼容子类转换这个返回值，padding 子类在调用原更新循环前准备损失权重。
- `raw_advantages` 仅用于审计，从 returns 与 values 计算；不替换复制函数给 Actor 的优势。

模型加载、FSDP、vLLM 服务、检查点读写继续使用当前已安装的 veRL 0.7.1。
没有替换共享 Python 环境，也没有将整个旧 ALFWorld 训练器及其依赖搬入。
SWE 任务通过现有 Agent Lightning gateway/controller 和 Docker agent 执行。
完整 episode 校验、终局奖励放置和日志仍复用现有适配。检查入口另外启用通用编辑提示、
精确替换工具和源码提交检查；最终评分仍在独立的新容器执行原测试，不增加中间奖励。

因此这是 **CAPO PPO 核心的直接移植**，不是旧环境、旧数据、旧任务预算的完全复现。

## CAPO 补齐与权重适配

此版本不调用 `distributed_ppo.register_in_worker`，不覆盖当前 veRL 的全局 aggregation。
使用 CAPO 原来的补齐函数：补齐时复制已有行并标记 `is_pad`；
GAE 排除这些行，补齐行的 advantage/return 为 0，**response mask 不清零**。
原更新函数会让这些行参与 critic loss 和 reference KL。严格兼容层保留这些行的 forward，
但把其损失权重置零；真实行按跨 rank 的实际调用数归一化，包括不足一批的尾部。
只有显式关闭 `capo_strict_padding` 时才保留原虚拟行损失行为。

日志 `capo/padding_rows` 与每步审计 JSON 的 `is_pad` 会记录它；
验证器从真实 episode 统计中排除补齐行，同时检查复制 GAE 的完整输出。
`padding/real_calls` 和 `padding/ignored_calls` 是每 rank 计数的平均值。
单 minibatch 时的 on-policy 分母选择仍为 CAPO 原代码行为。

## 配方与启动

`run_capo_ppo.sh` 使用 CAPO PPO 核心配方及上述适配：

- token GAE，gamma=1，lambda=1，有效 token whitening 开启。原 .99 按生成 token 折扣，
  会严重削弱长轨迹上的终局奖励传播。
- Actor LR=1e-6，Critic LR=1e-5，AdamW weight decay=.01。
- reference KL `low_var_kl`，系数 .001；不将 KL 加入环境奖励。
- PPO clip low/high=.2/.2，dual clip=3，entropy=0，actor/critic PPO epochs=1。
- 关闭不参与目标的 actor 熵统计；SWE 长上下文入口启用已安装 veRL 的 Torch 分块输出层，
  避免一次物化全上下文的词表 logits 及其梯度。没有另写 PPO loss，也没有缩短任务预算。
- 调用内 token mean、调用间 mean；critic warmup=0。

SWE pilot 的明确缩小：当前只有 32 条缓存训练数据，因此默认任务 batch 和调用 minibatch
都为 32（CAPO 原配方为 128），4 张 GPU，每卡 microbatch=1（原脚本为 4）。
通用入口仍可配置旧 SWE 预算。`run_checked_swe_ppo.sh` 使用诊断通过的 4B、32 轮、
4096 response tokens、65536 context、32000 字符观察，并开启编辑和提交检查；
每任务一个 rollout，四卡 FSDP。这些都不能视为与 ALFWorld 原实验相同。

从部署后的 Agent Lightning 仓库根目录运行：

```bash
# 两步工程短测：Qwen3-0.6B，四个不等长任务，每步 9 次真实调用补齐为 12 行。
AGL_TRAIN_TAG=capo-smoke-my-run AGL_GPUS=0,1,2,3 AGL_TRAIN_PORT=18381 \
  bash examples/multiturn_ppo/run_capo_smoke.sh

# 修复后的 SWE pilot：Qwen3-4B-Instruct-2507，32 train / 6 val，默认两步。
AGL_TRAIN_TAG=capo-swe-my-run AGL_GPUS=0,1,2,3 AGL_TRAIN_PORT=18381 \
  bash examples/multiturn_ppo/run_checked_swe_ppo.sh

# 将新日志与检查点放入独立目录，延续相同 smoke 配置到 step 3。
AGL_TRAIN_TAG=capo-resume-my-run AGL_GPUS=0,1,2,3 AGL_TRAIN_PORT=18381 \
  bash examples/multiturn_ppo/run_capo_smoke.sh --steps 3 trainer.total_epochs=3 \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path=/absolute/path/to/previous-run/checkpoints/global_step_2

python examples/multiturn_ppo/verify_run.py --run /absolute/path/to/training-TAG
```

新版本通过 `backend=capo` 明确选择；旧 `backend=agl` 路径仍可使用。
不能把 `backend=capo` 与旧 `distributed_padding=true` 或关闭 whitening 混用。
默认 pilot 不是全量训练，也不保证复制实现就能解决当前零成功率问题。
