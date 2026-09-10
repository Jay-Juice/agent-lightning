# 当前 SWE PPO 的来源、参考实现对照与零奖励诊断

日期：2026-09-10。被审计版本：`2ae3870`。本轮未启动 GPU 训练、未修改训练参数。

**结论：当前版是基于 Agent Lightning / veRL 新增的跨轮 token PPO；不是上游 SWE 示例的原样运行，也不是旧 Agent-R1 或 CAPO 实验的完整复现。跨轮 GAE 的独立数值对照通过，但当前任务仍没有形成有效修复，不能用工程验收通过来代替 baseline 有效。**

## 1. 哪些来自原仓库，哪些是本项目新增

以 Git 导入快照 `6e56fd6` 为原仓库依据，不以函数名 `run_ppo` 判断实际算法。

| 部分 | 来源与当前行为 |
| --- | --- |
| Ray/FSDP worker、Actor/Critic 前向及优化器、value clipping、token ratio PPO clipping、保存恢复 | 复用 veRL；不是从零重写 PPO optimizer |
| Agent Lightning trainer、网关、rollout adapter、controller | 复用后扩展 |
| `multi_turn_ppo.py` | 新增：完整 episode 校验、只在最终动作的末 token 放终局奖励、按 episode/turn/token 跨行递推 GAE、审计张量 |
| `distributed_ppo.py` | 新增：四卡推理补行并还原；更新时补行 loss mask 为零；按全局真实调用数求平均。会覆盖选定的 loss aggregation，因此不能说底层 loss 路径完全未改 |
| trainer / events 改动 | 新增开关，保存所有模型调用，避免 GAE 前丢弃中间行或给同一 episode 重复计入终局奖励 |
| SWE 本地 Docker agent / grader / runners | 本项目纳入并适配的离线 pilot，使用固定 prompt helper；不是上游 SWE worker 的完整原样复制 |
| 关闭 whitening、actor LR 1e-6、KL 0.02、warmup 对照 | 后续稳定性实验配方；尤其 whitening 和 KL 会影响学习，不属于纯性能优化 |

上游快照 `examples/swe_smith/train_smith_agent.py` 的示例实际为 **GRPO**：
`adv_estimator=grpo`，默认 Qwen3.5-9B，16 个任务、每任务 8 条 rollout、4 GPU。
当前是 **Qwen3-1.7B、learned critic PPO、4 个任务、每任务 1 条 rollout**。
这里只借用了上游四卡规模，不能据此称为复现上游 SWE 配方。
当前数据也是 SWE-smith exceptiongroup 32/6 pilot，不是完整 SWE-bench 分数。

## 2. 与用户旧项目到底是否一致

依据 A800 已保存的实际配置：

- 旧 Agent-R1：`Agent-R1-qwen3-1.7b/outputs/2026-09-05/15-20-16/.hydra/config.yaml`。
- CAPO 项目 PPO baseline：`CAPO/runtime/capo_q17_ppo_mem_0907/resolved_config.yaml`。
- CAPO 最小 P1：`CAPO-P1-minimal/runtime/capo_q17_p1_min_0908/resolved_config.yaml`。
- 当前：`training-ppo-fast-four-1p7b-20260910-01/resolved-config.json`。

前三个路径的共同前缀为 `/media/ubuntu/D1/zsj/agent-r1/Agent-R1/`；最后一个在 Agent Lightning runtime 的 `logs/` 下。

| 项目 | 旧 Agent-R1 1.7B step 实验 | CAPO 项目的 PPO baseline | 当前 SWE fast PPO |
| --- | --- | --- | --- |
| 优势与 Critic 监督单位 | 一次完整动作；开始位置 V，优势广播 | 每个生成 token；跨轮 token GAE | 每个生成 token；跨轮 token GAE |
| gamma / lambda | 1 / 1 | 0.99 / 1 | 1 / 1（SWE 初始配置已与用户确定） |
| 优势标准化 | 按动作标准化 | 按有效 token 标准化 | 关闭；首次 SWE 退化实验是开启的 |
| Actor / Critic LR | 1e-6 / 1e-5 | 1e-6 / 1e-5 | 1e-6 / 1e-5；首次 SWE Actor 为 1e-5 |
| reference KL loss 系数 | 0.001 | 0.001 | 0.02；首次 SWE 未开启 |
| PPO clip low / high / dual clip | .0003 / .0004 / 10 | .2 / .2 / 3 | .2 / .2 / 3 |
| 每个 rollout batch 的任务数 | 128 | 128 | 4 |
| optimizer minibatch 的单位 | 128 条完整轨迹 | 128 条动作记录 | 4 条动作记录 |
| warmup | 0 | 0 | 0；另有历史 4 步 critic-only 消融 |
| Actor/Critic loss aggregation | 调用内 token 平均、调用间平均 | 同左 | 同左，增加四卡尾批零权重归一化 |
| Critic 的特权信息 | 依旧实验组别 | baseline 无；P1 有 | 无 |

旧 Agent-R1 的 minibatch 单位由 `trajectory_batching.py` 分组与自定义 worker 决定，不能直接把配置数值 128 与 CAPO 的 128 当成同义。
当前四任务产生约 32 个调用时，每个 global step 会有约 8 次 optimizer update；并非整个 batch 只更新一次。
相同数量的 GPU 不会自动形成相同有效训练 batch。

还必须区分 **CAPO 仓库里的 PPO baseline** 与 **CAPO 算法本身**。
前者实际是 `token_gae + vanilla`；后者在论文中使用动作边界的价值/优势以及带长度校准的动作级概率比。
当前代码只有前者类型的 token PPO，没有实现 CAPO 的动作级概率比。
来源：[CAPO 作者论文](https://arxiv.org/abs/2604.18401)。

## 3. 独立 GAE 数值对照，而非只看自写单元测试

新增诊断入口 `audit_reference_gae.py`。参考文件直接读取 A800 已有项目源码；
本地与远端文件 SHA256 相同：

- Agent-R1 `core_algos.py`：`3a29d8797d57e48fe42bf500bd098bc30cc320d40fcbf21c4e3a6ea042f08087`。
- CAPO `arft/core_algos.py`：`5873341434b06a13b9a9676ef673a3b3d5836dfa3e32b90f154e9b4cb2539d00`；CAPO baseline 与最小 P1 的该文件也相同。

从参考文件提取其原有 `compute_token_gae_advantage_return` 和校验函数，在当前同一 torch/veRL 环境执行。
没有把另一项目的整个依赖栈混入训练环境。

- 12 组人工输入：乱序调用、1/2/4 轮轨迹、非连续 action mask、零/非零奖励、FP32/BF16 values；gamma/lambda 分别为 1/1、.99/1、.97/.95。
- 3 个真实保存批次：原退化实验第 16 步、无 warmup 实验第 8 步、fast 实验第 4 步。
- **在 gamma/lambda/whitening 一致时，全部用例对两个参考函数的有效 token advantage 与完整 masked return 最大绝对误差均为 0。**
- 真实 batch 还复现了各自当时保存的 advantage/return。
- 原实验第 16 步：32 行、24,576 个生成 token，实际开启 whitening，与参考同设置输出一致。
- 无 warmup 第 8 步：32 行、2,671 token；fast 第 4 步：29 行、2,519 token。
  后两者实际关闭 whitening，若恢复 whitening，actor advantage 最大变化分别为 5.61563 和 2.69347。
  因此不能把“对齐开关后递推相同”说成“当前实际 actor targets 与旧配方相同”。

比较只针对有效 action advantage：旧函数 whitening 后 padding 值可非零，其 loss 会 mask；当前实现额外将 padding advantage 清零。
首次诊断把这类无效位置也纳入比较，已修正诊断比较范围，未因此更改训练算法。

该结果支持 GAE 数学实现一致，**不能证明整个 PPO 训练、随机数、分布式更新和任务效果完全等价**。
value 前状态对齐也检查了当前真实安装的 `verl/workers/critic/dp_critic.py`，使用 `[-response_length-1:-1]`。

## 4. 零成功率发生在哪里

对原始 Qwen3-1.7B 的 24 轮、32K context、无训练更新评测逐条核查：

- 六题、127 次调用，均为模型版本 0，成功率仍为 0/6。
- **127/127** 次网关请求中的消息与 agent 保存的完整历史一致。
- **127/127** 次 vLLM 返回的实际 prompt token IDs 与该完整历史经 Qwen 模板编码完全一致。
- 127 个动作都通过解析；67 次 shell 返回非零状态。
- 四题导出空补丁；两题只在仓库根创建了错误的 `exceptiongroup.py`，没有修复真正位于 `src/exceptiongroup/` 的代码。

| rollout ID 前缀 | 具体失败 |
| --- | --- |
| `21889706` | 24 次重复查询 `exceptiongroup.__file__`；每次观察已返回真实源码路径，仍没有读取或修改它 |
| `9dc47c53` | 24 次重复同一条存在正则语法错误的 sed 命令 |
| `61db4703` | 直接执行包内 `_suppress.py` 触发相对导入错误，11 次重复；未形成净代码变更 |
| `065dc9fe` | 新建根目录 `exceptiongroup.py`，内容为 `def ExceptionGroup(*args, **kwargs): pass`；遮蔽正常包，pytest 导入失败 |
| `89ab6494` | 新建只有一行类声明的根目录模块，尝试不存在的交互编辑器，7 轮即提交 |
| `cded6ceb` | 多次把 Python 复现脚本直接当 shell 文件运行；还因提交包含 `test_*.py` 被整个拒绝 |

这排除了这些轨迹中“工具观察没传回”与“仅仅 8 轮预算太少”的解释。
失败存在于 PPO 更新之前，因此不能把初始零成功率归因于 GAE；
但六个同仓库任务也不足以证明 Qwen3-1.7B 在所有 SWE 设置下都没有能力。

本地 agent 流程还有需要改进的设计：prompt 中给出无实际意义的 `MY_VAR=value cd ...` 例子，被模型照抄；
推荐创建复现脚本，却未清楚要求放在 `/tmp`，而 patch 导出会拒绝任何 `test_*.py` 并拒绝整个补丁。
这些会消耗探索预算或造成额外拒绝，不应统称为模型大小问题。
不过上述被拒绝案例本身也没有有效源代码修复，不能声称改该规则就能使它成功。

评分链以前的 38/38 正确修复对照通过，说明这个子集可以拿到正奖励。
本轮读取的两个错误根模块让 pytest 启动失败，是候选改动破坏了依赖导入，不是“任何补丁都无法通过”的证据。
此前正对照不等于已证明所有 grader 边界情形正确，也不等于模型自行解出了题目。

## 5. 为什么原 PPO 又把输出训坏

已有审计在终局全零、gamma=lambda=1 的批次中验证 `raw advantage = -V`、`return=0`。
初始 Critic 的 score head 为新初始化；原实验第 16 步 raw advantage std 已降至约 .02215，
whitening 仍将其拉回约 1，相当于放大约 45 倍。小任务 batch、多次 mini update、
Actor LR 1e-5 且无 reference KL，使该阶段更新存在偏离任务目标的风险。

本次独立对照特别确认：**已经退化的第 16 步 GAE 数值也与旧 token GAE 相同**。
因此现有证据不支持“跨轮 GAE 写错导致全部失败”的结论。
关闭 whitening、降低 LR、增加 KL 的组合缓解了持续输出退化，但没有产生修复成功；
没有进行足以确定三项各自因果贡献的单因素实验。

## 6. 后续应怎样建立可解释的 baseline

1. 将当前 fast/stable 明确视为稳定性和执行速度实验，不作为“与 CAPO PPO 完全一致”的 baseline。
2. 先用未训练模型检查 SWE agent 是否会定位真实文件、编辑源代码、正确运行复现并提交有效补丁；简化 prompt，明确复现脚本位置。保留独立评分和终局二值奖励。
3. 固定 agent 流程后，单独配置与 CAPO **PPO baseline** 对齐的 token PPO 对照，明确哪些 SWE 设置有意不同。gamma=1 是此前已确定的 SWE 设计，不应为凑一致而无解释地改回 .99。
4. 增大任务采样 batch、明确 optimizer minibatch 的调用/轨迹单位；不能以“显存空闲”推断随意修改学习参数不影响结果。
5. 在观察到可学习的成功轨迹和不退化的短程训练后再考虑全量；之后 PI 与普通 critic 必须保持同一 agent/reward/GAE/loss/budget。

本轮没有自动将 gamma/KL/whitening 恢复旧值，也没有启动新的训练。

## 7. 复现与审计文件

服务器目录：`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/ppo-reference-audit-20260910-01/`。

- `gae-comparison.json`：退出码 0 的独立数值对照、参考源码和真实 batch 哈希。
- `trajectory-check.json`：全部 127 次请求的历史/token 对齐统计及六条失败轨迹摘要。

复现 GAE（在 Agent Lightning 已激活环境运行）：

```bash
python examples/multiturn_ppo/audit_reference_gae.py \
  --reference agent_r1=/media/ubuntu/D1/zsj/agent-r1/Agent-R1/Agent-R1-qwen3-1.7b/agent_r1/trainer/ppo/core_algos.py \
  --reference capo_ppo=/media/ubuntu/D1/zsj/agent-r1/Agent-R1/CAPO/arft/core_algos.py \
  --audit /media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-ppo-baseline-four-1p7b-20260910-01/ppo-audit/step-000016.pt \
  --audit /media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-ppo-no-warmup-four-1p7b-20260910-01/ppo-audit/step-000008.pt \
  --audit /media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-ppo-fast-four-1p7b-20260910-01/ppo-audit/step-000004.pt \
  --output /absolute/path/to/new-audit/gae-comparison.json
```

源码从 Windows 经 WSL rsync `-anizc` 预览、`-aizc` 实际部署，未使用 `--delete`。
本地参考源码通过 SSH 发送的最初方案被自动审批拒绝，未执行；
随后核实服务器已有副本及相同哈希，改用服务器原有文件就地计算，未外发本地参考源码。
