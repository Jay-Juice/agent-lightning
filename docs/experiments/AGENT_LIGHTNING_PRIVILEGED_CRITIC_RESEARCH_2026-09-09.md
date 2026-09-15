# Agent Lightning 上重建 Privileged Critic 与扩展 SWE benchmark：调研记录

日期：2026-09-09（Asia/Shanghai）。范围：本地项目与参考文档、关键训练数据链、官方 benchmark 文档。本文是实现前调研，不是已经实现、部署或训练成功的报告。

## 1. 结论与建议路线

**适合在这份 Agent Lightning 中重新实现，但需要先补齐面向多步 PPO 的数据语义，再加入 Critic 独立输入。不能把内置 SWE-smith 的 GRPO 配置改成 GAE 就称为完成迁移。**

建议围绕用户希望扩展的编码任务推进：

1. 用现有 SWE-smith agent/harness 接通编码轨迹与终局 reward，并单独建立 SWE-bench 官方评测出口。
2. 在 Agent Lightning 内实现普通 multi-turn PPO/token GAE，建立可学习的编码 baseline。
3. 在同一实现、相同 reward、Actor prompt 和训练配置上，仅增加 pre-action state → Critic 的通路。
4. SWE-smith 用于训练及独立开发集选模；SWE-bench Lite 用于小规模工程验证，Verified 作为标准结果对照。测试集不用于 RL reward 或反复挑配置。
5. 如需第二个新环境，ScienceWorld 是当前代码基础上很合适的候选；它已有本地 runner 示例，并有真实 simulator object-tree 接口。SWE 仍按用户提出的方向优先。

GRPO 可以保留为算法对照，但它没有本研究所需的 learned value critic。只有普通 PPO 与 privileged PPO 的匹配对照，才能回答“Critic 多看到环境状态是否有帮助”。

## 2. 项目位置、版本和审查范围

用户已将应用中的项目关联修正为：

```text
D:/ai project/RL学习/agent-lightning-main/
├── A800/
│   ├── AGENTS.md
│   └── A800_PACKAGE_INSTALL_GUIDE.md
└── agent-lightning-main/       # 实际 Python 源码根目录
```

参考资料位于同级 `D:/ai project/RL学习/Agent-R1/`。本次起初工具工作目录是空的 `C:/Users/zsj/Documents/ChatGPT/agent-lightning`；实际源码审查一直基于上述 D 盘目录，未将代码搬到 C 盘。

源码审查前的盘点：

| 项目 | 已核验事实 |
|---|---|
| 包版本 | `pyproject.toml`：`agentlightning==1.0.1`，Python `>=3.12` |
| 文件数量 | 218 个文件，101 个 Python 文件，38 个 Markdown 文件 |
| 下载归档 | `D:/ai project/RL学习/agent-lightning-main.zip` |
| ZIP comment | `218f1f7c0bac0800de4d5a4e5e6f61cf7b5038b4` |
| 解压一致性 | 218 个文件逐字节匹配 ZIP，0 修改、0 缺失、0 额外文件 |
| Git 状态 | D 盘外层和内层均未发现 `.git`；这是源码归档，不是已有完整 Git 历史的 checkout |
| 静态语法检查 | 本地 Python 3.13.5 对全部 101 个 `.py` 执行 `ast.parse`，均通过 |
| 运行验证范围 | 未运行训练、GPU 测试、Docker 评测或完整 pytest；本地默认 Python 缺 torch 和 openai |

ZIP comment 是本地归档标识，本次没有独立证明该 SHA 就是当前远端 main。语法检查也不代表依赖兼容或训练正确。

目录职责：`agentlightning/` 30 个文件，覆盖 schema/client/hooks、Gateway、Controller、veRL；`examples/` 112 个文件，含 SWE-smith、ScienceWorld、Search-R1、Calc-X、GSM8K、LLM-in-Sandbox、multimodal QA；`tests/` 15 个文件；`docs/` 31 个文件；另有依赖锁、安装脚本、CI 和技能资料。本次完成全目录盘点，深入阅读与迁移有关的数据链、配置和测试；没有将每个 vendor 示例都当作已经完成逐行审计。

已梳理 Agent-R1 根目录的九份参考 Markdown：原始 idea、`delta info.md`、09-02 总交接、`handoff.md`、token GAE guide、日志规范、磁盘泄漏复盘、09-06 CAPO 调研、09-09 CAPO P1 handoff。另核对了 CAPO 当前独立 Critic view 与 token GAE 源码。

## 3. 对 idea 的准确理解

动作级形式：

```text
Actor:  π(a_t | H_t)
Critic: V(H_t, Z_t)
```

token 形式：

```text
Actor:  π(a_t,k | H_t, a_t,<k)
Critic: V(H_t, Z_t, a_t,<k)
```

`H_t` 应明确为实际给 Actor 的上下文。窗口化、压缩后的 history 与完整历史不是同一信息集。`Z_t` 从环境控制接口获取，在本次模型调用前捕获；同次生成的全部 token 使用同一份 `Z_t`。执行动作之后的状态归下一次调用。

研究主变量是 Critic 的信息集。Actor rollout、old/ref logprob、Actor update、评测推理保持部署可见输入。Reward 及折扣定义在配对实验中保持一致。

Delta 定义是 `D(C(S_0), C(S_t))`，即相对 episode 初始状态的当前净变化。它不是事件日志，也不是相邻两步差分。Delta 只是表示方式：它可能删掉未变化但仍重要的隐藏信息；空初始 delta 也不是完整初始状态的替代物。

还应区分“真正未观测的状态信息”和“更容易读取的 Actor 历史摘要”。给 Critic Actor 刚写出的代码 diff，可能主要降低状态重建计算成本。仅因为信息来自另一个接口，不能证明其具有条件信息增量。建议后续用 Actor-visible summary、同信息集 current/delta、错配 state 等对照区分机制，而不是第一轮同时加入全部消融。

“保持目标”不等于保证近似 Critic + GAE + PPO clipping 的更新完全无偏，更不保证方差必然下降。原 idea 中的 privileged value function 数学形式已有相关工作，本轮也不建立“首次提出”的 novelty claim。

## 4. 历史结果和偏好如何继承

这些数据均来自交接文档，本次未重读 A800 原始日志：

| 实验 | 文档记录 | 应保留的解释边界 |
|---|---|---|
| 旧 4B step PPO/P1 | 中期 P1 明显领先；最终 baseline seen/unseen 78.57%/86.57%，P1 83.57%/81.34% | 有中期收益信号，最终 unseen 没有超过 baseline；单 seed |
| 旧 1.7B step P1 | 最新 CAPO handoff 记录最终 116/274 | 不能与 CAPO 直接当作单变量比较 |
| CAPO 1.7B token PPO baseline | 最终 252/274，91.97% | gamma=.99、动作行 minibatch、不同 Actor history 等均与旧 step 项目不同 |
| 最新 CAPO 最小 P1 | 09-09 20:26 快照：完成 157/270；验证155为223/274，baseline同点233/274 | 快照中未结束，不能当作最终结果；150时P1领先、155时反转 |

09-02 文档中的通用 all-token EV 混入了 step Critic 没训练的位置；后续 handoff 已纠正。不能继续用该旧表直接证明 Critic 改善。当前 CAPO 的 token EV 也要结合 return variance 和实际数据分布解释。

最新交接明确：CAPO 最小 P1 保留原终止动作训练副本，避免额外 head 初始化、warmup、LR、target-KL 等配方变更。早期“应保留去重”的建议已被该实验的后续决定覆盖。

这项决定约束 CAPO 复现；Agent Lightning 新编码环境不需要人为制造一个原本不存在的终止副本。新实验应先固定共同 baseline 的记录语义，普通/privileged 两组保持一致。不能将旧文档中的启动、监控、删 checkpoint 等历史操作当成本次指令。

## 5. Agent Lightning 1.0.1 的实际数据链

[官方仓库](https://github.com/microsoft/agent-lightning)明确说明 v1.0 完成了架构重构。当前源码路径是：

```text
dataset row
  → entrypoint / trainer 创建 rollout
  → Gateway 保存 rollout
  → Controller 启动 local subprocess 或 K8s Job
  → agent 通过 Gateway 调用模型并执行环境动作
  → model_request / reward / 自定义 events
  → agl_rollout_manager: CompletedRollout + Triplet
  → rollout_adapter: transition 或 trajectory rows
  → veRL DataProto
  → old/ref logprob → values → advantages → Critic / Actor update
```

关键接入点（均相对实际源码根，行号对应本地归档）：

| 文件 / 位置 | 当前行为 | 迁移需要 |
|---|---|---|
| `schemas.py:13` | Event 支持任意 event_type/data | 增加可验证的 state/request 关联协议 |
| `server/routes/events.py` | triplet 格式裁剪 request，并按 prompt token IDs 保留最后一次调用 | 必须处理重试、重复 prompt 与 state 的一一对应 |
| `verl/agl_rollout_manager.py:41,448` | 构造 Triplet；metadata 当前主要保存 server 信息 | 将匹配到的 pre-action state / request ID 传入 Triplet |
| `verl/rollout_adapter.py:368` | 聚合并 padding Actor prompt/response | 增加 Critic 专用输入及完整轨迹元数据 |
| `verl/rollout_adapter.py:502` | transition 模式每一行都分配 final_reward | PPO 路径应保留真实时序 reward，仅终局行放终局奖励 |
| `verl/trainer.py:502` | 在 value/GAE 前按行丢样本以满足 minibatch/update cap | GAE 前保证完整轨迹；不能先随机删中间 turn |
| `verl/trainer.py:575,630` | value inference 和 Critic update 都读取公共 batch | 两个入口都使用同一独立 Critic view |
| `verl/rollout_level_advantage.py` | 每 rollout 选代表行，要求有效 token 优势为常量，再广播 | 适合 outcome-style 路径，不能用于一般 token GAE |
| `verl/per_rollout_loss.py` | 对 rollout token 数与 batch 行数做 loss 权重处理 | 明确选择并固定 loss weighting；不是普通 whitening 的同义词 |
| `server/store.py` | rollout/event/model 保存在进程内字典 | 增加可靠的落盘日志，不能以 Gateway 内存作为研究档案 |

### 5.1 为什么不能仅切 `adv_estimator`

内置 SWE-smith 默认 `adv_estimator=grpo`，基础配置 `enable_rollout_level_advantage=true`。rollout-level helper 会拒绝非恒定 token advantage。普通 PPO 需要先停用该 GRPO 风格广播，并提供真正的多步 GAE。

即使关闭广播，直接调用外部 veRL 的逐行 `compute_advantage` 也不等于 CAPO 的跨 turn token GAE。当前行之间没有递推接口。若使用 transition 模式，同时保留每行重复 final_reward，会把稀疏终局奖励变成重复奖励。

当前 trainer 还会在计算 values/advantages 前丢弃行；对 GRPO 有其既定目的，但对跨行 GAE 会破坏时间连续性。因此 PPO 数据选择、GAE 和优化 minibatch 的次序必须一起设计，不能只复制一个 GAE 函数。

### 5.2 初版建议使用 transition 模式

每次模型调用独立成行，更容易给该调用绑定 `Z_t`，并复用相同 response token IDs。trajectory 模式会把多次 observation 和 response 合并；一条合并行只有一个开头 PI，无法表示后续更新的状态。把所有未来 PI 放在初始 prefix 又会造成未来信息泄漏。

初版可按 `rollout_id + attempt_id + turn_index/request_id` 重建完整链，对有效模型生成 token 做跨 turn GAE；observation、PI、padding 不推进时间线。之后再研究 causal interleaved PI 与 trajectory packing，避免首轮复杂化。

## 6. SWE-smith 训练与 SWE-bench 评测

[SWE-smith](https://swesmith.com/)提供面向编码 Agent 的大规模训练任务。当前 repo 的 `examples/swe_smith/` 已具备 shell agent、镜像准备、任务加载、终局测试 reward、K8s launcher；这是直接可借鉴的编码环境基础。

但当前示例不等同于完整 SWE-bench adapter：

| 项目 | 当前 SWE-smith 示例 | SWE-bench 所需 |
|---|---|---|
| 数据 | 预处理 SWE-smith JSONL，字段含 image_name、F2P/P2P | issue/problem_statement、repo、base_commit 等官方实例字段 |
| 初始代码 | `git checkout instance_id` 的注入 bug 分支 | 官方实例对应的 base commit / 实例镜像 |
| 测试恢复 | 从 `HEAD~1` 恢复 SWE-smith 删除的测试 | 按官方 harness 的 test patch 和 repo-specific 测试流程 |
| 判分 | 内置 pytest parser，默认 `SMITH_F2P_ONLY=1` | 官方 harness 的 resolved/report |
| 回归测试 | 默认仅保留 F2P 文件内的 P2P | 遵守该 benchmark 的完整判分语义 |
| 训练 reward | 成功后可能叠加 turn/context 长度惩罚 | 最终报告使用官方 `% resolved`，单独记录训练 shaped reward |
| patch 输出 | `agent_output` event 的 `patch` | JSONL：instance_id、model_name_or_path、model_patch |

相关实现位置：`smith_agent.py:495` 初始化，`:586` evaluator，`:656,680` 长度惩罚，`:715` loop，`:843` main。K8s 模板还覆盖 max turns 为100、单次输出上限12288、观察字符上限6000。Python 默认值、run.sh 和 Job 环境变量并不完全一致，必须看最终 resolved config 与容器环境。

训练前还缺素材：源码归档没有 `train_dataset_mixed.jsonl` / `val_dataset_filtered.jsonl`；本地文档要求单独下载。repo 镜像也不在这份源码中。数据难度过滤基于上游 Qwen3.5-9B 探测，换成1.7B后不能假设仍有足够成功样本。

官方评测流程应为：固定实例 → Actor-only 生成补丁 → 干净环境应用补丁 → 官方 harness 判分 → 保存逐实例 report。JSONL 格式和 Docker 评测入口见 [SWE-bench Evaluation Guide](https://www.swebench.com/SWE-bench/guides/evaluation/)。patch exporter 需要覆盖新建/删除文件、拒绝空或损坏记录，并在隐藏测试恢复前捕获 Actor patch；不能仅假设 `git diff HEAD` 自动包含所有未跟踪新文件。

Lite 有300题、Verified有500题，见 [SWE-bench 官方页面](https://www.swebench.com/)。可以先选固定小子集验通基础设施，但必须明确叫子集结果。Lite 与 Verified 可能交叠，应按 instance ID 去重，避免把前者反复调参后将后者称为完全独立测试。

本研究可以继续用 Verified 与已有工作对照，但不能将其视为无污染的新任务泛化证明；已有 [公开审计](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)指出测试质量与污染问题。后续可考虑按时间划分的 SWE-rebench V2 作为补充，其 [官方运行环境](https://github.com/SWE-rebench/SWE-rebench-V2-OpenRewardEnv) 已提供容器化任务接口。本轮没有评估该补充方案的全部镜像成本，也不宣称它天然无污染。

## 7. SWE 环境中 PI 的设计

建议第一版只采集当前环境状态，保持原始研究问题：

| 状态候选 | 用法 | 必须核实 |
|---|---|---|
| 工作区文件与相对起点的净变化 | 路径、创建/删除/修改、限预算内容，覆盖未跟踪文件 | 不读取修复后的历史版本、gold patch、隐藏测试 |
| 当前运行状态 | 由任务实际产生且仍存活的进程、退出状态、任务相关临时产物 | 只访问本 rollout 所属沙箱，不能扫描共享服务器 |
| 工具输出未完整展示的实际结果 | 被截断输出的状态后果、自动生成文件等 | snapshot 在下一次模型调用前完成，不能提前运行 evaluator |
| 完整当前状态 vs Delta | 后续在相同支持集上比较 | 空 Delta 与未变化的隐藏状态不能混为等价输入 |

不把 hidden tests、F2P/P2P 答案、gold/reference patch、未来轨迹、最终 resolved 或事后 evaluator 输出作为主 PI。若未来研究 test-progress Critic，应单独命名、说明时序与 oracle 强度，它不是此次 environment-state 主方法的无声扩展。

SWE 里很多状态可通过 shell 主动查询；因此“Actor 当下没看见”和“部署时绝对不能访问”要分开。主实验可验证额外当前状态是否改善 value，但要以证据区分未观测信息与计算便利。还需承认完整 repo snapshot 昂贵；哈希只能证明文件变了，不能传达其语义。序列化要记录预算、省略字段与耗时。

### 建议的数据协议

```text
rollout_id / attempt_id / logical_call_id / turn_index
state_schema_version / snapshot_sequence / state_hash
actor_prompt_hash / actor_model_version
privileged_state_before_call
response_token_ids / rollout_log_probs / response_mask
step_reward / terminated / truncated / termination_reason
```

PI 通过独立 event/控制通道传输，不进入模型请求 messages。Gateway 当前按 prompt 去重，而相同 prompt 可能对应重试或不同真实调用，不能只按文本 hash 配对。state ID 要与有效 response 明确绑定；缺失或歧义时应拒绝样本，而不是错配后继续训练。后台进程可能随时间变化，snapshot 还要记录采集时刻/一致性边界。

## 8. 建议的实现拆分和验收

以下是待实现设计，不是现有模块清单：

| 阶段 | 主要改动 | 验收产物 |
|---|---|---|
| P0：任务与评测 | SWE 数据 adapter、隔离 runner、官方 predictions exporter | 小批固定任务可得到可复核的官方 report；环境失败与未解决分别记录 |
| P1：普通 PPO | transition 数据协议、真实 reward 时间线、跨 turn token GAE、完整轨迹 batching | 两步非零 value 递推测试；真实模型 pre-token 对齐；普通 PPO 学习信号 |
| P2：Privileged PPO | snapshot event、Critic-only token builder、两个 worker 入口切换 | 同轨迹 Actor tensors/logprobs/reward 与无 PI 构造一致；state 时间对齐 |
| P3：研究日志 | manifest、逐调用原始 token 数据、value/return/adv、逐实例评测 | 可从日志还原一条真实完整轨迹和训练语义 |
| P4：正式对照 | 同 baseline 配置，仅 PI 开关不同 | 固定任务集的学习曲线、样本/时间成本、多 seed 后续复核 |

可考虑新增 `agentlightning/privileged_state/`、`agentlightning/verl/privileged_critic.py`、`agentlightning/verl/trajectory_gae.py`，在 `examples/swe_smith/` 增加 state adapter；另设 `examples/swe_bench/` 负责官方评测。优先做通用接口和独立 adapter，避免复制整套 CAPO 或更换其历史实验。

CAPO 的 `arft/privileged_critic.py` 可作为小而明确的参考：独立左 padding、复用 Actor response IDs/mask、重算 Critic positions、构造不修改原 batch 的 view。其 token GAE 也可作为语义参考；不能一并继承 ALFWorld 特有终止副本、旧 veRL 依赖或所有训练超参。

GAE 验收应覆盖：跨 turn、mask 空洞、padding、重排、重复/缺失索引、无效动作、reward 所在最后有效 token、terminal 与 truncation。若是固定任务预算结束，可定义为终局；若是采样切段，则需要 continuation bootstrap；基础设施失败应独立记录。

SWE 长回复下，CAPO 的每 token `gamma=.99` 不应默认照搬：100 token 后权重约0.366，1000 token后约0.000043。为了研究稀疏成功率，可以考虑显式 `gamma=1`；这是新编码 baseline 的设计选择，须在普通/PI两组一致。`lambda=1`、终局bootstrap为0时，return为折扣MC，raw advantage为return−V，不能将效果解释成lambda<1的TD混合。最终 gamma/lambda 应在实现时明示。

初版建议固定一种 loss aggregation，显式记录 minibatch 单位和每批实际 optimizer updates；不要同时变更 Actor 历史、prompt、head 初始化和学习率。模型规格需要根据编码任务成功信号和显存再定，1.7B 可用于通路验证，但不能预设它适合正式 SWE 学习。

## 9. 日志：吸收参考中的最小必要部分

当前 Gateway Store 是内存数据；W&B 压缩轨迹上传也是抽样路径。两者不能代替可重算的实验日志。

从第一轮 baseline 开始统一保留：

- run：源码版本/归档标识、dirty patch、resolved config、模型/tokenizer/chat template、数据任务清单与 hash、镜像 digest、harness/veRL/vLLM版本、初始化与 seed 记录。
- 每 update：成功数、实际任务/环境步/生成 token 数、raw/normalized advantage统计、value MSE/EV/return variance、all/first/last scope、KL/clipfrac/ratio/entropy/梯度、optimizer updates和耗时。
- 抽样完整轨迹：原始 token IDs、mask、old/ref logprob、pre-action PI、V/return/advantage数组、request/attempt/turn关联。仅保存解码文本后重分词不能恢复历史训练 batch。
- 每次正式评测：全量逐实例 patch、resolved、失败原因、时间/命令预算、harness report；失败项保留在预先约定分母中，基础设施错误另列。
- 本地持久化与 checkpoint retention 分开；日志抽样采用固定hash，避免消耗训练RNG。

EV 在 return variance 为0时应标未定义或不可解释。不同 on-policy batch 的EV不能直接证明纯Critic因果优势。固定诊断集应记录采集 Actor/continuation policy 与target来源，避免把旧策略return当成新策略真实value。

## 10. A800 工程可行性

已阅读 `A800/AGENTS.md` 与 `A800/A800_PACKAGE_INSTALL_GUIDE.md`。Windows 为源码工作站，A800 为部署目标；使用 WSL rsync 预览后同步，远端模型/数据/日志/checkpoint 独立保护。

本轮没有连接 A800，所以指南中的GPU数量、镜像IP、Conda目录、磁盘余量和旧实验占用均只按历史资料理解，不能声称已实时核验。

关键依赖差异：

| 项目 | 当前源码要求 / 示例 | 实际含义 |
|---|---|---|
| Python | >=3.12 | 不能默认复用旧 Agent-R1 环境 |
| veRL | 项目标明兼容0.7.1/0.8.0；0.9存在入口变化 | 固定一套版本再实现，避免同时迁移后端 |
| vLLM | 安装脚本映射0.7.1→0.12.0，0.8.0→0.20.2 | 与torch/CUDA配套核实 |
| CUDA示例 | cu129/cu130，FlashAttention2.8.3源码构建 | A800旧指南的torch2.5.1/cu124只是旧验证，不是此项目兼容证明 |
| SWE官方示例硬件 | 4×B200、Qwen3.5-9B、长上下文 | 不能据此承诺相同配方可在4×A800运行 |
| 运行环境 | 当前SWE示例K8s；local runner启动Python子进程 | 单机Docker runner需要新增，不是改一个runner_type即可复用smith main |
| 素材 | 数据JSONL、Python wheels、镜像 | 离线服务器必须预备并核对可访问性 |

SWE 评测的CPU、内存和镜像磁盘成本可能比GPU更早成为瓶颈。[官方 harness 文档](https://www.swebench.com/SWE-bench/reference/harness/) 给出至少120GB可用空间的参考，缓存全部instance可更大；这不等于128个repo的SWE-smith训练镜像总量。需要对所选小子集独立做磁盘预算。

单机A800优先考虑 Gateway+trainer+受限并发Docker rollout，避免以复现“两台机器K8s”作为第一目标；若现成K8s和CPU节点已可用，则直接复用官方Job路径更省工作。运行器应保证每rollout的容器/进程所有权与退出清理，保留之前native资源泄漏修复的生命周期经验。

现成多个 `run_local.sh` 含全局 `pkill -f agl-server/agl-controller` 和 `ray stop --force`；在共享A800不能原样执行，应改为仅清理本run创建的进程。SWE示例的regex命令拦截与移动.git也不能独立视为不可绕过的隔离边界；评测数据、网络、只读素材和容器权限要按实验边界配置。

部署实施前仍需确定该项目的远端代码目录、专属Conda环境、实际GPU/磁盘/驱动和Docker/K8s权限。此处只是待核实信息，不要求为了完成本次源码调研立即创建环境。

## 11. 其他新 benchmark 的选择

| 候选 | 与 idea 的关系 | 当前准备程度 / 建议 |
|---|---|---|
| SWE-smith → SWE-bench | 编码任务，跨域价值高；真正PI增量需要验证 | 当前优先方向，已有训练harness雏形，需PPO和官方评测接入 |
| ScienceWorld | 当前物理状态与局部文本观测天然不同，更接近原AAC问题 | 已有local示例；[官方API](https://raw.githubusercontent.com/allenai/ScienceWorld/main/scienceworld/scienceworld.py)含getObjectTree，可作为第二新环境候选 |
| LLM-in-Sandbox / Terminal类 | 文件、后台进程、服务副作用，适合通用snapshot/delta | repo已有LLM-in-Sandbox示例；本轮未验证完整Terminal-Bench接入，工程成本较高 |
| AppWorld / WebShop | 数据库、session/backend状态更直接 | 原idea中有方案；当前Lightning归档未发现对应现成example，列为后续adapter |

ScienceWorld也不能直接把现成示例成绩叫官方test成绩：`train_sw_agent.py:55`目前按variation索引尾部切出20%作为val；官方环境提供train/dev/test variation接口。正式benchmark应固定所选split、simplification和reward口径。对象树API存在不等于所有字段都允许进入主PI，仍需审查其内容并排除gold行动序列和隐藏评估目标。

## 12. GitHub 维护与本轮状态

用户指定后续维护仓库为 [Jay-Juice/agent-lightning](https://github.com/Jay-Juice/agent-lightning)。本轮 GitHub connector 请求返回404，安装范围搜索未找到该repo；非交互 `git ls-remote` 也未取得refs。不能仅据此判断仓库不存在，私有仓库授权/连接范围也可能导致不可见。

后续以实际内层源码目录作为Git根较清晰；访问恢复后先检查远端是否已有提交，再选择保留远端历史的接入方式。建议 `origin` 指向用户仓库、`upstream` 指向Microsoft，开发用 `codex/` 分支。若远端为空，可用当前已核验归档建立明确的初始版本；不要将参考handoff、模型、数据、日志或所有未知文件一并提交。

本次实际完成：全目录/归档盘点、参考材料梳理、关键源码审查、101份Python静态语法检查、官方SWE资料核查、形成此调研报告。

本次未执行：算法实现、源码改动、Git初始化/commit/push、A800部署/环境创建、Docker镜像下载、训练或SWE-bench评分。后续实现已有具体入口和验证边界，但这些待实现模块尚未交付。
