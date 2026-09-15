# Agent Lightning：PPO baseline 就绪情况与 SWE 扩展核查

日期：2026-09-10，Asia/Shanghai。最终 CPU 诊断时间为 14:42，退出码 0。

**结论：运行环境已经具备，现有 GRPO 小规模流程可复用；当前还没有可以直接作为 privileged-critic 主对照的 SWE 多轮 PPO baseline。缺口主要在配置和训练数据语义，不是重新安装环境。**

本次进行了目录盘点、参考文档与关键源码阅读、A800 只读检查及 CPU 数据链诊断。没有启动新训练，没有修改训练算法、共享环境或历史实验，没有部署训练源码，也没有 Git 提交或推送。

## 1. 本轮阅读范围与目录职责

实际 Python 源码根是 `D:/ai project/RL学习/agent-lightning-main/agent-lightning-main`，本次盘点仍为 218 个文件；应用工作区是其上一层。两层目录均没有 `.git`。

已阅读原始 idea、09-02 项目总 handoff、09-06 `handoff.md`、09-09 CAPO P1 handoff、token GAE 说明和日志建议；结合当前目录的 09-09 调研记录、A800 指南与最新环境汇总。旧 handoff 中的运行进度和操作安排只作为历史背景，没有据此管理旧任务。

源码重点覆盖：

- `agentlightning/server/`：Gateway、事件裁剪与 prompt 去重。
- `agentlightning/controller/`：本地进程/K8s 运行方式及配置。
- `agentlightning/verl/`：入口、rollout manager、adapter、训练次序、优势广播和 loss weighting。
- `examples/swe_smith/`、相关文档与测试：数据、任务执行、评分及训练配置。
- C 盘辅助目录的 `a800-training`：实际部署的训练入口、Docker worker、trace hook、启动与验证记录。
- 同级 CAPO 的 `arft/privileged_critic.py` 和 token GAE：作为独立 Critic view 与跨行递推的实现参考。

全目录盘点不等于对每个 vendor 示例逐行审计。与当前就绪判断直接相关的五个源码/配置文件，本地和 A800 SHA-256 全部一致，哈希见本目录 `research/ppo_readiness_evidence_2026-09-10.json`。

## 2. A800 实际状态

| 项目 | 本轮核实结果 |
|---|---|
| SSH 账号 | `ubuntu` |
| GPU | 8 × A800 80 GB；检查时每卡约 14 MiB 显存，利用率 0% |
| 磁盘 | 系统盘可用约 175 GiB；D1 可用约 779 GiB，属于检查时快照 |
| 主环境 | `/media/ubuntu/D1/zsj/agent-lightning-runtime/envs/agent-lightning-d1` |
| Python | 主环境实际可执行文件，版本记录为 3.12.14 |
| 包版本 | torch 2.9.0+cu129、verl 0.7.1、vLLM 0.12.0、Ray 2.58.0、Transformers 4.57.6、FlashAttention 2.8.3 |
| 依赖检查 | 本轮 `python -m pip check`：No broken requirements found |
| CUDA 可见性 | `torch.cuda.is_available()=True`，8 卡可见；本轮未重新做 GPU 训练 |
| 源码导入 | `/media/ubuntu/D1/zsj/agent-lightning-main/agent-lightning-main/agentlightning/__init__.py` |
| Docker | `ubuntu` 可访问；数据根在 D1；检查时无运行容器 |
| 已有任务镜像 | 1 个 exceptiongroup SWE-smith 镜像，3 个 SymPy SWE-bench 实例镜像 |
| SWE-smith 元数据 | 本轮逐行计数：训练 6,343，验证 474 |
| 可离线训练子集 | exceptiongroup：32 条训练、6 条验证；全量仓库镜像未备齐 |

最新的本地 A800 环境汇总已经覆盖了 09-09 老调研中的 Docker 权限、数据缺失和单机运行器缺失问题，不应继续将这些当作当前阻塞。该服务器指南包含运维信息，不随公开仓库归档。

实际辅助脚本在服务器 `agent-lightning-runtime/tools/a800-training/` 和 `tools/swe-bench-local/`；本地原件仍在 `C:/Users/zsj/Documents/ChatGPT/agent-lightning/`。后续要纳入 D 盘源码维护，需按明确文件清单迁入，不能假定它们已随原仓库受版本管理。

## 3. 之前究竟跑通了什么

本轮读取了以下三个运行目录中的 `resolved-config.json`、退出码、验证报告和检查点目录：

| 运行 | 真实算法/critic | 已验证内容 |
|---|---|---|
| `training-smoke-20260910c` | GRPO；`need_critic=False` | 算术诊断 2 步，actor 梯度约 15.405 / 1.696；既有报告比较的 311 个参数张量发生变化 |
| `training-resume-20260910b` | GRPO；`need_critic=False` | 从第 2 步恢复到第 3 步，梯度约 1.067；既有报告再次验证参数变化 |
| `training-swe-20260910` | GRPO；`need_critic=False` | SWE-smith 单步流程，2 条训练轨迹、前后各 1 条验证轨迹；报告明确 `effective_learning=false` |

三次退出码均为 0，检查点组件均只有 `actor/`。本轮读取的是已有参数核查报告，没有重新加载大检查点逐张量比较。

三份配置都使用 `adv_estimator=grpo`、`enable_rollout_level_advantage=true`。`critic.enable` 为 null，实际 `need_critic()` 返回 false；critic 模型路径仍继承默认 `~/models/deepseek-llm-7b-chat`，没有配置成本实验的 Qwen critic。

因此，函数名 `run_ppo`、类名 `RayPPOTrainer`、配置里的 `ppo_mini_batch_size`，以及 optimizer step 增长，都不能证明执行了带 learned value critic 的 PPO。

SWE 两条训练轨迹最终奖励都是 0，GRPO 没有组内奖励差异。环境说明还记录了真实模型 SWE-bench 三题 0/3；那是小规模评测记录，不是完整 benchmark 成绩或模型提升证据。

## 4. 对 PPO 能力的准确判断

框架并非完全没有 PPO：入口已经委托 veRL 创建 Critic worker；trainer 已包含 value inference 和 critic update；服务器安装的 veRL 原生 GAE 可以跳过 observation/padding token。

本轮用非零人工 values 验证：两轮交互完整合并为一行时，原生 GAE 在 `gamma=lambda=1` 下，将终局奖励 1 正确传播给全部四个生成 token。即使 observation 位置的 values 为 8 和 9，也不影响有效 token 的 returns。

**如果限定每条轨迹始终完整合并成一行，普通 PPO 可以复用这条原生 GAE 路径；不能笼统声称它必须从零实现。** 但当前实际 SWE 数据不满足这个保证，且逐轮 PI 还需要独立的数据通路。

已确认的缺口：

| 位置 | 当前行为 | PPO 所需处理 |
|---|---|---|
| `agentlightning/verl/config.yaml:10` | 默认打开 rollout 标量优势广播 | 对 token GAE 关闭；本轮已复现非恒定 token advantage 的 RuntimeError |
| `rollout_adapter.py:502` | transition 模式给每个调用都分配 final reward | 保留时序 reward；稀疏成功奖励只归终局对应 token |
| `rollout_adapter.py:568,587` | trajectory 前缀失配后拆行，每段均分配 final reward | 不能把这些段当作独立终止轨迹计算 GAE |
| veRL 0.7.1 原生 GAE | 每个 batch row 独立递推，不读取跨行轨迹 ID | transition/分段情形需要按轨迹与顺序接续 |
| `trainer.py:497–532` | value/GAE 之前按超长标记、minibatch 和更新上限丢行 | GAE 输入必须具有完整轨迹或明确 bootstrap；不能静默删中间/末尾调用 |
| `trainer.py:575,630` | values 和 critic update 都用公共输入 batch | baseline 可复用；PI 版本两个入口都必须切到相同的 Critic 专用 view |
| 实际辅助入口 | 没有适合当前模型的 critic 参数、microbatch/offload 与恢复验证 | 显式设置并验证；使用自定义 `token_gae` 时也要显式确保 critic 被启用 |

补充：`per_rollout_mean` 是已有的轨迹权重设计，并非 token GAE 必然不兼容。选定 baseline 后应固定 loss weighting，不能以接入 PI 为由顺便改变。

### 实际 SWE 轨迹里的反例

使用已保存原始 events，经生产事件裁剪/去重和 adapter 重建，并未重新 tokenize 解码文本：

| 轨迹 ID 前缀 | 用途 | 模型调用 | 合并行数 | 前缀失配 | 超长标记 | 原始生成 token → 按超长标记保留 |
|---|---|---:|---:|---:|---:|---:|
| `669f824d` | 训练 | 5 | 1 | 0 | 0 | 609 → 609 |
| `9cadd9b2` | 训练 | 8 | 2 | 1 | 1 | 1,443 → 1,202 |
| `7f063885` | 验证轨迹重建 | 8 | 1 | 0 | 0 | 1,375 → 1,375 |
| `c8aa2d31` | 验证轨迹重建 | 8 | 1 | 0 | 0 | 874 → 874 |

“验证轨迹重建”仅用 adapter 检查数据形状，不意味着验证轨迹用于训练。表中保留量仅应用了超长标记，还不是执行整个 optimizer pipeline 后的实际训练计数。

训练轨迹 `9cadd9b2...` 有 241 个生成 token 所在的后一段会被当前超长过滤丢弃。两条训练轨迹都零奖励，使这个问题在单步 GRPO 流程验证里不易从成功率发现；它不是当前 PPO 可以安全忽略的纯假设。这里不把 1/2 的小样本比例外推为全数据集发生率。

重建还出现 mixed rollout log-probs 提示。当前 trainer 可以另算 old log-probs，所以这不是独立的必然启动阻塞；研究日志仍需要区分采样时 log-prob 与后算 old log-prob，不能宣称所有历史 token 都有完整采样概率。

## 5. Idea 在新项目中的最小落地范围

保持主问题：Actor 使用实际部署可见历史 `H_t`；普通 Critic 使用 `V(H_t)`；PI Critic 使用 `V(H_t, Z_t)`。token 版本对应 `V(H_t, Z_t, a_t,<k)`，同一次模型调用的所有 token 使用同一份 pre-action state。

推荐先建立两组共用的 **transition 数据链 + 跨轮 token GAE**，再增加 PI。这样每个调用可以独立绑定状态，不依赖 prompt 总能精确合并，也不会为了追加 PI 再更换 baseline 的轨迹表示。

最小开发顺序：

1. 把已验证的本地 Docker 辅助入口纳入实际源码仓库，保留其隔离和单次运行资源归属；固定已有环境版本。
2. 完成普通 PPO：Critic 配置、轨迹/调用标识、终局 reward、跨轮 GAE、完整轨迹选择、长度与终止规则。超长、采样截段、预算终止和基础设施失败应明确区分。
3. 验证人工多步回报、mask 与排序，再验证实际模型的 pre-token value 对齐、Actor/Critic 非零梯度、参数更新、两组件保存及恢复。CPU 代数诊断不能替代这些验收。
4. 在 SWE-smith 开发集上建立有实际学习信号的 baseline，记录任务/环境调用/token/optimizer 更新预算；32/6 子集用于工程试验，不能直接代表正式跨仓库泛化。
5. 再加入 pre-action snapshot → 独立 Critic inputs → 两个 Critic 入口。配对实验保持相同 Actor prompt、reward、PPO/GAE、模型与预算。

SWE 的 PI 可研究当前工作区文件/相对初始状态的净变化，以及工具输出未完整呈现的真实结果。不要将 gold patch、隐藏测试、评分器答案、最终 resolved 或当前动作之后才发生的状态当作主 PI。Delta 是 `D(C(S_0), C(S_t))`，不是追加式事件日志。

SWE 中不少状态可由 Actor 主动用 shell 查询：需要区分“当前未观测状态”和“对已有历史的方便摘要”，不能仅凭额外通道就宣称具有真正的信息增量。

已有 CAPO 的 token GAE 和 Critic view 可参考，但不整体复制旧环境、终止动作副本、超参或所有稳定化实验。特别是每 token `gamma=.99` 对长编码输出折扣很强，新任务需要明确决定 gamma/lambda，并保证 baseline/PI 一致；本轮没有更改这些设置。

## 6. SWE 训练与正式 benchmark 的边界

建议采用 **SWE-smith 训练/开发 → SWE-bench Actor-only 生成补丁 → 官方 harness 评分**。SWE-smith 提供训练任务；SWE-bench 的标准评测读取实例补丁并在容器内测试。参考 [SWE-smith 官方项目](https://swesmith.com/) 与 [SWE-bench 官方评测指南](https://www.swebench.com/SWE-bench/guides/evaluation/)。

当前已有单机 Docker 出口，不需要为了复刻上游两台机器/K8s 部署重新搭环境。当前 worker 采用完整 F2P+P2P 二值判分且无长度惩罚，与上游示例默认设置有区别；正式两组对照应共用选定口径。

全量 6,343 条 SWE-smith 和完整 SWE-bench 尚需更多镜像与资源预算。正式测试实例不进入 RL reward，也不用测试集反复选超参。独立官方评测报告应记录逐题 resolved、补丁、环境失败和评测预算。

日志建议只吸收当前必要部分：resolved config/源码与数据版本、逐步标量、原始 token 与 mask、抽样 value/return/raw 与 normalized advantage、逐实例验证、将来的 PI 时间对齐检查。EV 配合 return variance 解释；不能直接比较不同 on-policy batch 的 EV 来宣称因果收益。

## 7. GitHub 维护状态

用户指定后续维护目标为 [Jay-Juice/agent-lightning](https://github.com/Jay-Juice/agent-lightning)。本轮 GitHub connector 返回 404；本地 Git 的非交互 `ls-remote` 成功退出但没有返回 HEAD/main refs，符合空仓库的表现，尚未获得可核对的远端提交。

后续以 D 盘内层源码作为 Git 根接入该仓库；先保留当前源码快照、确认远端 refs，再创建开发分支。C 盘辅助脚本按明确清单纳入，模型、数据、镜像、日志和检查点不进入 Git。本轮没有初始化、提交或推送。

## 8. 可复核产物与命令

- 本地 CPU 诊断脚本 `research/ppo_readiness_probe.py`：只读调用现有生产函数，不载入模型或更新参数。
- 本地诊断结果 `research/ppo_readiness_evidence_2026-09-10.json`：人工反例、真实轨迹重建、退出码及五个源码哈希。
- 本地 A800 环境使用指南：GRPO 启动方式与运行资源路径；不随公开仓库归档。

本轮最终诊断执行命令（PowerShell，工作目录为应用项目根）：

```powershell
Get-Content -LiteralPath 'research\ppo_readiness_probe.py' -Raw |
  & 'C:\Windows\System32\OpenSSH\ssh.exe' -o BatchMode=yes -o ConnectTimeout=12 -l ubuntu A800 `
  'source /media/ubuntu/D1/zsj/agent-lightning-runtime/admin/activate-agent-lightning-d1.sh && cd /media/ubuntu/D1/zsj/agent-lightning-main/agent-lightning-main && python -'
```

环境检查通过相同 SSH 连接运行 `python -m pip check`、包导入、`nvidia-smi`、`df -h / /media/ubuntu/D1`、`docker info`、`docker ps`、`docker images`，并用 Python 只读解析上述三个运行目录。没有执行同步命令或新的训练命令。

前两次临时诊断分别遇到探针未传入 config、Windows 末尾 CRLF 的 shell 问题；最终保存脚本改为直接送入 `python -`，完整成功退出。这些是本轮诊断脚本问题，不是远端训练失败。
