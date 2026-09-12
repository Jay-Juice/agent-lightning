# SWE PPO 端到端排查：2026-09-12

本次所有模型运行限定物理 GPU **0–3**。GPU 4–7 留给同门。
模型、数据和任务镜像不变更，不把参考修复或验证集成功轨迹加入训练。

## 当前结论

不能把问题统称为“模型太小”，也没有证据表明 Lightning 整个框架坏了。
已经找到并修复具体的 agent / 配置 / PPO 适配问题。相同未训练 4B 在原工具流程下
连续三遍训练集 0/32；增加精确源码编辑和语法提交检查后，两遍为 **3/32、4/32**，
验证集均为 **1/6**。这证明原工具流程是训练缺少正奖励的重要原因。
修复后已完成真实 SWE 第一步及从其完整 checkpoint 恢复的第二步：两批训练均为
**3/32**，各次完整验证均为 **1/6**。恢复运行 exit 0，四 rank 的 actor/critic
优化器计数均从 10 增至 22，checkpoint/HF 导出完整。
这证明已恢复正奖励、实际更新与恢复链路；尚未证明 PPO 提升成功率或收敛。
少量采样/重算 logprob 异常点仍有未归因的数值差异，详见文末，不宣称全系统严格等价。

1. **4B 的训练前验证能力可以复现**：独立 vLLM 与 Lightning 初始评测均为 1/6。
   成功集中在同一个构造器参数顺序任务，不意味着对 32 个训练任务也应有相同成功率。
2. **发现真实的奖励误拒绝**：新建 `test_repro.py` 导致整个源码补丁被拒绝。
   用一份模型原始补丁回放，仅排除新建复现文件后，60/60 修复测试和 29/29 回归测试通过。
3. **训练缺少正奖励，不是优化器没执行**：历史保存的 SWE 批次全部 reward=0；
   随机 critic 的误差经优势标准化后仍驱动 actor，不能解释为学到了修复。
   历史 32 步 baseline、8 步 no-warmup 和 CAPO SWE 更新均为 **1.7B**；
   先前 4B 的成功来自初始评测，不能拿不同模型的两项结果直接判断训练是否退化。
4. **CAPO 的长序列折扣和 padding 有适配问题**：默认恢复 gamma=1；
   新兼容层排除虚拟行损失并修正短尾批缩放，复制的 GAE/loss/update 源码保持原哈希。
5. **工具使用失败有逐条证据**：无效 sed 仍返回 0、空补丁提交、字面 `\n`
   写入源码造成 SyntaxError、修错函数、破坏导入，以及只修复组合任务中的部分缺陷。
6. **精确编辑和语法检查使完整训练集获得正奖励**：两次重复分别 3/32、4/32，
   四个不同训练任务成功；验证集两次均 1/6。定向先导中，4B 自行在 6 轮内将旧式
   `format_exception_only(__exc)` 调用修正为 `format_exception_only(value)`，
   2/2 F2P、87/87 P2P 全通过。没有提供任务答案或注入人工补丁。
   完整评测 76 个 episode 全部正常退出。编辑器和语法检查同时开启，不能把收益
   全部单独归因于某一个开关，也不能将此训练前结果写成 PPO 学习提升。

## 运行和证据位置

服务器代码：`/media/ubuntu/D1/zsj/agent-lightning-main/agent-lightning-main`。
下表路径均相对 `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/`。

| 检查 | 结果 | 证据目录/文件 |
| --- | --- | --- |
| 独立 vLLM，原始 agent，4B，24 轮/4096 输出/32K 上下文/6000 字符观察 | val 1/6，train 0/32 | `swe-standalone-ab-20260912-01`，original profile |
| 同设置，提示编辑验证并拒绝空补丁提交 | val 1/6，train 0/32；部分修复增多但没有完整成功 | 同目录，verified profile |
| Lightning + CAPO 入口，只评测初始权重 | val 1/6，35 次调用，3 份非空补丁，0 PPO update | `training-ppo-parity-4b-val6-20260912-02` |
| 独立后端与 Lightning 四个推理副本的固定 token 概率 | 六段固定输入，共 505 个响应 token；四副本最大/平均误差均 0 | 上述目录 `backend-parity-0.json` 至 `-3.json`；独立目录 `backend-direct.json` |
| 6000 vs 32000 字符观察，65K 上下文，排除新复现脚本 | 两组均 val 1/6、train 0/32，均无补丁拒绝 | `swe-4b-observation-ab-20260912-01` |
| 30B-A3B 原始 agent，TP4 | 严格原规则 val 0/6、train 0/32；分别 5、19 次补丁拒绝 | `swe-standalone-30b-20260912-01` |
| 30B 的全部 24 份被拒绝补丁，仅回放源码部分 | 只有一个验证任务成功，训练任务仍无成功；0 回放错误 | `swe-standalone-30b-rejected-20260912-01` 和 `-02` |
| 补丁导出规则实际容器回归 | 旧规则拒绝；新规则同源码修复 60/60 + 29/29；配置篡改仍拒绝 | `swe-export-regression-20260912-01` |
| 验证集评分正/负对照 | 未修复 0/6，恢复 pre-bug 源码 6/6，0 环境错误 | `swe-val-controls-20260912-01` |
| 真实语法错误补丁回归 | 原始源码可解析；模型候选补丁定位到 IndentationError | `swe-syntax-regression-20260912-01` |
| 精确替换编辑器实际容器回归 | 一次替换成功且读回一致；再次同请求因 0 匹配拒绝 | `swe-editor-regression-20260912-01` |
| 精确编辑器定向先导：2 个训练任务 + 1 个验证任务，12 轮/2048 输出 | train 1/2，val 0/1；训练成功为 `func_basic__l78q257t`，另一题部分修复 5/15 | `swe-4b-editor-pilot-20260912-01` |
| 完整窗口/提交检查/模型采样参数，三次重复 | 每次 val 1/6，train 0/32；全部 exit 0 | `swe-4b-verified-full-20260912-01` |
| 相同模型/数据/预算，增加精确编辑器和语法检查，两次重复 | train 3/32、4/32；val 均 1/6；76 个 episode 全部 exit 0 | `swe-4b-editor-20260912-01` |
| Transformers 独立概率及失败历史核对 | 8 段、1017 token；61K 重复上下文 MAE 0.0000064，HF 也重复同样的 sed 文本 | `swe-hf-parity-4b-20260912-02.json` |
| 历史十份 PPO 张量审计 | 包括更新后和恢复批次；详见下文 | `ppo-tensor-audit-20260912-01.json` |
| 模型完整性 | 4B 全部 3 个权重分片及配置/分词文件匹配下载凭据；30B 有凭据的 9/16 分片匹配 | `model-integrity-20260912-01.json` |
| 实际 SWE 第一步 | train 3/32，初始/步后 val 1/6；step 1 checkpoint 完整；原运行第二步 OOM、exit 1 | `training-capo-swe-editor-4b-20260912-01` |
| 63,900 token 真实调用，四卡 actor 压力更新 | 原路径 OOM；原生分块输出层路径 exit 0，四 rank 均完成更新，峰值张量显存各 36.88 GiB | `swe-capo-long-fused-20260912-03.json` |
| 从完整第一步恢复并完成第二步 | train 3/32，初始/最终 val 1/6；exit 0；四 rank actor/critic 优化器 10→22 | `training-capo-swe-editor-resume-4b-20260912-01` |

固定 token 概率相同，不能称为完整随机采样轨迹相同：配对六个任务的初始 prompt token
全部一致，但仅两个首个响应完全一致，完整轨迹并不相同。服务器批处理和执行顺序会影响采样。
后端对照只证明已检查输入上的初始概率路径正确，不是整个训练系统的形式化等价证明。

### 历史同题：为什么 8 轮配置成功，100 轮配置反而失败

逐字匹配初始 messages，将以前唯一成功的验证任务对应到后一次失败轨迹：

- 成功：`training-ppo-eval-4b-8turn-val6-20260910-01/agent/c406c81facd84106b116c5be66c8e51c`。
  实际修正 `super().__new__(cls, __exceptions, __message)` 的参数顺序，第 7 轮提交，reward=1。
- 失败：`training-ppo-eval-4b-100turn-12k-val6-20260911-01/agent/cfca642beb5542338ba50fc1b0b220bb`。
  第 4 轮因两个 bash 块格式错误，第 5 轮 sed 搜索不存在的单行 `def __new__(self, ...)`，
  退出码 0 却没有改动；第 6 轮直接提交。`changed_paths=[]`、补丁为空，0/60 F2P、29/29 P2P。

两者均是未训练的同一 4B。这次具体失败没有用完 100 轮预算，也没有训练更新，
不能归因于训练把模型能力破坏或预算不足。随机轨迹不同，以及原工具流程未检查编辑是否生效，
共同导致了观测差异。此处不能推断所有历史失败都由同一个原因造成。

Transformers + FlashAttention 2 的 8 段核对中，前 7 段 logprob MAE 约 .0094–.0208，
61,304 token 上下文段的 MAE 为 0.00000642、最大差 .000702。
Transformers greedy 在这段历史上也生成同样的重复 `sed`，支持该重复不是 vLLM 特有故障。
早期失败历史上的 greedy 输出是另一条仍不可靠的编辑命令；这不是同采样器完整 episode 对照。
此前 SDPA 核对完成前 7 段，在 61K 段触及诊断进程 20 GiB 显存限额，保留 `.log/.exit`；
后续仅使用已有 FlashAttention 2 重试，未更换模型或安装环境。

30B 补丁回放是反事实检查，不能把回放成功直接当作新模型评测分数，更不能当作训练成功。

## 数据、容器和评分

- 使用 SWE-smith 发布的 exceptiongroup 子集：32 train / 6 val，不是完整 SWE-bench。
- 逐项对照发布的 `train_dataset_mixed.jsonl` / `val_dataset_filtered.jsonl`：
  全部 38 条 instance ID、problem statement、FAIL_TO_PASS、PASS_TO_PASS 一致。
- 数据压缩包 SHA256：`848f0b91d7b4322f8f399ffa81d0f6ab9ea6eb4144bed35000bf9417a1f9ca1c`。
- agent 使用删除隐藏 F2P 测试后的 Bug Patch 分支；grader 在另一个新容器应用候选源码补丁，
  恢复评分测试后执行 pytest。模型不能读取 Git 历史，容器无网络和宿主挂载。
- 38 条源码分支审计见 `swe-source-audit-20260912-02/audit.json`。
  15 个训练任务、3 个验证任务的 Bug Patch 修改多个源码文件；不能据此把所有多文件任务都判为描述错误。
- 已确认 `combine_module__3hhws1dx` 的描述聚焦 exceptions 顺序，但 Bug Patch
  还改坏了旧式格式化参数处理。只修顺序的候选会留下格式化失败；二值奖励仍为 0。
- 评分正对照仅证明测试环境可通过，**不会**把正确补丁提供给 agent。

## PPO 和权重路径

当前 CAPO 移植的是 `token_gae + vanilla PPO` baseline，
不是 CAPO 论文中动作级概率比的算法。来源提交仍为
`e8407baea32fb36191f029f0a4666bf681bdf1ca`，vendor manifest 校验通过。

历史张量包括原 baseline 的首步和第 32 步、no-warmup 首步和第 8 步、fast 首步和第 4 步、
CAPO SWE 首步、合成 smoke 两步及恢复后的第 3 步。
SWE 全零奖励时，gamma=lambda=1 的 return 为 0；CAPO .99 下只有微小数值残差。
CAPO SWE 初始 value 标准差约 2.2288，raw advantage ≈ -V，标准化后 advantage 标准差约 1。
因此非零梯度并不证明修复学习发生。

已保存更新前后及恢复批次中的 actor/rollout logprob 差异约为 MAE .00594–.0150，
退化的 baseline 第 32 步更小，约 .00022。没有发现“actor 更新了但推理端始终还是原权重”
的明显迹象；该统计不能排除所有未检查的权重同步故障。

gamma=.99 在此实现中按每个生成 token 折扣，而不是每个工具轮次。
4096 token 终局奖励序列首 token 的折扣约为 `1.337e-18`；gamma=1 时 return 可传播至整个轨迹。
这会影响有奖励后的学习，但不能解释训练之前就失败。

CAPO 原 padding 只把虚拟行的 advantage/return 清零，response mask 仍为 1：
虚拟行会进入 critic loss 和 reference KL；固定的累积除数还会压低短尾 minibatch 的权重。
新 `capo_padding.py` 在固定 microbatch=1、FSDP、无序列并行配置下，在整个 optimizer minibatch
内归一化真实调用权重。原 actor/critic 更新循环不改，虚拟行保留 forward 以对齐集合通信，损失权重为零。
真实复制循环的标量模型梯度与未补齐解析基准一致，包含极大虚拟目标、actor KL 和不足一批的尾部。

## 本次源码改动

- 网关采样：从最终解析配置生成启动参数，包含训练/验证 temperature、top_p、top_k；
  验证 greedy 显式转为 temperature=0，拒绝 greedy PPO 训练。历史已检查训练的 temperature=1
  恰好等于旧默认，因此这不是已证实的历史零成功主因。
- 导出：默认 SWE runner 排除新建复现脚本，既有测试、配置、symlink 修改仍拒绝；
  `sandbox.json` 记录被排除路径。只把源码候选送入新 grader。
- 可选 `SMITH_VERIFY_SUBMISSION=1`：通用编辑提示，并将空补丁/禁止修改的提交退回模型。
- 可选 `SMITH_CHECK_SYNTAX=1`：在上述验证提交路径中解析修改过的 Python 源码，
  错误反馈给模型继续修复；不执行源码、不读取评分测试。
- 可选 `SMITH_CHECKED_EDITOR=1`：在临时 agent 容器安装 `agl-edit` 精确替换工具，
  无正则、必须唯一匹配、返回真实 diff。它只编辑模型指定的当前源码，不包含任务答案。
- CAPO 默认 `gamma=1`、严格 padding 权重；可通过明确配置重现旧行为。
- 训练 preflight 失败现在也会写 `run.exit`，避免误认为任务仍在运行。
- 所有新开关进入 provenance；独立后端对照保存 seed、实际请求、token IDs、源码 hash。
- 另发现网关注册 Model 时未传 version，导致每次重新注册都记录为 0。
  本地修复改为记录最近一次成功权重同步的 step，而不是即将训练的 step。
  第一轮 SWE 对照使用旧日志标记；该运行退出后，版本修复已部署主副本。
  本次同步训练的 `max_ppo_update_times=None`，该标记没有用于丢弃样本。
  不能据旧标记推断权重没更新；实际固定 token 对照见下文。

## 空间与部署

按用户明确授权，仅删除 6 个已确认完成且 SWE reward 全零的 checkpoint 目录，
未删除日志、轨迹、PPO 审计或有效的合成 smoke/resume 检查点。
释放约 424.96 GiB 表观空间，D1 可用由约 149 GiB 增至约 570 GiB。
清单：`checkpoint-cleanup-20260912-01.json`。

源码在 Windows 主副本编辑。使用 WSL rsync，逐次 dry run 审阅后部署，未使用 `--delete`。
进行中的独立对照使用 `.audit-20260912` 隔离代码，不在运行中替换其 agent 实现。
未提交或推送 Git，未安装新训练环境，未改写基座模型文件；PPO 更新保存在独立运行的 checkpoint 中。

## 验收状态

- 真实 SWE 第一步完成；第二步第一次遇到 actor OOM，正在从第一步 checkpoint 恢复重试。
- 恢复运行同时验收版本标记（初始/训练为 1，更新后验证为 2）。

最终组合 76 项单元/集成回归通过，含复制 actor/critic 更新循环、分块输出层的梯度对照；Ruff、shell 语法和
`git diff --check` 通过；padding/worker 相关 Pyright 检查 0 错误。
首次四卡 smoke 的额外 mini-batch=8 覆盖超过合成 train batch=4，被 veRL 配置校验拒绝，
没有执行更新。使用原 smoke batch=4/mini-batch=4 重试：
`training-capo-padding-smoke-20260912-02`。短尾缩放另有复制更新循环的解析梯度回归。
原排队序列因该配置失败退出，不能将它们的旧状态解释为后续 SWE 训练仍在进行。

`training-capo-padding-smoke-20260912-02` 已 exit 0，`verify_run.py` 验收通过：
两步均 4 条 episode、9 次真实调用、3 个 padding 行，没有丢调用；四个 rank 的 actor/critic
优化器计数均为 6，张量重算 GAE、终局奖励传播和 checkpoint 完整性通过。
这是合成任务上的工程验收，不能当作 SWE 学习效果。恢复运行
`training-capo-padding-resume-20260912-02` 也已 exit 0、验证通过：仅继续执行第 3 步，
四 rank 的 actor/critic 优化器计数从 6 增至 9，均有有限非零梯度。

复现修复后 SWE 两步检查的入口是 `run_checked_swe_ppo.sh`：固定物理 0–3 卡，
默认 4B、32 轮、4096 响应、65536 上下文、32000 字符观察，开启上述编辑和提交检查。
训练 T=1/top-p=1/top-k=-1，验证 T=.7/top-p=.8/top-k=20，仍是 32 train / 6 val。
第一轮真实 SWE 两步验收目录：`training-capo-swe-editor-4b-20260912-01`。
此轮最终 exit 1：第一步完整成功，第二步 actor 在长轨迹上 OOM，不能记作两步训练完成。

### 真实 SWE 第一步结果

- 32 train 中成功 3 个（9.375%）；初始及第一步后的 val 均为 1/6。
- 312 次训练调用、约 1,558,417 输入输出 token，最长 16,484 token，没有补齐行。
  32 调用 mini-batch 的尾批为 24 行，实际覆盖了不足一批的权重归一化。
- 359 次初始 train+val 调用的历史、prompt token、响应逐项一致；无观察截断。
  每次独立评分与发给网关的奖励一致；成功奖励正确传播到全部动作的 return。
  证据：`swe-live-integrity-initial-20260912-01.json`、运行内 `initial-tensor-check.json`。
- actor grad norm=7.51275、critic grad norm=179.39468，均有限；actor PPO KL=.00140693，
  reference KL=.02801046。初始 value std=3.734375，critic loss=58.8832，
  仍需观察 value 学习的稳定性，不能把正奖励解释为已收敛。
- step 计时约 1862 秒：采样 451 秒，旧策略/参考概率/value 重算分别约 128/133/113 秒，
  critic/actor 更新约 514/504 秒，权重同步约 9 秒；验证与保存另计。
  Ray 任务状态确认阶段完成，没有仅用 GPU 利用率推断进度。
- 四个推理副本的固定 505 token 概率完全一致，相对初始权重 MAE=.0216114。
  独立 Transformers 重载已保存的 step 1 HF 权重，与推理端 MAE=.0126038；
  Transformers 相对初始概率也发生变化，变化量与 vLLM 的相关系数约 .593。
  数值对照支持实际权重已同步，不是只有日志版本变化；不作任意精度阈值的全系统保证。
  证据：运行内 `post-update-1-probe-{0,1,2,3}.json`、`post-update-1-hf.json`、
  `weight-sync-step1-check.json`。
- 保存的 `tokenizer.json` 与基座解析后完全相同；chat template 从配置迁移至 jinja 文件，
  加载后的模板及实际初始 prompt token 均一致。加载保存目录时出现 Mistral regex 提示，
  没有据此给 Qwen tokenizer 套用 Mistral 的修正。

版本标记的 16 项 rollout-manager 回归已通过（含初始、更新、恢复版本注册），
Ruff/格式检查通过。隔离测试显式加载隔离模块，避免 editable 安装优先导入正在运行的旧主副本。
部署到完整主副本后，16 项回归再次通过，三个修改文件的 Pyright 为 0 错误、0 警告。

### 真实 SWE 第二批采样与张量

- 32 个训练任务全部完成，仍成功 3 个；373 次真实调用，3 行 padding，没有丢弃失败轨迹。
- 其中一条失败轨迹连续生成重复且格式错误的 bash 块，14 次输出达到 4096 token 上限，
  最后在 prompt=63,962 token 时由上下文预算检查结束。它得到 0 奖励并保留在训练中，
  不是容器异常或人为过滤后的结果。
- 实际评分与终局 token reward、每轮 return 一致；补齐行 advantage/return 全零。
  采样侧与训练侧 logprob MAE=.00663986。
- critic value std 从首批 3.7344 降至 1.7732；本批成功/失败动作的平均标准化优势分别
  为 .85188 / -.02095。随机 value head 的误差仍存在，但没有发现成功奖励方向被算反。
- 截至第二批结束，774 次调用的完整历史、prompt token 和响应逐项一致；0 观察截断。
  证据：运行内 `second-tensor-check.json`、`swe-live-integrity-two-batches-20260912-01.json`。
- 第二批输入输出 token 总数 2,797,762（含补齐），最长实际调用 63,900 token；
  第一批为 1,558,417、最长 16,484。第二步 critic 实际更新耗时约 725 秒。
- 对目前 16 条没有测试状态明细的 episode 逐项核查：2 条补丁在 Python 3.10 下
  无条件 `from typing import Self`，导入即失败；4 条只补上 `output=[]`，却未恢复
  `while exc` 循环中推进 `exc` 的赋值，最终超时；10 条在异常处理路径触发
  TypeError / AttributeError / UnboundLocalError，导致 pytest 本身不能正常报告结果。
  exceptiongroup 是 pytest 自身依赖，因此源码错误可能先破坏测试框架的异常处理。
  这些日志不能统称为依赖环境不匹配或评分漏报；语法可解析也不代表运行时正确。
  证据：运行内 `grading-failure-check.json`，只检查原日志和补丁，没有改写奖励。

### 第二步暴露的长上下文显存问题

第二步 critic 完成，但 actor 在 `loss.backward()` 申请 11.03 GiB 时失败：
PyTorch 已分配约 67.28 GiB，GPU 剩余约 10.33 GiB。进程自行退出并释放 0–3 卡。
这是实际长上下文反向传播超出显存，不能记成第二步训练成功，也不能只归因于模型答错。
第一步完整 checkpoint 与失败批次的原始轨迹、张量均保留。

现配置 `entropy_coeff=0`，但 `calculate_entropy=true`；复制的 CAPO forward 因此
仍计算全序列熵，并把 logprob 交叉熵的 `inplace_backward` 置为 false。
先验证仅关闭熵统计，保持 PPO 目标、预算、数据和学习率不变。
`audit_capo_long_context.py` 用安装的完整 FSDP actor worker，重载第一步模型和优化器，
每个物理 0–3 rank 都重放已保存的最长 63,900 token 调用，并设置 90% 显存上限。
只执行一次临时压力更新，不保存诊断权重、不改写来源 checkpoint 或奖励。

仅关闭熵统计的压力测试仍 OOM：`swe-capo-long-noentropy-20260912-01.log/.exit`。
因此 SWE 入口进一步启用已安装 veRL 的 `model.use_fused_kernels=true`、
`fused_kernel_options.impl_backend=torch`，使用其 `FusedLinearForPPO` 按 512 token 分块
计算输出层及反向传播。CAPO 的原更新循环与损失源码仍未修改。
8 项直接概率/梯度对照通过，覆盖 FP32/BF16、T=.7/1、熵系数 0/.01 和非整分块尾部。

`swe-capo-long-fused-20260912-03` **exit 0**：四个 rank 都完成最长真实调用的反向和更新，
张量显存峰值各 36.88275 GiB，梯度范数约 .00332875，全部优化器计数 10→11。
实际 4096 个响应 token 与原保存 logprob 的 MAE=`1.3509998e-7`，max=.00011826。
同系列 -01/-02 的反向已完成，但诊断脚本立即读取异步拷回 CPU 的 optimizer step，
读到未完成的数据，断言失败。-03 在读取前 `torch.cuda.synchronize()`，保留原断言后通过。
此修正位于诊断脚本，不能把前两次误读写成真实训练优化器被清零。

恢复的真实 SWE 运行：`training-capo-swe-editor-resume-4b-20260912-01`。
从第一轮 `checkpoints/global_step_1` 继续到 step 2，初始与最终均评测完整 6 val，
完整 32 train。已确认解析后的 model/actor fused 开关为 true、actor 熵统计 false、
熵目标系数仍 0；GPU 仍为物理 0–3。未更改奖励、任务预算、学习率或数据集。

恢复运行的初始 val 为 1/6，第二步完整采样为 3/32。372 次真实调用、0 padding，
2,245,346 个输入输出 token，最长调用 19,925 token；GAE 独立重算一致，全部实际奖励与
每轮 return 一致，成功/失败动作平均标准化优势为 .45840 / -.02151。
证据：恢复运行内 `tensor-check.json`。

恢复运行已经 **exit 0**，`verify_run.py` 返回 `status=verified`：

- 第 2 步 actor grad norm=3.96882、critic grad norm=46.26124，全部有限。
  critic loss=8.25030，reference KL=.14161547、PPO KL=.00187655。
- 四 rank 的 actor 和 critic 优化器计数全部 **10→22**，新增 12 次 minibatch 更新，
  与 `ceil(372/32)` 一致；保存的 step 2 checkpoint 完整。HF 导出 4 分片、399 个张量
  的索引和文件头逐项匹配。证据：`verification.json`、`checkpoint-chain-check.json`。
- 初始 val 使用版本 1，训练使用版本 1，更新后 val 使用版本 2。32 train、两组各 6 val
  的实例集合分别与预期一致，没有混用 split。最终 val 仍为 **1/6**。
  证据：`version-and-split-check.json`。
- 恢复运行全部 44 条 episode、455 次调用的历史、prompt IDs、response IDs 一致，
  0 观察截断，所有调用 finish reason 为 stop；仍有 22 次模型输出解析失败，属于模型
  工具格式问题，未过滤其轨迹或修改失败奖励。
  证据：`swe-live-integrity-resume-final-20260912-01.json`。
- 第 2 步约 2186 秒：采样 406 秒，旧策略/参考/value 重算约 153/168/147 秒，
  critic/actor 更新约 645/644 秒，权重同步约 9 秒；验证和保存另计。
  与不同长度的第一批不能直接比较吞吐，更不能宣称所有长任务都已达到最佳速度。

当前数据只支持“工程链路已能获取正奖励并完成 PPO 更新”，不支持“PPO 已提高分数”。
32/6 的 SWE-smith 单仓库子集不是全量 SWE-bench；不能把 1/6 验证分数外推到整个基准。

最终相关回归合并运行 **76 passed**，包含复制的 PPO 更新循环解析梯度、padding、
多轮时间结构、分布式缩放、采样配置、提交/编辑/导出、版本注册与原生 fused 概率/梯度。
Ruff、修改文件格式检查、Bash 语法和 `git diff --check` 通过；版本传递相关三个文件
Pyright 为 0 错误、0 警告。所有源码修改先在 Windows 完成，再逐文件 rsync dry run、
检查清单后部署；未同步或改写模型、数据集和已有实验输出。

### 少量 logprob 异常点的追加核对

不能只看平均误差：恢复批次 old-vs-rollout logprob MAE=.0117667，但 94,811 个动作 token
中有 5 个绝对误差超过 1，max=2.75955。未启用分块的旧批次也存在：首批 3/67,859 个，
max=1.42304；失败的第二批 4/144,193 个，max=9.17458。
选取三批各自最大误差点，逐项确认训练 input IDs、response IDs 与原模型调用完全相同，
原始 logprob 也与保存的 rollout tensor 一致。
独立 Transformers BF16 + FA2 重载对应原模型/step 1 checkpoint，前两个点与训练端
相差约 1e-6，第三点相差约 .01386；与原采样端仍有上述大差异。
这些点没有发现 token 错位；独立 HF 也支持训练侧分数，不能归因于 CAPO 特有输出层错误。
进一步用全新 vLLM 进程、关闭 prefix cache、TP=1、BF16、eager，分别重载基座和
step 1 HF checkpoint，比较固定前缀的 prefill 与 next-token 概率：二者相同，
但仍不能重现原在线采样的大误差点。

| 异常点 | 原采样 logprob | 训练 logprob | 对应权重新 vLLM prefill/next-token |
| --- | --- | --- | --- |
| 首批 row 149 / token 111，基座 | -3.77325 | -2.35021 | -1.70141 |
| 原第二批 row 342 / token 353，step 1 | -.70125 | -9.87583 | -11.12551 |
| 恢复第二批 row 257 / token 0，step 1 | -1.61545 | -4.37500 | -4.26637 |

后两个前缀若改用基座，分别为 -10.87547、-2.59359，仍不匹配原采样，
没有支持“简单地回退到初始权重”的解释。实际请求 T=1/top-p=1/top-k=-1；
基座配置没有 repetition penalty，不能拿模型自带 T=.7/top-p=.8 的默认值解释这些点。

最后对三个原始 prompt 在独立 vLLM 上重新生成，然后把新生成的整段 token
作为 prompt 重算其概率；不运行 Lightning/CAPO、agent 工具、权重同步或 prefix cache。
三段共 352 token，各段平均绝对误差 .00952/.00791/.01316，最大 .19958/.13123/.48825，
没有超过 1 的误差。这证明单独 vLLM 的生成/重算路径也并非逐位相等，
**但没有复现旧在线轨迹最大 9.17 的差异**；其精确原因仍未闭合，不能把全部异常
武断归类为正常 BF16 误差、缓存错误或框架故障。

证据均在 `swe-logprob-outliers-20260912-01/`：`{base,step1}.json`、`*-hf.json`、
`{base,step1}-vllm.json`、`base-generation-vllm.json` 及各自 exit 0 的日志。
诊断入口 `audit_swe_vllm_outliers.py --generation-check`，最后一项只使用物理 GPU 0。

已定位并修复导致无正奖励和更新失败的具体问题，真实训练/恢复验收通过。
仍需将低成功率、尚未证明的学习提升，以及未完全归因的少量 logprob 异常保留为后续实验限制。
不能将这次两步检查写成全量训练已经完成或所有潜在问题均已排除。
