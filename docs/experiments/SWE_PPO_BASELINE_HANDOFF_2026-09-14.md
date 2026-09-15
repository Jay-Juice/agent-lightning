# SWE 多轮 PPO baseline 工作总结与 Handoff

更新时间：2026-09-14 13:13，Asia/Shanghai。以下运行状态为该时点的只读检查结果，不代表此后状态。

## 1. 当前结论与接手重点

后四卡的 Qwen3-4B-Instruct-2507 全量 baseline 已成功启动，并完成 24 个 PPO step，正在第 25 步采样（检查时 30/32）。训练前及第 20 步的完整验证均完成。没有在训练主日志中发现 Traceback、OOM、RuntimeError 或 AssertionError，已记录的数值指标无 NaN/Inf。

训练前验证为 **70/470（14.89%）**，第 20 步为 **73/470（15.53%）**，增加 3 条、约 0.64 个百分点。这个幅度不足以证明稳定的能力提升；验证采用随机采样，需要继续观察。前 24 步累计训练成功 188/768（24.48%），这是不同训练任务上的在线成功率，不能直接与验证分数比较。

接手后优先事项：

1. 查看后四卡当前 step、运行是否退出及最新完整验证结果，不要重复启动已有作业。
2. 在第 40 步检查全量验证和首次 checkpoint：actor/critic 权重、optimizer/RNG 是否完整，保留策略是否符合要求。
3. 持续比较第 40、60……步验证与初始 70/470，区分模型修复失败、测试执行问题和 rollout 执行失败。
4. 保留当前 microbatch=2。不要根据某一时刻显存较空就直接改成 4。
5. 后续加速实验必须固定任务/轨迹进行对照，不能仅比较两次在线训练的 ETA。

**后四卡尚未到第 40 步，因此本次运行的 checkpoint 保存、恢复和最终模型导出尚未验收。** 当前无需重启训练。

## 2. 两个运行务必区分

| 项目 | 后四卡：本次新 baseline | 前四卡：保留的旧运行 |
|---|---|---|
| 物理 GPU | 4,5,6,7 | 0,1,2,3 |
| tag | `capo-swe-pythonfull-v7-4b-gpu47-20260914-01` | `capo-swe-pythonfull-v6-4b-20260913-03` |
| screen | `agl-full-v7-4b-gpu47-20260914-01` | `agl-full-v6-4b-20260913-03` |
| 数据 | v7：6248 train / 470 val | v6：6251 train / 470 val |
| agents | 32 | 16 |
| vLLM max_num_seqs / max_num_batched_tokens | 8 / 16384 | 4 / 8192 |
| 验证 | 每 20 步全量 | 每 196 步全量 |
| 保存 / 最多保留 | 每 40 步 / 3 组 | 每 32 步 / 1 组 |
| Lightning 端口 | 18501 | 18401 |
| 13:13 核对状态 | 完成 24 步，第 25 步采样 | 完成 32 步，继续运行 |
| checkpoint 目录 | 尚无 | 已出现 `global_step_32`，本次仅核对目录存在 |

此次操作没有停止、重启或修改前四卡原训练。旧运行仍使用其原始评分器和数据，不能把两组指标当成严格同条件实验。不要对共享服务器执行全局 `ray stop`、按名称批量 kill 或 Docker 全局清理。

## 3. 本地、GitHub 与远端位置

本地项目主文件夹，即本报告所在位置：

`D:\ai project\RL学习\agent-lightning-main`

本次源代码工作树：

`D:\ai project\RL学习\agent-lightning-main\agent-lightning-speedtrial-20260913`

原代码工作树：

`D:\ai project\RL学习\agent-lightning-main\agent-lightning-main`

GitHub：[Jay-Juice/agent-lightning](https://github.com/Jay-Juice/agent-lightning)，本次分支为 `experiment/swe-speed-20260913`。截至本次核对，该工作树干净；此前已确认推送到远端的最新提交为：

`ed12734c1afcd89f273e004e2b0f47a3aadf0fa9` — Record verified GPU 4-7 full baseline startup。

重要提交：

- `5e0b695`：记录 actor4/critic2 容量试验，拆分试验 microbatch 控制。
- `9179047`：配置后四卡加速全量入口、三条数据剔除脚本及启动报告。
- `ed12734`：补充实际启动验收记录。

本报告是放在外层主文件夹的新交接文件；该外层不属于上述 Git 工作树，本次没有把本报告另行提交或部署。仓库内原有启动记录仍是启动时快照，以本报告的带时间戳状态为更新信息。

SSH 主机别名：`A800`，用户：`ubuntu`。远端路径：

| 用途 | 绝对路径 |
|---|---|
| 新代码部署目录 | `/media/ubuntu/D1/zsj/agent-lightning-speedtrial-20260913` |
| 原代码部署目录 | `/media/ubuntu/D1/zsj/agent-lightning-main/agent-lightning-main` |
| runtime 根目录 | `/media/ubuntu/D1/zsj/agent-lightning-runtime` |
| 环境激活脚本 | `/media/ubuntu/D1/zsj/agent-lightning-runtime/admin/activate-agent-lightning-d1.sh` |
| 模型目录 | `/media/ubuntu/D1/zsj/GOPD/G-OPD-main/models/Qwen3-4B-Instruct-2507` |
| 新 Ray 临时目录 | `/media/ubuntu/D1/zsj/ray-speed-0913` |

开发约束见本地主文件夹下 `A800/AGENTS.md`：Windows 是源代码权威副本；本地编辑后使用 WSL rsync，先 dry-run 再同步；不使用 `--delete`。远端数据、权重、日志是运行状态，不要用代码同步覆盖。此次没有升级共享训练环境。

## 4. 新 baseline 的最终配置

以本次日志目录中的 `resolved-config.json` 为准，避免依据某层脚本的默认值误判最终覆盖结果。该文件含服务凭据，检查时只选择必要字段，不要整份公开、粘贴或提交。

| 项目 | 已生效的值 |
|---|---|
| 模型 | Qwen3-4B-Instruct-2507，从原始权重开始，`resume_mode=disable` |
| GPU | 4 张 A800 80GB，物理编号 4–7 |
| PPO 后端 | `capo`，使用 vendored CAPO actor/critic 更新循环与适配层 |
| 数据量 | 6248 Python train，470 Python val |
| 训练长度 | 4 epochs，保留尾批，196 steps/epoch，总计 784 PPO steps |
| task batch / PPO minibatch | 32 / 32 |
| actor / critic PPO epochs | 1 / 1 |
| actor / critic 更新 microbatch | 2 / 2 |
| old/ref log-prob、critic forward microbatch | 全部 2 |
| actor / critic 学习率 | 1e-6 / 1e-5 |
| weight decay | 0.01 |
| warmup | 无学习率 warmup；ratio=0，steps=-1 为按比例计算；critic_warmup=0 |
| gamma / lambda | 1 / 1 |
| PPO clip / KL 系数 | 0.2 / 0.001 |
| 优势处理 | token GAE、优势 whitening、padding loss 适配 |
| actor FSDP | FSDP1，`reshard_after_forward=false` |
| critic FSDP | FSDP1，保留现有 FULL_SHARD 路径 |
| CPU offload | actor/critic 参数和 optimizer 均开启，保留梯度检查点 |
| 采样并发 | 32 agents；每个 vLLM 实例最多 8 条序列 |
| vLLM 调度 tokens / 显存比例 | 16384 / 0.30 |
| 任务预算 | 32 turns；每次输出最多 4096 tokens；上下文 65536 tokens |
| 观测字符上限 | 32000 |
| 训练采样 | temperature=1、top_p=1、top_k=-1 |
| 验证采样 | temperature=0.7、top_p=0.8、top_k=20、n=1 |
| 超时 | 模型 600s、命令 120s、测试 600s、rollout 3600s |
| 检查和终止 | 保留 checked editor、syntax/submission 检查与最多 3 次格式错误限制 |
| 验证 | 初始全量验证；每 20 步全量；末步全量；val batch=470 |
| 保存 | 每 40 步及末步；最多 3 组 actor+critic；保留 model/optimizer/extra |
| 导出 | 中间 checkpoint 不额外重复导出 HF；末步后导出最终 actor，尚待实际验收 |

“784 步”是完整四遍数据的计划，不是之前的 32 步试验。日志中父类初始化曾打印 780，随后 `FullDatasetRayPPOTrainer` 设置保留尾批，最终明确输出 784。

## 5. 数据与评分器改动

本次数据目录：

`/media/ubuntu/D1/zsj/agent-lightning-runtime/data/swe-smith-training/python-full-v7`

原始数据包包含 6315 条 Python 训练任务和 470 条 Python 验证任务。用户已选择全部 Python 范围；Go 不纳入此 baseline。

此前已批准的 64 条训练排除记录保留：55 条依赖离线环境不可用的在线服务，9 条仅修改测试、恢复测试后不再保留实际生产代码 bug 的任务。相关历史修复不应简单归结为“PPO 算法错误”；需要区分数据、评分协议、测试运行环境与训练适配层。

此次直接相关的 Paramiko 修复：

- 数据中的两个参数化 pytest node ID 被空格截断，造成测试无法正确收集。
- 上游结果解析的非空白匹配也会截断带空格的 node ID。
- `full_python_agent.py` 对已核验的两个 Paramiko ID 做精确映射，并按完整期望 ID 解析测试结果。
- 17 条受影响任务的参考补丁全部通过；14 条空补丁失败。
- 剩余 3 条训练任务的空补丁及参考补丁都通过所有 113 项指定测试，会形成虚假成功奖励。

用户于 2026-09-14 明确批准从新训练中剔除以下三条，保留原始文件及全部验证任务：

1. `paramiko__paramiko.23f92003.lm_rewrite__22j0m5ms`
2. `paramiko__paramiko.23f92003.lm_rewrite__ekwh2bqe`
3. `paramiko__paramiko.23f92003.func_basic__pi2mww3n`

最终排除训练任务 67 条，保留 6248 条。`manifest.json`、`exclusions.json` 记录授权、原始任务哈希、证据路径和证据哈希。生成脚本为 `examples/multiturn_ppo/prepare_paramiko_filtered.py`。保留的训练行及整个验证文件字节未改变，v6 目录未修改。

验证文件 SHA256：`efba07a2e430fc55208053601aef0d90bf1b48f82504c7a70ef397f84210e80b`。

证据目录：

- `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/paramiko-collection-20260913-01`
- `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/paramiko-grading-20260913-01`
- `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/paramiko-original-val-regrade-20260913-01`

没有通过强制 reward、篡改测试断言或重写旧指标使成绩“通过”。旧运行的原始验证指标仍保留原值。

## 6. 已完成的验收

- `tests/examples/test_full_python.py`：25 passed。
- 新数据记录脚本 Ruff 检查通过；启动 shell 通过 `bash -n`。
- 使用新评分器、新数据重新完成 124/124 镜像空补丁/参考补丁对照，`audit.exit=0`。
- 所有 6718 条保留任务（6248+470）的代码分支检查通过。
- 启动 guard 核对数据和评分代码哈希，没有复用旧签名绕过检查。
- 运行时确认主训练进程 GPU mask 为 `4,5,6,7`，agents=32、port=18501；服务健康接口 HTTP 200；18 项最终配置断言通过。
- 已完成真实多轮采样、actor/critic 更新以及第 20 步完整验证。

注意：124 个镜像对照是每个镜像选择代表任务进行测试，**不是对全部 6248 条训练任务逐一执行空补丁/参考补丁评分**；全任务检查主要核对代码分支与修复协议兼容性。因此仍需留意新任务上的评分异常。

启动验收目录：

`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/swe-full-python-envs-speed-20260914-01`

其中 `summary.json`、`signature.json`、`all-task-branches.json` 和每个镜像的结果构成证据。训练目录另有 `startup-verification.json`。

## 7. 加速结果、显存与 ETA 的正确解释

### 验证明显加速，训练整步目前基本持平

| 完整 470 条验证 | 实测耗时 |
|---|---:|
| 旧运行初始验证 | 97 分 34 秒 |
| 独立加速验证试验 | 34 分 54 秒 |
| 新全量运行初始验证 | 33 分 37 秒 |
| 新全量运行第 20 步验证 | 39 分 55 秒 |

主要改动是 32 agents、vLLM 8 条序列/16384 调度 tokens、470 条验证一次排队并持续补充空闲 worker，减少原先分批等待长尾任务的空闲时间。470 是队列/验证批大小，不表示同时运行 470 个 agent。

两组都截取前 19 步，平均值如下：

| 指标 | 旧运行 | 新运行 |
|---|---:|---:|
| 每步采样 | 8.17 分钟 | 6.42 分钟 |
| 每步 actor+critic 更新 | 14.80 分钟 | 15.95 分钟 |
| 整个训练 step，不含验证 | 28.08 分钟 | 27.86 分钟 |
| 每步模型调用数 | 379.89 | 383.95 |
| 每次调用平均输入上下文 | 7055 tokens | 7752 tokens |
| 每步新生成动作 tokens | 101639 | 96233 |
| 每步输入+输出计算 tokens | 2813601 | 3109835 |

新运行采样约省 21%，但平均输入上下文更长，计算 tokens 多约 10.5%，抵消了大部分采样收益。回复输出反而略少，不能描述为“多生成了 10% 的回答”。两组各 608 条任务仅有 61 条相同，且模型轨迹随机，不构成固定工作负载的速度对照。

`perf/total_num_tokens` 来自 `attention_mask` 有效位置总数，包含各次调用的输入上下文和输出；`ppo/action_tokens` 来自 response mask。多轮交互中历史上下文在后续调用里再次参与计算，但不等于这些输入位置都作为动作计算 PPO loss。两边 PPO epochs 都是 1，microbatch 都是 2，没有增加 PPO 训练遍数。

按上述 token 总量粗略归一化，新运行 actor/critic 更新时间分别少约 1.7%/3.2%；这支持计算量变化的解释，但不能替代固定轨迹 benchmark，也不说明所有差异都由 token 数决定。

### 进度条 ETA 不等于纯训练速度

初始验证结束后才创建 `Training Progress`。第 19 步旧运行显示 8:53:28，新运行 8:49:20，这两者不包含初始验证，因此看不到新版在初始验证上节省的约 64 分钟。

之后进度条在验证完成后才 `update(1)`。新运行第 20 步纯训练约 23.8 分钟，再加约 39.9 分钟验证，这一次更新间隔变为约 63.8 分钟，导致平滑 ETA 从约 367 小时跳到 500 小时。不要将这个瞬间的 ETA 作为总工期。

新运行每 20 步验证，旧运行每 196 步验证。即使单次验证更快，新运行更频繁的验证仍有额外总开销；评价全流程速度时必须使用相同频率或将验证单独统计。

### 显存低不代表 GPU 没在计算

11:32 的一次抽样中，两组都在 critic 更新：旧运行显存约 67–69 GiB，新运行约 37–38 GiB，但两组 GPU 计算利用率均为 100%。当步展开的 microbatch 数分别为 136 和 71，上下文长度、所处子阶段以及内存缓存也会改变占用。

截至当时，旧运行每卡历史 nvidia-smi 显存峰值约 72–74 GiB，新运行约 64–65 GiB。13:13 新运行又在采样，显存约 25 GiB。比较必须同时注明阶段、任务量与时间，不能用瞬时显存代表训练峰值。

### 已试过但未用于正式运行的参数

- actor=4、critic=2、所有前向 microbatch=2：长上下文压力测试未通过。
- 默认 allocator、90% 容量预算：actor OOM；开启 `expandable_segments` 后仍 OOM。
- 放开至 100% 容量的诊断测试：进程约占 78.78 GiB，仍需再分配约 4.18 GiB 而失败。这只是诊断配置，没有用于全量运行。
- 压力样本取真实轨迹中的长上下文，最长约 63900 tokens。上述结果不表示短上下文绝对无法使用 4，但固定 4 无法覆盖本次任务预算。
- 先前 actor/critic 都为 4 的测试也曾在 critic 阶段失败；前向 microbatch=4 的数值对照未过既定容差。
- 安装的 VERL FSDP1 critic 路径不读取尝试过的 `reshard_after_forward` 开关，不能只改该参数就声称 critic 已切换分片策略并加速。

用户最终选择全部保留 2；没有把失败试验的 allocator/容量环境变量带入全量作业。

## 8. 日志与只读检查入口

新运行日志目录：

`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-capo-swe-pythonfull-v7-4b-gpu47-20260914-01`

启动器日志目录：

`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/launch-capo-swe-pythonfull-v7-4b-gpu47-20260914-01`

| 文件或目录 | 用途 |
|---|---|
| `trainer.log` | step、rollout、CAPO 更新进度和训练错误；开头配置可能含凭据 |
| `metrics.jsonl` | 每步 reward、完整验证分数、更新指标与计时 |
| `gpu.csv` | 每 2 秒记录 GPU index、utilization、memory、power |
| `controller.log` / `server.log` | agent controller 和 Lightning 服务运行情况 |
| `agent/<rollout_id>/` | trajectory、model.patch、grade.json、test-output 等 |
| `traces/<rollout_id>.json` | rollout 输入、is_train、状态时间戳和事件 |
| `ppo-audit/` | 有限频率的训练张量审计，非每步都保存 |
| `checkpoints/global_step_N/` | 按保存频率产生的 actor/critic checkpoint |
| `run.exit` | 作业退出后写入退出码；缺失本身不能单独证明进程健康 |
| `launcher.log` / `launcher.exit` | 位于启动器目录，覆盖启动检查及最终导出流程 |

SSH 登录后，以下命令仅查看状态：

```bash
RUN=/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-capo-swe-pythonfull-v7-4b-gpu47-20260914-01
screen -ls
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
tail -n 30 "$RUN/trainer.log"
tail -n 2 "$RUN/metrics.jsonl"
df -h /media/ubuntu/D1
```

`completed=... succeeded=... failed=...` 表示 rollout 调度/执行结果，**succeeded 不等于修复成功**。模型修复率看 `val/reward`、`training/reward` 和对应 grade。

第 20、40……步的训练指标会等该步验证结束后一起落盘。因此验证期间 metrics 的最后一行可能仍是上一 step，这不表示参数更新没发生。用户要求验证期间不要不断查询，验证完成后再统计；状态询问时做有针对性的检查即可。

如需定位入口，查看独立工作树下 `examples/multiturn_ppo/run_full_python_ppo.sh`。当前作业已运行，不要再次执行同 tag。该入口默认从原始权重开始，**不是现成的断点恢复命令**；如中断，先核验已完成 checkpoint，再明确配置恢复路径和独立 tag。仅有 screen 存在不算启动验收。

## 9. 磁盘、剩余风险与后续验证

13:13 磁盘剩余约 **587.4 GiB**，相比启动时约 684 GiB 已下降；前四卡出现了首个 checkpoint。此前完整 checkpoint 实测约 107.82 GiB，仅作容量估计，新运行的实际大小仍待首次保存确认。

用户希望每 40 步保存、最多保留 3 组。后续检查应同时考虑旧运行 checkpoint、保存期间临时文件及最终 actor 导出，不能把“保留 3 组”理解为没有瞬时额外占用。用户曾授权清理自己的失败尝试 checkpoint，但不要在未辨认归属及可恢复价值前删除其他作业数据。

尚不能承诺的事项：

- 全量长期训练已收敛或能力已经显著提升。
- 所有任务的评分环境都完全正确；代表镜像测试不能替代逐任务评分验证。
- 新配置完成全程一定比旧配置更快；任务轨迹和验证频率不同。
- 后四卡 checkpoint 保存、保留三个的轮转、断点恢复及最终导出已经通过真实验收。

截至本报告，没有为新 tag 新建或修改定时监控。本会话此前提到的三小时监控针对旧运行；接手时先确认监控对象，不能假设新运行已被覆盖。

## 10. 相关工作记录

以下文件位于本地独立工作树 `D:\ai project\RL学习\agent-lightning-main\agent-lightning-speedtrial-20260913\examples\multiturn_ppo`：

- `SWE_FULL_GPU47_2026-09-14.md`：此次启动时的配置及验收记录。
- `SWE_SPEED_TRIAL_2026-09-13.md`：独立完整验证加速试验。
- `PARAMIKO_AND_TRAINING_SPEED_2026-09-13.md`：Paramiko 评分问题、17 条对照结果及训练耗时分析。
- `ACTOR4_CRITIC2_TRIAL_2026-09-14.md`：三次 actor4/critic2 压力试验及 OOM 证据。
- `SWE_EXPORT_RECOVERY_2026-09-13.md`、`SWE_LOOP_ALIGNMENT_2026-09-13.md`：此前修复背景，阅读时注意其运行 tag、分支及记录时间。

接手者应以本报告说明的运行身份找到实时日志，再决定下一步；无需重新开始调研、重新搭环境或重新启动已经运行的 baseline。
