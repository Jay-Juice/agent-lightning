# 全量 Python baseline：准备与排查进度

更新时间：2026-09-13 00:58（Asia/Shanghai）。

**正式全量训练尚未启动。四卡加速短测已完成；最新环境验收已结束，123/124 通过，Tenacity 仍失败。后台没有自动修复或训练进程。**

用户当前要求先提交和推送已有代码、文档。Tenacity 的具体原因尚未完成诊断，不能把它归因于模型，也不能声称已经解决。先前询问的 15 条损坏测试任务仍保留，待确认处理口径。

## 训练范围与待确认项

原始 Python 数据为训练 6315、验证 470、124 个固定 digest 的镜像。计划从原始 Qwen3-4B-Instruct-2507 开始，4 个完整 epoch，batch 32，保留尾批，仅使用物理 GPU 0–3。原始范围对应 792 个 PPO step，不能把此前 32 条任务短测称为全量。

已确认 Patsy 的 15 条训练任务直接改写测试函数或断言，而非只注入生产代码缺陷。例如把测试预期的 ValueError 改为 TypeError，或重写整段测试。完整 ID、代码对照位置保存在服务器 `logs/swe-test-mutation-scope-20260913-01.json`。这 15 条仍保留，等待用户决定；预检会阻止带此冲突的正式训练启动。

用户已明确批准剔除另 55 条依赖在线服务的训练任务：gTTS 8 条、Safety 47 条。它们所需的测试与参考修复中失败的联网测试相交；清单保存任务 ID、原始行 SHA-256、失败测试节点、证据日志路径及哈希，文件为 `logs/swe-approved-network-exclusions-20260913-01.json`。原始发布文件和旧版准备数据没有修改。

当前候选数据是 A800 `agent-lightning-runtime/data/swe-smith-training/python-full-v5`：**6260 条训练、470 条验证、124 个镜像**。每 epoch 196 批，四轮共 **784 步**。若用户再批准剔除上述 15 条，则训练 6245 条，仍为 784 步；最终数目还取决于剩余完整审计结果。准备器拒绝重复、未知、验证集或原始行哈希不匹配的剔除记录。正式入口按最终 manifest 计算每轮验证间隔及最后模型导出路径。

## 四卡短测的实际结果

运行：`training-capo-swe-fullproto-micro2-4b-20260912-01`，2026-09-13 00:06 前后正常退出，`run.exit=0`。本次只运行 1 个 PPO step，32 条混合项目训练任务、6 条旧验证任务，不保存模型。

- 从原始 4B 采样，真实修复成功 **3/32**；更新后验证成功 **1/6**。
- actor/critic 梯度分别约 5.49 / 211.53，均有限，actor 确实更新，更新后权重同步完成。
- 单步时间 1289.94 秒（约 21.5 分钟）；前一次 microbatch 1 短测为 1816.94 秒（约 30.3 分钟）。两次任务组成略有不同，不作为严格的加速比例实验。
- actor 更新 323.31 秒，critic 更新 351.21 秒；实际峰值 allocated 38.11 GiB、reserved 46.64 GiB。
- 更早的极限压力测试使用两条真实长序列 63900 / 59742 token，四卡连续两次更新，覆盖优化器状态初始化后的峰值 **65.49 GiB**，无 OOM。推理 batch 2 与逐条推理在有效响应 token 上的输出一致。
- 38 条训练/验证轨迹共 **379 次调用**：消息历史、prompt token、响应内容全部逐一匹配，奖励与独立评分结果一致。证据：`logs/swe-fullproto-micro2-trace-integrity-20260913-01.json`。
- 18 次输出不符合单动作格式，另有模型补丁导致测试崩溃或超时。这些仍是任务失败，不能把进程正常完成当成修复成功。

**没有验证性能提高的证据。** 短测说明采样、评分、更新和权重同步能工作；不能据此保证长训练有效。

## 已修复的适配问题

1. 全量 dataloader 保留尾批，重新计算优化器总步数；验证按全部批次加权汇总，缺失奖励直接报错。
2. Arrow 全项目覆盖率门槛不适用于指定测试子集；关闭覆盖率统计，保留全部选定测试和其他项目配置。
3. cantools、Dask 强制彩色输出导致状态解析为空；关闭终端颜色并去除 ANSI 样式。
4. stackprinter 镜像的安装元数据被误判为脏目录；只保留 `.egg-info` 白名单元数据，仍拒绝其他脏文件和跟踪文件变更。
5. Tornado 的原生 unittest 名称、继承测试别名和原生警告/日志检查需要专门适配。使用其项目 runner，15 条任务全部通过空补丁失败、参考修复成功的对照。
6. Parso 和 PyQuery 少数实际缺陷位于 conftest helper 中。允许受限的谓词及路径修复，保持 pytest hooks、测试函数和其他结构。正确补丁能够导出并评分，删除测试等篡改仍被拒绝。
7. 部分源码内含 doctest 或测试函数，镜像删除 F2P 文件时连生产源码一同删除；评分时整份恢复又会覆盖模型修复。现在只从 **Bug Patch 提交**恢复带缺陷的生产源码，定义私有干净导出基线，并保持内嵌测试内容不变。没有从参考修复提交恢复生产代码。PyQuery、Parso、Patsy、Pinyin 六个代表性对照均通过。
8. PyQuery 的一个真实联网测试访问俄语维基百科。保存实际公共网页，固定 SHA-256，在评分容器的 loopback HTTP 服务内回放；保持原测试、原断言和 `network=none`。参考修复 106/106 F2P、38/38 P2P，导出补丁也得到奖励 1；空补丁仍为 0。记录及网页位于 `data/swe-smith-training/http-fixtures-v1`。
9. Furl 的镜像 Python 3.10.16 与旧项目要求的 URL 解析行为不一致，参考修复仍有 3 个 P2P 失败。对该项目的临时容器使用原封不动的 CPython v3.10.0 urllib.parse，空补丁 0/1 F2P、72/72 P2P，参考修复全部通过。主机环境未改变；原版代码、许可证和校验值在 `vendor/cpython310`。CPython 官方文档记录了相关版本的 [URL 解析行为变化](https://docs.python.org/3.10/library/urllib.parse.html)。
10. Mido 原始测试节点带 `../dev/tests/`，与实际仓库 `tests/` 不符；用逐条验证的别名映射规范化，保留路径防护。6 条任务全部通过正反对照。
11. Scrapy 镜像预先删除了无关测试文件；仅从镜像当前 HEAD 恢复这些已有测试，再 checkout 任务，不恢复生产修复。Pyfiglet 保留镜像自带字体资源，拒绝符号链接及非白名单文件，并从补丁导出排除。
12. Pydicom 缺少官方 DICOM 测试资源；按项目登记的 SHA-256 校验原始文件，在临时容器的 HOME 缓存安装。Scrapy、Pyfiglet、Pydicom 三个环境均通过空补丁失败、参考修复成功的对照，证据 `logs/swe-resource-controls-20260913-01`。
13. Sunpy 的镜像闰秒表过期导致警告被当作错误；安装已校验的官方 IERS 表，当前有效期至 2027-06-28，仅修改临时容器。Modin 的测试自行启动 Ray，xdist 多进程导致竞争；保持原测试，使用单 pytest 进程、2 CPU、8 GiB 内存、1 GiB shared memory，并限制 Ray 线程。两者正反对照均通过：`logs/swe-sunpy-modin-controls-20260913-01`。

以上不是“模型太弱”可以解释的现象，也不能统一归因于 Lightning。问题分别涉及自定义评分/导出适配、全量训练适配以及数据镜像本身；修复后模型能力与学习效果仍须由后续正式实验测量。

## 算法与速度配置

CAPO vendor commit：`e8407baea32fb36191f029f0a4666bf681bdf1ca`，复制的更新循环未改写。当前运行的是 token GAE + vanilla PPO，并不是论文全部 CAPO action-ratio 方案，也不是 SWE 示例原本的 GRPO。

保持 gamma=1、lambda=1、actor LR=1e-6、critic LR=1e-5、KL loss 系数 0.001、clip 0.2、warmup 0；32 轮动作、4096 输出 token、65536 上下文预算不变。

已验收并写入正式入口的速度配置：16 个本地 agent、vLLM 每实例最多 4 序列/8192 batched tokens、actor `reshard_after_forward=false`、actor/critic 训练 microbatch 2、actor/ref/critic 推理 microbatch 2。dummy padding 与短 microbatch 的损失归一化有实际复制更新循环的数值测试。

评分与 PPO 核心回归测试 **58 passed**，Ruff 通过；本次提交前补充代理采样参数和 rollout 模型版本回归后，共 **78 passed**（仅依赖库弃用警告）。CPython 原版文件排除格式化，保持校验值不变。这轮评分/数据适配没有修改复制的 CAPO PPO 核心更新公式。

## 当前运行与磁盘

当前版本完整验收原 screen：`swe-full-python-envs-v6-0913`；日志目录 `agent-lightning-runtime/logs/swe-full-python-envs-20260913-06`。它使用部署后的资源修复和 6260/470 数据，4 路 CPU/Docker 并发，不使用 GPU。最终 **123/124 通过，pipeline.exit=1**，进程已结束。剩余失败任务为 `jd__tenacity.0d40e76f.combine_file__69rgwhxp`，参考修复评分仍为 0，原因待查。

上一版 V5 最终完成 124/124，114 通过；保留旧失败，不冒充当前源码的验收。全部验证任务另有独立检查 `logs/swe-full-validation-readiness-20260913-01`，最终 **21/22 通过**，失败同样是 Tenacity。这些是环境对照结果，不是模型验证成绩。

旧 V4 审计因评分器修复而按授权停止，保留全部结果；它检查 40 个镜像、38 个通过，另外两个是已经针对性修复的 PyQuery/Furl，不能作为最终全量验收依据。00:58 检查时上述环境验收及镜像预取 screen 均已退出，GPU 0–3 空闲，同门 GPU 6、7 的任务继续运行。

最近检查 D1 可用约 **513 GiB**；前四卡短测进程已全部退出，后四卡同门任务未动。正式入口至少要求 360 GiB 空闲。单个中间完整 FSDP 检查点预计约 91 GiB，保留最新 actor/critic；每 32 步保存，末次另行导出 HF 模型。实际短测 tensor audit 约 0.53 GiB/步，轨迹与 agent 日志约 0.03 GiB/38–44 条任务，后续空间估算应以全量任务长度为准。

此前授权删除的两个旧检查点共约 215.65 GiB；有效 step 4 检查点约 107.82 GiB 及所有日志仍保留。后续其他空间变化不记为本次清理成果。

## 正式启动前仍需完成

- 全部 124 个镜像通过当前源码的正反对照。
- 全部任务分支、参考修复导出兼容性检查完成；确认损坏测试任务的处理口径。
- 生成最终数据清单与总步数，再以 `run_full_python_ppo.sh` 启动。
- 核对实际 screen、端口、数据哈希、配置、GPU 占用和初始验证进度后，单独写正式安全启动汇报。

本文件是准备与排查记录，不能代替尚未发生的正式启动汇报。
