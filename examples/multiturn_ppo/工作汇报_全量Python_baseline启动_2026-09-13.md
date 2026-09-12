# 全量 Python PPO baseline 启动汇报

状态核验时间：2026-09-13 02:16（Asia/Shanghai）。

**正式运行已启动，正在进行训练前的完整验证集评测。尚未完成 470 条验证，也尚未进行首次 PPO 更新。** 首批 32 条已有 11 条完成并保存评分、轨迹，真实修复成功 1 条；这只是启动快照，不能当作完整验证集成功率或训练提升。

## 数据与验收

- 使用全部有效 Python 任务：6251 train / 470 validation，124 个固定 digest 的镜像。用户批准剔除此前 55 条联网依赖任务及本次 9 条纯测试变异任务，共 64 条。
- 数据目录：`/media/ubuntu/D1/zsj/agent-lightning-runtime/data/swe-smith-training/python-full-v6`。验证文件与 v5 逐字节一致，其余训练行及顺序不变。原始发布文件及旧版本保留。
- 剔除记录：`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/swe-approved-exclusions-20260913-02.json`，包含逐条 ID、理由、原始行 SHA-256、证据路径/哈希及用户授权说明。
- 6 条业务代码与测试共同变异的 Patsy 任务已修复适配并保留，逐题满足空补丁 0、参考修复 1、导出重放 1；详见 [Patsy 修复报告](PATSY_TEST_REPAIR_2026-09-13.md)。
- 当前源码与 v6 数据的完整环境验收 **124/124 通过**，共 248 次代表任务正反评分，进程 exit 0。证据：`logs/swe-full-python-envs-20260913-07`，包含数据/源码签名及每个镜像的结果。
- 启动入口随后检查全部 **6721 条训练/验证任务**的分支与导出兼容性，成功输出 `FULL_PYTHON_READY`：6251/470、124 images、784 steps、4 epochs。检查结果在同一审计目录的 `all-task-branches.json`。
- 本地与服务器 65 个相关源码/配置文件 SHA-256 全部一致；启动脚本 `bash -n` 通过。最近评分适配回归测试 34 项通过，Ruff 通过。

## 实际运行配置

| 项目 | 已核对的值 |
| --- | --- |
| 模型 | 原始 Qwen3-4B-Instruct-2507，`resume_mode=disable` |
| GPU | 仅物理 0、1、2、3；4–7 未启动本次任务 |
| 训练量 | 4 个完整 epoch；每轮 196 批；共 784 次 PPO 更新，保留尾批 |
| Batch | train 32 / validation 32，验证覆盖全部 470 条 |
| 算法 | CAPO 移植的 token GAE + vanilla PPO；独立 critic，gamma=lambda=1 |
| 学习率/KL | actor 1e-6、critic 1e-5；KL loss 0.001；无预热 |
| 任务预算 | 最多 32 轮；单次输出 4096 tokens；上下文 65536；观测上限 32000 字符 |
| 并发 | 16 个本地 agent；每个 vLLM 实例最多 4 条序列、8192 batched tokens |
| Microbatch | actor/critic 训练每卡 2；critic forward、old/ref logprob 每卡 2 |
| 其他加速 | actor 不在 forward 后重新分片；已启用 fused kernels；保留原精度与奖励协议 |
| 验证 | 训练前全量验证，每 196 步再次全量验证 |
| 保存 | 每 32 步保存 model/optimizer/extra，每类保留最新 1 份；最后只导出一次 HF actor |

本次没有为提速缩减训练集、验证集、测试节点、交互预算或更新次数。加速设置沿用此前通过四卡短测和长序列压力测试的组合，未临时提高到未经验证的 microbatch。

日志中父类初始化可能先打印 `Total training steps: 780`；随后完整数据类明确输出 `196 batches/epoch, retain tail, total steps=784`，并同步设置 actor/critic 调度器总步数。实际采用后者，区别是每轮保留不足 32 条的尾批。

## 启动位置与状态证据

- screen：`agl-full-v6-4b-20260913-01`
- tag：`capo-swe-pythonfull-v6-4b-20260913-01`
- 运行目录：`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-capo-swe-pythonfull-v6-4b-20260913-01`
- 启动/预检日志：`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/launch-capo-swe-pythonfull-v6-4b-20260913-01.log`
- 本地服务端口：18401。`resolved-config.json` 保存脱敏后的实际配置；`provenance.json` 保存源码哈希、设备与版本；`datasets.json` 保存完整数据快照。
- 实际权重加载成功，前四卡已处理模型请求；02:16 每卡约 25–26 GiB 显存。此时处于推理评测阶段，不代表后续反向传播峰值。既有长序列压力测试峰值约 65.49 GiB。
- 02:16 检查时无 `run.exit`，运行未退出；首批日志 `completed=11/32, succeeded=11, failed=0`。这里 succeeded 是执行完成数，真实奖励以 `agent/*/grade.json` 为准：11 条已评分中 1 条奖励 1。对应 11 份 trace 已保存。
- 启动前 D1 可用约 505 GiB，入口要求至少 360 GiB。未删除其他文件，未停止同门进程；GPU 6/7 的既有任务继续运行。

启动命令在已核对不存在同名运行后，通过独立 screen 执行；外层记录最终退出码。核心命令如下，**不要在当前运行仍在执行时重复启动**：

```bash
cd /media/ubuntu/D1/zsj/agent-lightning-main/agent-lightning-main
export AGL_TRAIN_TAG=capo-swe-pythonfull-v6-4b-20260913-01
export AGL_TRAIN_PORT=18401 AGL_GPUS=0,1,2,3
bash examples/multiturn_ppo/run_full_python_ppo.sh
```

源码在 Windows 修改，经 WSL `rsync -anvzR` 预览后使用 `rsync -avzR` 部署，没有使用 `--delete` 或直接修改远端源码。后台 screen 会继续运行；本报告只确认启动和首批任务执行正常，不保证后续训练一定提高性能。
