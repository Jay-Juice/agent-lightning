# SWE PPO baseline 超时修复与 step 40 恢复 Handoff

更新时间：2026-09-15 13:31，Asia/Shanghai。运行状态是该时点的只读快照。

## 结论

GPU 4–7 的 v7 Qwen3-4B baseline 已从原运行的 `global_step_40` 完整 checkpoint 恢复。actor、critic、optimizer、RNG、LR scheduler 和 dataloader 状态均走 VERL `resume_path` 路径加载；新运行已进入恢复权重的 470 条完整验证，13:30 检查时完成 1/470，失败 0。没有从原始模型重新训练，也没有把旧运行未保存的 step 41–59 指标接入新曲线。

本次修复针对单个长尾 rollout 在控制器 3600 秒硬超时处被强杀、来不及评分而导致整场训练退出的问题。模型、数据、PPO、GAE、学习率、采样与任务 token/turn 配置保持原 baseline。

## 原故障

原运行：

- tag：`capo-swe-pythonfull-v7-4b-gpu47-20260914-01`
- 工作树：`/media/ubuntu/D1/zsj/agent-lightning-speedtrial-20260913`
- 退出：第 60 步完整验证，`run.exit=1`
- 最后有效 checkpoint：`global_step_40`

唯一失败 rollout：

- rollout ID：`8cea91f95c3b4733aecb7f578866177a`
- task：`bottlepy__bottle.a8dfef30.func_pm_ctrl_shuffle__22hdlhws`
- 状态：`local subprocess timed out`
- 时间：06:44:30 启动，07:44:30 被控制器 SIGKILL

该 rollout 保存了 27 轮。模型反复执行无效的 `ls -la | grep -i request`，并交替产生达到 4096 token 上限的不完整回复。格式错误与合法 shell action 交替出现，因此连续三次格式错误终止条件没有触发。控制器在 3600 秒整强杀 agent 子进程，agent 尚未来得及导出补丁、运行独立评分和发送 reward。

第 60 步验证最终为 470 个 rollout terminal，其中 469 succeeded、1 failed；训练器发现奖励覆盖不完整后拒绝报告分数并退出。不是 OOM、NaN、PPO loss 或 checkpoint 损坏。

## 代码修复

本地权威工作树：

`D:\ai project\RL学习\agent-lightning-main\agent-lightning-speedtrial-20260913`

提交：`5fa1841` — `Gracefully finalize long SWE rollouts and preserve recovery state`

已推送 GitHub 分支：`experiment/swe-speed-20260913`。

改动：

1. `smith_docker_agent.py`
   - 新增 `SMITH_AGENT_WALL_TIMEOUT`。
   - agent 从进程入口开始计时；达到预算后不再发起下一次模型请求，记录 `wall_time_budget`，随后照常导出当前补丁、独立评分并发送真实二值 reward。
   - 没有把基础设施失败强行改成 reward 0，也没有修改连续格式错误逻辑。

2. `run_checked_swe_ppo.sh`
   - Actor 交互 wall budget 保持原来的 3600 秒。
   - 控制器 hard timeout 从 3600 调整为 5400 秒，为最后一个已开始的 model call、补丁导出、最多 600 秒评分、reward 回传和容器清理留下 1800 秒。
   - 这是可靠性窗口调整；32 turns、4096 response、65536 context、模型请求 600 秒、命令 120 秒、评分 600 秒均未改变。

3. `trainer.py`
   - 训练后验证若仍遇到不完整奖励或其他异常，先保存当前 step 的 actor、critic、optimizer、RNG 和 dataloader checkpoint，再保留原异常退出。
   - 不把不完整验证计为有效成绩。

4. `train.py`
   - 将新的 agent/hard-timeout 环境变量写入 provenance。

5. `test_smith_docker_loop.py`
   - 新增 wall timeout 回归：达到 agent budget 后不再请求模型，但仍评分并发出 `wall_time_budget` reward event。

## 验证

服务器真实运行环境：

- 定向 loop 测试：9 passed。
- `test_full_python.py`、`test_smith_docker_loop.py`、`test_full_dataset.py`：41 passed，1 个第三方 Ray deprecation warning。
- Ruff：通过。
- `bash -n`：通过。
- Windows 本地 pytest 因已有 pytest 临时目录权限错误未完成，因此不计为通过；代码在 A800 环境重新验证。

修改评分 agent 后没有复用旧签名。新验收目录：

`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/swe-full-python-envs-baseline-timeoutfix-20260915-01`

结果：124/124 代表镜像的空补丁/参考补丁对照全部通过，`ready=true`。正式入口随后重新检查全部 6718 个保留任务分支并通过。

本地修改按 `A800/AGENTS.md` 执行 WSL rsync dry-run，确认仅 5 个显式源码/测试文件且保留相对路径后，才执行实际同步；没有使用 `--delete`，没有覆盖远端日志、数据或 checkpoint。

## 恢复运行

新运行：

- tag：`capo-swe-pythonfull-v7-4b-gpu47-resume40-20260915-01`
- screen：`agl-full-v7-4b-gpu47-resume40-20260915-01`
- port：18501
- GPU：物理 4,5,6,7
- 运行目录：`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-capo-swe-pythonfull-v7-4b-gpu47-resume40-20260915-01`
- 启动日志：`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/launch-capo-swe-pythonfull-v7-4b-gpu47-resume40-20260915-01.log`
- 恢复来源：`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-capo-swe-pythonfull-v7-4b-gpu47-20260914-01/checkpoints/global_step_40`

VERL 日志已经明确输出：

```text
Load from checkpoint folder: .../global_step_40
Setting global step to 40
Resuming from .../global_step_40
```

四个 rank 均记录 actor/critic model、optimizer、RNG 和 LR scheduler 加载。`data.pt` 中 dataloader 已保存 1280 samples / 40 batches 的位置；恢复入口加载该 state，而不是重新从训练数据首批开始。

13:33 状态：

- screen 存在；`run.exit` 不存在。
- 恢复后的完整验证已推进到 112/470 succeeded、0 failed。
- GPU 4–7 各约 25.5 GiB，实际有计算利用率。
- 日志未见 Traceback、OOM、RuntimeError。

## 与原 baseline 的实际配置差异

逐项比较两份 `resolved-config.json`，非路径/运行身份差异只有：

- `agentlightning.rollout_timeout_seconds`: 3600 → 5400；
- `trainer.resume_mode`: disable → resume_path；
- `trainer.resume_from_path`: 指向旧 step 40。

Actor 自身新增 `SMITH_AGENT_WALL_TIMEOUT=3600`，所以原来由控制器硬杀形成的 3600 秒交互边界没有扩展为 5400 秒。其余模型、PPO、训练数据、验证数据、batch、microbatch、LR、KL、gamma/lambda、采样、turn/token/context、验证频率和保存频率保持一致。

## 后续检查

1. 等本次恢复后的 step 40 完整验证结束，确认 470/470 reward 覆盖。该分数是恢复权重的新一次随机验证，不要求逐条等于旧 step 40 的 72/470。
2. 确认随后从 global step 41 更新，而不是从 step 1 或训练数据开头开始。
3. 第 60 步验证时重点检查是否出现 `wall_time_budget`，以及这些 rollout 是否都生成 `grade.json` 和 reward event。
4. 若未来仍发生超时，当前 trainer 会先保存失败 step 的恢复 checkpoint；仍需区分 model/gateway、Docker、评分和 reward 回传故障，不能把缺失奖励当零分。
5. 13:17 启动前 D1 可用约 486.7 GiB。baseline 保留 3 组 checkpoint，PI 运行也会保存 checkpoint，需继续监控磁盘；未经确认不要删除其他运行数据。

本次没有停止或修改 GPU 0–3 的 PI 运行。
