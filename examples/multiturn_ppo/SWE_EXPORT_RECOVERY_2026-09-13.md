# SWE 补丁拒绝分支修复与全量恢复

## 故障与影响

2026-09-13 19:52 检查确认，运行 `capo-swe-pythonfull-v6-4b-20260913-02` 已于当天 03:45 退出，`run.exit=1`。初始验证产生 287 份评分，其中 45 个 reward=1；第 288 条任务没有评分，尚未进行 PPO 更新，没有 checkpoint。287 条的部分结果不能作为完整 470 条验证成绩。

失败 rollout 为 `43ac12720c2f44ba97434b22345b8fc2`，任务为 `jaraco__inflect.c079a96a.combine_file__1m7cawal`。控制器堆栈指向 `SmithDockerAgent.run()` 保存 `sandbox.json` 时读取不存在的 `box.excluded_patch_paths`。

根因属于我们新增的 Docker/全量 Python 适配代码：`FullPythonSandbox.export_patch()` 在内嵌测试保护检查失败时提前返回，没有进入父类为导出排除路径赋值的代码。调用方随后无条件读取该字段，抛出 AttributeError，零奖励事件未发送；训练器按既有的缺失评分保护中止验证。不是 PPO 优化算法或显存不足造成的中断。

## 修复与验证

- Sandbox 创建时初始化 `excluded_patch_paths=[]`。
- FullPythonSandbox 每次导出开始清空该列表，使提前拒绝路径也提供完整元数据，并避免复用上一次提交的排除列表。
- 不改变保护规则、测试集合或奖励。被拒绝补丁仍返回零奖励；真正的评分基础设施故障仍明确失败，不伪装成零奖励。
- 新增首次导出、重复导出旧状态、内嵌测试被修改、源文件语法损坏的回归覆盖；执行循环测试确认写出 sandbox/grade 并发送 reward=0。
- 服务器真实依赖环境中 71 项相关测试通过，Ruff 与全量入口 shell 语法检查通过。Windows 本机缺少训练依赖且 pytest 临时目录访问失败，未将本地测试算作通过。
- 使用失败任务保存的 32 条模型回复，在真实 Docker 中完整重放，正常结束并写出 `invalid_embedded_test_source`、reward=0 和一次零奖励事件。证据：`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/swe-export-rejection-replay-20260913-02/result.json`。这是录制回复重放，模型请求和奖励接收使用替身，不是新的模型评测。第一次重放正常返回零奖励，但诊断脚本误将预期拒绝类型限定为 `embedded_test_change`；修正诊断断言后第二次完整通过，两次证据均保留。
- 新代码独立复验全部 124 个评分环境，124/124 通过，exit=0，目录 `logs/swe-full-python-envs-20260913-09`。未复用旧源码的验收签名。42 个本地实验源码、脚本和配置与服务器 SHA-256 一致；未覆盖远端独有文件。

## 恢复配置

用户授权修复后重启全量。旧运行和原始日志保留；没有训练 checkpoint，因此新运行从原始 Qwen3-4B-Instruct-2507 开始，不把部分验证当作已训练模型。

目标运行 `capo-swe-pythonfull-v6-4b-20260913-03`，screen `agl-full-v6-4b-20260913-03`，端口 18401。保持物理 GPU 0–3、6251 train / 470 validation、4 个完整 epoch / 784 次更新、batch32、每卡 microbatch2、16 agents；32轮/4096输出/65536上下文。模型、PPO/GAE、学习率、KL、精度、采样与保存频率保持原配置。

20:05 预检查：前四卡均空闲，D1 剩余约 482.76 GiB，高于入口要求的 360 GiB。未删除 checkpoint 或其他人的文件、进程。

20:10 已创建独立 screen 并启动全量入口，启动前再次确认 GPU0–3 空闲、18401 端口可用、没有同名运行；D1 空闲482.82 GiB。20:11 全任务分支检查已完成89个镜像、3986条任务，仍在启动前检查阶段。每三小时巡检 `3-swe-ppo` 已更新为新 `-03` 并确认 ACTIVE；这仅确认调度配置，不能证明此前的巡检执行过或今后一定触发。

实际训练日志：

```bash
tail -f /media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-capo-swe-pythonfull-v6-4b-20260913-03/trainer.log
```

启动前检查日志：`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/launch-capo-swe-pythonfull-v6-4b-20260913-03.log`。

## 启动确认（20:16）

- 全部 6721 条任务的分支检查通过，正式入口输出 `FULL_PYTHON_READY`；训练器明确输出 `196 batches/epoch, retain tail, total steps=784`，actor/critic 优化器也记录784步、warmup0。
- 模型已加载，真实初始验证正在进行。20:16:26 第一批已有14/32执行完成，执行失败0；14份评分中2份reward=1。控制器未出现 Traceback/AttributeError，未生成退出码文件。这里只是启动快照，尚未完成470条验证，也未进行PPO更新。
- 20:15:22 前四卡各约25089 MiB显存；这是初始验证占用，不代表后续训练峰值。D1空闲约482.78 GiB。
- `provenance.json` 确认 GPU mask为`0,1,2,3`、16 agents、全部轮次/上下文/超时/提交检查配置与此前配方一致。未启动其他显卡任务。
- 修复提交 `8d70077` 已推送并通过远端 main SHA 校验；本节启动快照另随文档提交维护。
