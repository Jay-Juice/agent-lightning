# SWE baseline V2：评分分类修复与重启验收

## 已查明的结果

原B实验`capo-swe-v2-mini128-zero-4b-gpu47-20260918-01`在初始验证后退出，在线PPO更新0步，A未启动。
问题是评分可靠性层只接受退出码0/1，没有实现方案中要求的“有证据的候选失败”归类；不能归因于PPO数值崩溃。

28份原始候选补丁均已在相同任务、镜像、测试集合、600秒、4GiB/2CPU约束下复评分：28个参考补丁全通过，28个候选全复现原退出码（23×4、2×3、2×124、1×137），没有重新采样Actor。137重放有精确容器ID和时间窗口内的Docker OOM事件；该候选将有限循环改成无限生成。

因此原始固定补丁集的完整复核结果为69成功、401失败，69/470=14.68085%。这是离线复核结果；原始run仍保持退出1、原来的28条reward=null，未改写日志或冒称完成在线更新。

复评分输出：`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/audit-v2-grading-recheck-20260918-01`，run.exit=0。
逐题记录见工作树`research/V2_FIXED_PATCH_REGRADING_2026-09-18.md`。

## 修复内容

- 普通0/1结果仍直接交付。对2/3/4/5/124/137等歧义退出，执行同任务参考对照和一次固定候选重放。
- 仅在reference全测试通过、两次candidate均失败且退出码一致、task/patch/test spec/image/资源预算一致时归为`candidate_failed`，最终奖励必须为0。
- 137要求两次候选的容器cgroup `oom_kill`计数增量；124要求真实执行达到原600秒上限，且对照与重放满足资源证据要求。不能仅凭退出码认定OOM或将未知基础设施问题记0。
- 参考失败、候选重放变成成功、两次结果不一致、证据不完整继续requires_review；没有挑选幸运重试结果。
- Docker/HTTP传输错误每阶段最多重试一次，同一任务与patch不变，不重新生成模型轨迹。
- 参考补丁只在agent交互容器关闭后由隔离grader使用，不进入模型上下文。
- 终止原因统计优先使用校验过的`swe-v2 rollout_outcome`，并核对reward reason；缺奖励不再被当成未知终止原因。
- 新增评分模块SHA纳入环境审计signature；容器标签包含episode及评分阶段，便于准确关联。

模型交互预算仍3600秒，单次评分仍600秒；外层安全超时由5400增为8400，覆盖candidate/reference/replay及每阶段最多一次传输重试：3600+6×(600+120)+300=8220秒。它不增加模型turn/token预算。Actor/Critic LR、KL、gamma/lambda、batch/minibatch及20步边界不变。

## 已完成的测试

- 实际Linux环境：112项pytest通过、26个subtests通过；覆盖受控分类、参考失败、成功重放不替换奖励、证据不匹配、未知退出、评分/动作契约、终止统计及启动配置。
- bash -n通过；生产改动Ruff检查通过（保留显式if/else的SIM108风格豁免）。Windows上的16项Linux信号测试跳过，不拿其结果冒充Linux验收。
- A/B正式入口config-only退出均0；归一化实验路径后仅`critic_head_init`有差异，外层timeout均8400。
- 全470条真实轨迹走正式triplet转换、CompletedRollout构建、episode contract及validation metrics聚合，全部通过。28条奖励使用有证据的内存审计覆盖层，所有原始trace哈希前后不变，未用于训练。

轨迹重放：`audit-v2-episode-replay-all-20260918-01`，accepted470/held0/passed=true。
5169次真实调用，1368833输出token；submitted417（88.72%）、turn_budget50、format_errors3；7题含length截断。

## 重启前最后验收

- 新版124镜像审计：`swe-full-python-envs-reliable-v2-20260918-02`，执行中。
- 真实生产自动分类验收：`audit-v2-grading-live-20260918-01`，CPU两并发；先运行两个600秒timeout案例，随后语法、导入、internalerror和OOM案例。每题调用未经替换的`grade_fixed_patch(full_python_agent.grade)`，固定candidate→reference→candidate，6题必须全部candidate_failed/reward0。
- `run_reliable_ab_pair.sh`已增加上述live6及470轨迹重放的启动门槛，核对证据及源码SHA；原有四卡worker恢复证据继续保留。

全部通过后用新pair `20260918-02`，GPU4–7先B再A，各20步，初始及step20均完整470题验证。GPU0–3的PI保持运行。尚未启动新pair；不得将当前审计进程说成在线训练。

## 路径与运行方式

本地权威源码：`D:\ai project\RL学习\agent-lightning-main\agent-lightning-baseline-v2-20260917`。
WSL：`/mnt/d/ai project/RL学习/agent-lightning-main/agent-lightning-baseline-v2-20260917`。
SSH：`C:\Windows\System32\OpenSSH\ssh.exe -o BatchMode=yes -l ubuntu A800`。
远端部署：`/media/ubuntu/D1/zsj/agent-lightning-baseline-v2-20260917`。

部署使用`research/p2-grading-sync-files.txt`精确清单，WSL `rsync -anvz`先预览，再`-avz`；不使用--delete，不修改PI部署目录。

计划在远端部署目录执行以下命令，须先确认最后两项验收成功：

```sh
screen -dmS agl-ab-v2-gpu47-20260918-02 bash examples/multiturn_ppo/run_reliable_ab_pair.sh \
  20260918-02 \
  /media/ubuntu/D1/zsj/agent-lightning-runtime/logs/audit-checkpoint-roundtrip-v2-20260918-03 \
  /media/ubuntu/D1/zsj/agent-lightning-runtime/logs/swe-full-python-envs-reliable-v2-20260918-02 \
  /media/ubuntu/D1/zsj/agent-lightning-runtime/logs/audit-v2-grading-live-20260918-01 \
  /media/ubuntu/D1/zsj/agent-lightning-runtime/logs/audit-v2-episode-replay-all-20260918-01
```
