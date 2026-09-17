# SWE baseline V2：在线对照准备与执行

## 授权与资源

用户授权继续下一步，并清理无继续价值的 checkpoint。旧 baseline 已停止，PI 在 GPU0-3 保持运行；新验收及在线候选使用 GPU4-7。

已经删除 15 个历史 checkpoint，释放 315.39 GiB，删除后空闲 495.54 GiB：
14 个完成的 0.6B smoke/resume 验收点，以及 `training-capo-swe-full-4b-20260912-01/global_step_4`。
只删除精确清单中的 step 目录，保留 run、配置、日志及 REMOVED 标记；baseline40/80/120、PI40、v6step32、早期SWEstep1 均保留。
证据见工作树 `research/checkpoint_cleanup_verification_20260917.json`。

## 已实现并部署

- `critic_initialization.py`：显式 `critic_head_init=default|zero`，只在创建 Critic 时清零 head，随后照常加载 checkpoint；不修改加载/更新方法。
- 配置由 Ray runtime_env 显式转发，来源写入 resolved config 与 provenance。
- `call_batch_config.py`：区分任务 batch32 与展开后的调用 minibatch128。保留实际配置，只在原 veRL 校验副本中适配“每任务一行”的假设，保留任务整除、worker形状及 strict-padding 检查。
- A/B 入口：两臂同为20步、全量训练数据、初始与step20的470题验证，minibatch128；唯一配方差异是 head 初始化。
- `AGL_CONFIG_ONLY=1` 只生成配置，不启动 Ray、controller、server 或GPU训练；真正训练仍强制镜像/代码签名检查及GPU空闲检查。
- 短试验不导出不存在的“四epoch最终step”Actor，保留完整训练 checkpoint。

两臂已沿正式启动脚本完成 CPU 配置生成，归一化实验路径后仅一个差异：`agentlightning.multi_turn_ppo.critic_head_init`。
日志：`training-preflight-v2-mini128-{default,zero}-20260917-01`，退出码均0。
实际原 veRL 配置校验通过，确认任务 batch 仍32，调用 minibatch为128。

## 验收证据与边界

定向 CPU 回归共33项通过，另26个subtests通过；Ruff、bash -n通过。

四卡 checkpoint 验收使用真实4B模型、正负各16条调用、原CAPO更新、真实Adam状态；执行：更新→保存→参考下一步→扰动→恢复→重演下一步。
验收调用 minibatch32用于构造一次真实optimizer更新，不是后续在线配方的128。

1. `audit-checkpoint-roundtrip-v2-20260917-01`：Actor立即恢复的全部状态与预测逐字节一致，但下一步不一致；FA2默认反向计算非确定。
2. `...20260917-02`：仅诊断启用确定性GPU计算，Actor四rank恢复及下一步全部精确通过。Critic只有最后rank的flat parameter hash失败，其optimizer/RNG/预测全部精确一致；验收脚本扰动了FSDP尾部非模型填充区域。
3. `...20260918-03`：按照PyTorch官方state_dict规则排除`_shard_numel_padded`，重新验收Critic；Actor复用02的通过证据，并校验两次输入/配置/seed相同。实际只有Critic rank3有3个尾部填充元素。两角色各四rank全部恢复/预测/下一步精确通过，`completed.json.passed=true`、`run.exit=0`。没有放宽容差。

这属于真实 worker 状态恢复，不能冒称完整在线 trainer/dataloader/rollout 服务恢复已经通过；后者须在首个在线完整 checkpoint 上继续验收。
确定性计算开关只用于精确复演诊断，未加入在线A/B配方。

## 评分环境审计

P0修改了评分与agent源码，旧审计signature正确拒绝复用。没有改旧signature或绕过训练前检查。
用现有124个缓存镜像重跑空补丁/参考补丁审计，2个CPU/Docker并发，不下载镜像、不占GPU。
新输出：`swe-full-python-envs-reliable-v2-20260918-01`。
124/124个镜像全部空补丁0、参考补丁1，ready=true，run.exit=0。
实际训练仍检查新signature、镜像ID及全部任务的分支/保护路径。

## 在线候选

计划先B（零head），再A（默认head），各20步；任何失败/行为门槛退出会中止顺序运行。
固定：Qwen3-4B-Instruct-2507、task batch32、call minibatch128、microbatch2、Actor/Critic LR=1e-6/1e-5、KL=.001、gamma=lambda=1、无LR warmup/无Critic预热。
不使用离线诊断训练后的权重。两臂均从原模型开始，resume_mode=disable。
未通过真实worker恢复与新版环境审计之前，不允许顺序启动脚本开始在线训练；这也不是全量训练授权标志。

## 当前执行状态

真实worker恢复和124镜像审计已通过；4个本轮可再生临时验收权重已清理，31份证据文件SHA256前后不变。清理后空闲494.04 GiB；共享盘空间会随其他任务变化。
临时验收权重清理后保留全部逐rank hash、恢复证据、completed与日志；它们不是线上训练恢复点。

2026-09-18 00:17（北京时间）已启动顺序A/B试验，screen为`agl-ab-v2-gpu47-20260918-01`。
00:20再次确认正式启动的完整分支检查输出`FULL_PYTHON_READY`，train6248/val470、124镜像；真实Ray配置校验通过，确认task batch32/call minibatch128、总步数20，四卡worker开始加载4B模型。
00:22确认`CRITIC_HEAD_INITIALIZATION mode=zero`，head为score、2561个参数；四个vLLM服务完成启动，`AglRolloutManager completed=0/470`表明已进入完整初始验证。此时尚未进行在线PPO更新。

- Pair日志：`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/ab-pair-20260918-01/run.log`
- 先运行B：`training-capo-swe-v2-mini128-zero-4b-gpu47-20260918-01`。
- B正常完成后运行A：`training-capo-swe-v2-mini128-default-4b-gpu47-20260918-01`。
- 两者使用GPU4-7，PI的GPU0-3进程保持运行。

尚无新在线训练效果结论；应比较完整初始/step20验证、正奖励数量、实际optimizer更新、Critic MSE/EV、KL与异常终止比例，再决定是否全量。首个线上完整checkpoint还需trainer/dataloader/rollout级恢复验收。

## 路径与命令

Windows源码：`D:\ai project\RL学习\agent-lightning-main\agent-lightning-baseline-v2-20260917`。
远端部署：`/media/ubuntu/D1/zsj/agent-lightning-baseline-v2-20260917`，运行日志根：`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs`。
逐次用 `research/p2-sync-files.txt` 限定清单，WSL `rsync -anvz` 预览后 `-avz` 部署，不使用`--delete`。

四卡最后一次验收启动命令：

```sh
screen -dmS agl-checkpoint-roundtrip-v2-gpu47-20260918-03 bash research/run_checkpoint_roundtrip.sh \
  /media/ubuntu/D1/zsj/agent-lightning-runtime/logs/audit-checkpoint-roundtrip-v2-20260918-03 \
  /media/ubuntu/D1/zsj/agent-lightning-runtime/logs/audit-checkpoint-roundtrip-v2-20260917-02/actor
```

在线试验启动命令（在上述远端部署目录执行）：

```sh
screen -dmS agl-ab-v2-gpu47-20260918-01 bash examples/multiturn_ppo/run_reliable_ab_pair.sh \
  20260918-01 \
  /media/ubuntu/D1/zsj/agent-lightning-runtime/logs/audit-checkpoint-roundtrip-v2-20260918-03 \
  /media/ubuntu/D1/zsj/agent-lightning-runtime/logs/swe-full-python-envs-reliable-v2-20260918-01
```

本地实现提交：`e602e21`，已部署，尚未push到GitHub。
