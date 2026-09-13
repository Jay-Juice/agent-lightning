# 后四卡独立加速试验

用户授权：仅在 GPU4–7 测试加速，前四卡全量训练不动。全量验证仍覆盖470条，新的频率为每20步验证、40步保存、最多3份；用户负责空间清理。本分支不自动替换原训练或启动第二个长期任务。

## 隔离

- Windows 独立 worktree：`D:/ai project/RL学习/agent-lightning-main/agent-lightning-speedtrial-20260913`，分支 `experiment/swe-speed-20260913`，起点 `6792b77`。
- 服务器独立源码：`/media/ubuntu/D1/zsj/agent-lightning-speedtrial-20260913`，按 WSL rsync dry-run 后同步。原源码和原进程未修改。
- 试验GPU：4,5,6,7；SWE服务端口18501，Ray目录`/media/ubuntu/D1/zsj/ray-speed-0913`。原运行保持GPU0–3、18401及原Ray目录。
- 模型、环境、原始数据只读复用；临时测试只写独立runtime日志/数据，不保存模型权重。

## 已测基线与验证策略

原运行 `capo-swe-pythonfull-v6-4b-20260913-03` 的首次完整验证从20:15:25到21:52:59，470条耗时5854.4秒（97分34秒）。第一步训练1178.1秒；这一时间不能代表长期平均。按该步估计，每20步验证的额外耗时约25%，约占训练加验证总时长20%。

固定上游 `upstream/main` 的 SWE 示例默认 `test_freq=16`；`--max-val-instances` 默认None，允许显式限制验证数量。父训练器会遍历传入的整个验证集。用户选择继续全量，本试验不采用96条快速验证方案。

验证当前每32任务有一次批次等待，最后少数任务可能让大部分GPU空闲。检查 `_rollout(is_train=False)` 确认验证仅聚合指标，不构建训练张量，故候选从最初的batch128进一步改为一次排队全部470条；并发仍限制32agents，每vLLM实例8序列、16384调度tokens。排队470不等于同时推理470。单任务32轮、4096输出、65536上下文、评分协议与模型不变。先实测再决定是否采用；更多并发不保证更快。

## Microbatch 扩展

仅本分支把适配配置检查的训练microbatch上限从2扩展到4。vendor CAPO复制代码不修改。实际复制的actor/critic更新循环使用标量可微模型验证dummy零损失、短尾批和microbatch1/2/4的归一化，9项测试通过；Ruff、shell语法检查通过。首次扩展测试中测试输入仍残留3行shape，已修正测试数据构造，再跑全部通过；这不是运行中原训练的故障。

真实四卡压力测试复用 `audit_capo_long_context.py`：固定相同真实行数进行不同microbatch的比较，初始化原始4B，检查批量/逐条有效token输出、两次更新后的梯度和显存，并记录更新后输出用于对照。每进程显存上限90%，不保存checkpoint。

- `logs/swe-speed-micro4-long-20260913-01`：22:32在后四卡启动，最长4条真实历史轨迹，inference/train microbatch=4，actor Zero2，critic/ref cohort。22:37因批量/逐条输出比较未过容差退出：14/16384元素超限，最大绝对差0.0625；尚未进入训练反向。日志未记录具体role，不能归因于某个网络，也不能据此断言模型效果下降；不放宽容差以掩盖结果，暂不采用前向micro4。
- `logs/swe-speed-trainmicro4-long-20260913-02`：22:39启动，前向保持2，只测试训练micro4，仍使用4条固定真实最长轨迹。三个网络的batch2/逐条有效token前向差异均0；22:45在critic更新中超过90%显存预算而OOM，exit1。进程占用70.91 GiB、上限71.22 GiB，申请4.18 GiB失败，物理卡仍有8.17 GiB空闲；不能将其描述为已经证明80GiB物理容量绝对不够。没有放宽显存预算后硬上正式训练，当前候选仍保留训练/前向micro2。
- 固定任务速度试验输入：`runtime/data/swe-smith-training/speedtrial-20260913`。32条来自原训练第一批；额外的`train-seeded.jsonl`仅给对照试验设置按任务ID确定的随机种子，原训练没有这些种子，因此不能把对照奖励与原先7/32当作严格同随机数比较。
- `run_speed_trial.sh`：一次32任务PPO更新，跳过验证和checkpoint，独立日志。
- `run_speed_validation.sh`：只评测完整470条，默认更大验证batch及并发，不更新权重或保存checkpoint。

## 完整验证加速实测

22:46:21 启动 `capo-swe-valfast-4b-20260913-01`，screen `agl-speed-valfast-4b-01`；物理GPU4–7，端口18501，验证batch470、32agents、每vLLM实例8序列/16384调度tokens。microbatch保持2，使用原始4B，完整470条文件未改。日志为 `runtime/logs/training-capo-swe-valfast-4b-20260913-01`。已正常退出（run/launch exit均0），470个唯一任务全部覆盖，470条轨迹均为验证，没有 optimizer 更新或 checkpoint。验证实际耗时2093.65秒（34分54秒），相对原5854.39秒快2.80倍，耗时减少64.24%。四卡显存峰值分别25.41、27.81、27.43、25.76 GiB；验证时段平均GPU利用率48.6%、62.5%、61.4%、44.5%。

原始 metrics 记录本次65/470成功、原配置64/470成功。随后查明 Paramiko 测试 ID 被空格截断，修复后重评分两次保存的补丁，分别为68/470与67/470。原始日志未覆盖。详见 [Paramiko 修复与训练加速报告](PARAMIKO_AND_TRAINING_SPEED_2026-09-13.md)。验证采样本身有随机性，相差1个成功不能解释为模型性能提升；本试验没有更新模型。

## 保存频率

本分支全量入口已设 `test_freq=20, save_freq=40, max_actor_ckpt_to_keep=3, max_critic_ckpt_to_keep=3`，最终一步仍验证/保存/导出。原运行继续196步验证/32步保存/保留1份。已有4B checkpoint实测约107.82 GiB；新全量入口省去每次HF重复导出，实际大小待写出验证。本次短测不产生这类权重文件。
