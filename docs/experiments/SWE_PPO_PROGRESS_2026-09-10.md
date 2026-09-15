# 四卡 SWE PPO 中期检查（2026-09-10）

运行：ppo-baseline-four-1p7b-20260910-01，GPU 0–3，Qwen3-1.7B。
远端目录：/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-ppo-baseline-four-1p7b-20260910-01。

## 决定

未启动全量训练。用户授权条件是 baseline 顺利，而当前模型行为已退化。现有 32 步 baseline 仍运行，本次没有终止或重启它，也未修改训练代码、奖励或模型配置。

## 已观察证据

- 检查时完成 17/32 步，68 个训练 episodes，训练奖励全部为 0。
- step 0、8、16 验证每次 6 个任务，奖励全部为 0。
- 86 个已判分轨迹（含上述训练和验证）成功修复 0 个；5 个非空补丁，1 个补丁被规则拒绝。
- 按判分文件时间取最早 10 个轨迹：79 次调用中 77 次解析出 action，39 次命令返回 0，2 次 length 格式错误。
- 最新 10 个已完成轨迹：80 次调用，0 次解析出 action，80 次 length 格式错误。原始输出可见重复叙述、虚构目录观察和无完整命令。
- 第 6 步起输出长度明显增长；第 9–14、16–17 步平均回复 768 tokens，截断比例 100%。
- actor/critic 梯度日志有正常数值，暂未见 OOM 或退出文件；这仅说明执行链路运行，不代表学习有效。
- step 8 和 16 均存在 actor、critic 的全部 4 个 rank 模型文件。本次只检查文件数量，没有重新加载验证优化器内容。
- 已有 train/val reference controls 奖励均 1，bug controls 奖励均 0。这支持判分能区分已验证样本，不能替代全量各仓库验证。

## 全量准备缺口

原始 train_dataset_mixed.jsonl 共 6343 条，val_dataset_filtered.jsonl 共 474 条；训练数据涉及 132 个 repo/image_name。
Docker 目前只有 1 个 SWE-smith 仓库镜像（exceptiongroup），其余 3 个 SWE-bench sympy 镜像不能替代缺失的 SWE-smith 镜像。
当前 SmithDockerAgent 要求 image 字段为固定 digest，使用专用 pytest 路径和最多 200 个测试的 pilot 限制；原始全量数据使用 image_name，尚不能直接交给该 worker。
D1 检查时剩余约 570 GB；镜像准备前需要核算增量磁盘需求。

## 下一步建议（尚未执行）

先定位/修复 PPO 稳定性，再从原始 Qwen3-1.7B 做小规模对照，不能从退化检查点直接扩量。
需对学习率、critic 预热和约束策略逐项验证；零奖励条件下未收敛的 value baseline 仍可能产生非零策略梯度，但目前证据不足以把它认定为唯一根因。
全量启动前另需准备镜像 digest 清单、数据适配和逐仓库判分验证。保留既定终局二值奖励，不以放宽判分掩盖零成功率。

本次仅通过 SSH 读取 metrics.jsonl、trajectory.jsonl、grade.json、检查点目录、原始数据统计和 Docker 镜像列表；未下载远端数据、轨迹或权重，本文为汇总。
