# Paramiko 评分修复与训练更新加速

所有改动位于 `experiment/swe-speed-20260913` 独立分支及同名服务器部署。前四卡原训练、原数据、原评分日志均未修改。CPU 容器重评分没有调用模型；训练更新压力测试仅使用 GPU 4–7，不保存权重。

## 根因与修改

准备好的 SWE-smith 数据包含两个截断的 Paramiko 参数化测试 ID：

| 数据里的 ID 尾部 | 固定镜像实际收集到的完整 ID 尾部 |
| --- | --- |
| `test_accepts_single_possibly_quoted_argument[quoted` | `test_accepts_single_possibly_quoted_argument[quoted spaced-neil]` |
| `test_SSHConfigDict_as_int_failures[not` | `test_SSHConfigDict_as_int_failures[not an int]` |

在参考修复代码上做 `pytest --collect-only`：原 ID 无法匹配，退出 4；串行、并行及按文件排序均不能修复，排除了“只要关闭 xdist 就好”的猜测。按文件收集则能找到完整 ID。实际并行执行时，只显示 `2 workers [0 items]`，最后退出 5，掩盖了无效 ID 的原因。

采集依据：`runtime/logs/paramiko-collection-20260913-01/`。镜像为数据固定的 `paramiko__paramiko.23f92003`，容器本地 image ID 为 `sha256:cbef5b4e411fc320deb303fdda1687d1a5d9b43594d473daec5e2b505c371c56`。

`full_python_agent.py` 在评分边界添加两个精确、限定仓库修订的别名映射，保留 F2P/P2P 分类，不删除测试、不改测试断言、不修改原始数据。评分记录新增 `test_node_aliases` 以记录实际执行的 ID。

仓库 SWE 示例的状态解析还会截断含空格的参数 ID。因此，在完整 pytest `-rA` 摘要中按预期的完整 ID 匹配结果，避免测试执行成功却仍被计为缺失。没有修改 PPO loss、奖励定义或 CAPO vendor 实现。

## 真实容器验收

`runtime/logs/paramiko-grading-20260913-01/` 对扫描发现的所有 17 条受影响任务执行空补丁与参考补丁，另重放本次验证保存的 13 条模型补丁。

- 17/17 参考补丁成功。
- 14/17 空补丁正常执行测试并失败；3 条训练任务空补丁也通过，详见下节。
- 13 条验证模型补丁中，3 条成功、9 条正常测试失败、1 条仍为 pytest 退出 4；不再有退出 5。不能把剩余退出 4 自动归为环境问题。
- 成功补丁来自 `func_basic__4hw2yayo`、`func_basic__f3obtihd`、`func_basic__ark47nhx`。原先均被错误计为 0 分。
- 因此，仅修正这 13 条保存补丁的评分，本次验证从 65/470 变为 68/470（14.47%）。原配置保存的 13 条补丁也完成了同样重评分，同样增加 3 个成功，从 64/470 变为 67/470（14.26%）；依据为 `runtime/logs/paramiko-original-val-regrade-20260913-01/`。这是重评分统计，不是重跑模型，也没有覆盖原有 metrics.jsonl。两次采样随机性仍存在，不能把相差 1 个成功解释为性能改善。
- `tests/examples/test_full_python.py` 25 项通过，包含含空格测试 ID、前缀误匹配防护、映射范围与原始输入不变的回归测试。

## 单独发现的训练数据问题

以下 3 条训练任务，在测试 ID 修正后，空补丁和参考补丁都通过全部选定测试：

| 任务 | Bug Patch 修改的生产文件 | 数据指定的 F2P |
| --- | --- | --- |
| `paramiko__paramiko.23f92003.lm_rewrite__22j0m5ms` | `paramiko/ber.py` | `TestSSHConfigDict::test_SSHConfigDict_as_int[42_1]` |
| `paramiko__paramiko.23f92003.lm_rewrite__ekwh2bqe` | `paramiko/util.py` | `TestMatchExec::test_may_be_negated` |
| `paramiko__paramiko.23f92003.func_basic__pi2mww3n` | `paramiko/channel.py` | `TestMatchExec::test_may_be_negated` |

这些 F2P 均位于 `tests/test_config.py`。检查准备报告确认没有发生生产代码恢复或嵌入测试修复；现有证据指向任务标签/可复现性问题，不能通过强制修改奖励或测试断言处理。审计程序因此按设计退出 1，而不是宣称全部对照验收通过。三条任务尚未从正在使用的数据中剔除；应在后续数据修订中作为空补丁即可通过的异常单独处理。

完整训练入口的评分器指纹检查仍然保留。修复后的评分器不能直接复用旧的环境审计签名，也没有绕过门禁启动新的全量训练。

## 训练加速分析

原前四卡运行 `capo-swe-pythonfull-v6-4b-20260913-03` 前四个已完成更新的总时间为 6077.5 秒：

- actor/critic 更新约 3502.5 秒，57.6%。
- 模型采样与工具交互约 1348.5 秒，22.2%。
- old/ref log-prob 与 critic 前向约 1136.3 秒，18.7%。其余主要是优势计算和权重同步。

验证使用的更多 Agent/vLLM 并发，同样可以用于训练的采样阶段，但不会直接加速后面的 actor/critic 更新，更不能把验证的 2.8 倍加速当成训练加速。

当前 actor 已设置 `reshard_after_forward=false`，critic 仍为 true。最初考虑把 critic 也改为 false，但读取安装的 veRL `fsdp_workers.py` 发现：actor 的 FSDP1 构造会传入此开关；critic 的 FSDP1 构造直接调用 `get_sharding_strategy(fsdp_mesh)`，忽略了它。只有 critic FSDP2 路径使用此开关。因此不能仅靠修改这个 YAML 参数实现 critic ZeRO2。审计脚本加入显式拒绝，避免把未生效的参数当作对照试验。

`run_update_speed_audit.sh` 在 GPU4–7 使用相同原始模型、固定种子和四条最长的保存训练调用，运行两次真实 actor/critic 更新以覆盖 Adam 状态分配，检查前向一致性、梯度与显存。第一次目录为 `runtime/logs/swe-update-speed-20260913-01/`，其中 baseline 保留，原 critic 候选会被显式拒绝。后续脚本候选改为仅 actor microbatch=4、critic 与全部前向仍为 2，避免再次将两个网络一起调大。这是固定张量压力对照，不能代替完整一步训练吞吐验收。

查明 critic 开关不生效后，停止了仅属于本次试验的 torchrun（已核对 PID、命令行和独立输出路径），取消记录写入该目录 `cancellation.json`。原配置只完成前向一致性检查及部分更新，没有完成两次更新的完整基线，不能报告速度或持续训练显存验收通过。没有启动 actor-only microbatch=4 对照。

独立 `run_speed_trial.sh` 的训练采样默认值改为 32 agents、每实例 8 序列、16384 调度 tokens。这些参数作用于训练 rollout，不只作用于验证；但现有实测只证明完整验证可运行，没有完整 PPO step 的加速实测。训练任务 batch=32、actor/critic microbatch=2、优化 minibatch、PPO epoch、学习率、KL、轮次、上下文、输出预算均未改变。原前四卡运行与原全量启动文件没有切换到这些值。

后续更新阶段候选是仅 actor microbatch=4，或分角色减少 CPU offload；两者都需要同输入更新一致性和最长上下文显存验收。增大全局训练 batch、PPO minibatch，或减少 PPO epoch 会改变优化过程，不作为“性能不变”的简单加速项。当前没有已证明的额外 actor/critic 更新加速。
