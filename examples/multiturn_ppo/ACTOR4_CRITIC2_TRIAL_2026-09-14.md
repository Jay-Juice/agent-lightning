# Actor microbatch 4 / critic microbatch 2 试验

用户授权在后四卡测试 actor=4、critic=2，其余执行参数采用推荐值。前四卡原训练保持不变。所有源码位于独立分支 `experiment/swe-speed-20260913`。

## 配置与影响范围

- 模型：Qwen3-4B-Instruct-2507。
- actor 训练 microbatch=4；critic 训练 microbatch=2；old/ref log-prob 和 critic 前向 microbatch 均为2。
- 训练任务 batch=32，actor/critic 优化 minibatch=32，PPO epoch=1，学习率、KL、折扣和损失归一化保持原值。
- 全局优化 minibatch 32 在四卡上为每卡8条调用；actor 从4个 microbatch 累积改为2个，optimizer 更新次数不变。浮点计算顺序变化可能产生数值差异，不能预先保证训练质量完全相同。
- 训练采样：32 agents、每个 vLLM 实例8并发序列、16384调度 tokens；单任务32轮、输出4096、上下文65536不变。
- `run_speed_trial.sh` 新增 `AGL_SPEED_ACTOR_MICRO`，可单独覆盖 actor；未指定时仍继承默认训练 microbatch=2，避免无意把 critic 一并改为4。

## 验收顺序

先用现有长上下文脚本测试显存和真实更新，再决定是否进行基线对照与完整32任务 PPO step。压力输入来自保存的真实训练调用，最长63900 tokens，四卡各处理相同的四条最长调用。固定种子20260913、原始模型初始化，包含 actor/critic/reference 并运行两次更新以覆盖 Adam 状态分配。每进程保留原先90%的显存预算，避免比较时同时改变内存分配策略。

首个试验于2026-09-14 00:01在物理 GPU4–7启动：

- screen：`agl-actor4-critic2-0914-01`
- 结果目录：`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/swe-actor4-critic2-20260914-01`
- 命令入口：`AGL_UPDATE_AUDIT_TAG=swe-actor4-critic2-20260914-01 AGL_UPDATE_AUDIT_SKIP_BASELINE=1 bash examples/multiturn_ppo/run_update_speed_audit.sh`
- 本次只启动 actor4/critic2 候选，先检查它能否通过长上下文压力；没有重复运行先前未完成的基线，也没有修改 CAPO vendor 源码。
- 试验结束前不宣称显存验收通过、更新结果等价或训练提速。

用于后续完整 step 的固定32条训练任务已检查，不包含先前发现的3条 Paramiko 空补丁即可通过的异常任务。短测不保存 checkpoint，也不启动新的长期全量训练。

## 结果

第一轮退出1：actor/reference/critic 的前向 batch2 与逐条比较，MAE/max均为0；critic=2 更新通过，随后 actor=4 在63900 token 长上下文的 MLP 分配中 OOM，没有完成 actor 更新或稳态第二次更新。

错误记录：申请4.18 GiB；进程占用70.75 GiB，90%预算允许71.22 GiB；物理GPU仍空闲8.31 GiB；PyTorch allocated56.03 GiB，reserved但未allocated13.99 GiB。不能据此断言整张80GiB卡绝对放不下，也不能宣布该配置可用于正式训练。

第二轮在独立目录 `swe-actor4-critic2-expandable-20260914-02` 重试，同样是 actor4/critic2、相同90%预算、相同数据、精度和优化配置，只添加进程级 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 来排查分配碎片。已退出1：actor 更新的旋转位置编码分配申请1.76 GiB失败；实际allocated69.99 GiB，未使用reserved仅138.72 MiB，进程占用70.85 GiB，物理空闲8.21 GiB。说明碎片显著减少，但90%预算仍不足；不能再把此轮失败只解释为碎片问题。

第三轮目录 `swe-actor4-critic2-capacity-20260914-03`，保留expandable_segments，仅把压力脚本的 `--memory-fraction` 改为1.0，诊断完整物理容量是否够用。已核对进程命令行和CUDA掩码，证据写入该目录 `launch-config.json`。这只是容量诊断，不是正式训练的内存配置建议。普通脚本默认预算仍为0.9；没有修改共享环境或前四卡设置。

第三轮也退出1：actor 更新在 SiLU 激活分配中再次 OOM，申请4.18 GiB；进程占用78.78 GiB，物理空闲仅285.25 MiB；PyTorch allocated77.93 GiB，未使用reserved122.82 MiB；已允许完整79.14 GiB容量。此轮确认当前执行配置下的真实显存容量不足，不能只归因于人为预算或缓存碎片。

三轮均未进入稳态第二次更新，没有完整的更新后探针结果，也没有保存checkpoint。不能据此给出训练质量等价或速度提升结论。没有启动随后完整32任务训练，也没有因为短样本可能装得下就把固定actor4配置用于全量。

## 结论

固定 actor microbatch=4、critic=2 未通过当前约64k上下文的四卡验收。较短样本是否能用4，是另一个范围的测试；本次结果不表示所有上下文长度下都不能使用4。对于保持现有完整上下文预算的全量训练，继续保留 actor/critic=2/2。32 agents、8并发序列、16384调度tokens的训练采样候选仍保留在独立试验脚本中，其完整训练步速度尚未实测，不改变前四卡正在运行的配置。
