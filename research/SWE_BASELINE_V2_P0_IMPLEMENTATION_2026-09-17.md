# SWE baseline V2：第一步实施记录

本地代码已实施并部署到独立 V2 目录；A800 CPU 定向验收通过。没有停止、修改或重启原 baseline/PI，没有删除 checkpoint，没有启动新的训练。四卡 FSDP 保存恢复及在线训练验收尚未执行，不能把这份记录理解为新 baseline 已经训练成功。

## 参数来源核对

| 数值 | 已核对的出处 | 含义 |
|---|---|---|
| 1.2 | `../Agent-R1/CAPO/CAPO-main/arft/policy_losses.py:49` | 概率比超出 `[0.8,1.2]` 的统计阈值；PPO `clip_ratio_high=.2` 的上界也是 1.2。不是 KL 系数 |
| .95 | 同仓库 `verl/examples/gspo_trainer/run_qwen30b_gspo.sh:29`、SAPO 示例 `:94` | `gae_lam` 变量；两个脚本默认 `adv_estimator=grpo`，不能据此声称 CAPO PPO 使用 lambda=.95 |
| .95 | 同仓库 `examples/alfworld/run_gigpo.sh:20`；agent-R1 对应 GiGPO 脚本也有 | 状态相似度阈值，与 KL、gamma、lambda 不同 |
| .001 / .99 / 1 | CAPO `examples/alfworld/run_ppo.sh:55,88` 与继承的 `verl/verl/trainer/config/ppo_trainer.yaml:78` | PPO 的 reference KL 系数、gamma、lambda |

SWE 现配方保持 KL=.001、gamma=lambda=1；gamma=1 是此前已经声明的 SWE 长时域适配。第一步不改学习率、minibatch、GAE、value head、whitening、warmup 或奖励定义；`.005` 只是后续候选消融，未实施。

## 已实现

- SWE v2 outcome 合约：任务终止、基础设施中断、评分未决分开；终止原因、唯一奖励、attempt、policy version、调用顺序和 token 完整性均检查。格式终止依然依据实际补丁测试给分。
- logical call ID 贯穿 agent、proxy、raw/trimmed events。只移除确认无输出的 HTTP 拒绝记录；保留相同 prompt 的不同真实生成。可靠模式关闭 SDK 和 gateway 内部隐式重试，训练端只允许一次同任务、同 policy 的基础设施重试。核对任务输入而非每次重建的随机 data_id，恢复原批次位置。
- Linux 独立 agent 进程绝对 deadline 约束整个模型请求及命令；给导出、评分、清理留出预算。明确评分传输/daemon 异常只重评同一 patch，最多一次，并分别保留两个评分目录。未知评分状态不生成 PPO reward，不整题重采样。
- 新增排除 dummy 的 value MSE、EV、常数基线、token/call advantage、sampled old/ref/rollout k3 分布及行为统计；分批验证正确合并计数、均值、方差，不伪造全局分位数。
- 完整验证检查格式终止>5%、主动提交<80%、length 截断题占比>10%、平均输出>初始2倍。越界保存完整更新状态并停止；初始验证不合格则直接停止。
- 沿用 CAPO 原 optimizer/clipping 方法，以 hook 记录实际 step、尝试和跳过次数；非有限梯度在所有 rank 的 optimizer step 前一致停止。进程累计计数在重启后清零，不冒充持久化 optimizer state.step。
- 部分 Actor/Critic 更新失败禁止保存为完整 step；完成 checkpoint 写入后增加完成标记、epoch、初始输出长度基准。auto resume 跳过未完成的新写入，寻找最近完整 Actor/Critic/dataloader 状态。
- 逐题 outcome/reward 清单和失败 attempt 原始事件落盘；新增源文件 hash、可靠配置及显式 env_map provenance。
- 新入口 `examples/multiturn_ppo/run_reliable_full_python_ppo.sh`：默认关闭旧路径之外的可靠性开关，首次全验证、每20步验证、保留2份 checkpoint。新 tag、端口18521。任务预算仍32轮/4096输出/65536上下文；interaction3600秒、单次评分600秒、外部硬截止5400秒。

## 评分分类的边界

`pytest_exit` 不能单独证明基础设施故障。当前可靠入口对未分类的异常退出保守停止，要求复查；不会把它们静默记零，也不会重采样到成功。历史9例已完成固定候选/参考对照，结果如下；不把几个旧 patch 的证据推广成所有同退出码都属于模型错误。

复评分现已完成：9对/7个独立task、18次评分；9个reference均exit0/reward1，9个candidate均复现原exit/reward0。6个exit137均有与候选容器标签匹配的Docker OOM事件；2个exit124按原600秒测试预算复现超时（总耗时602.10/604.41秒）；1个exit3复现pytest internal error。最后bottle候选在489.73秒自然OOM，未被人为提前结束。9/9参考通过支持这些是候选工作区在相同评分约束下的失败，不能区分“引入新bug”与“没有修复原bug”，也不证明所有训练问题都来自模型能力。

2026-09-17 21:41:29确认审计screen已退出、只读OOM监听exit0、残余审计容器0；原baseline和PI screen仍在。逐对exit/reward/duration、源码/输入hash和六条OOM证据保存在新工作树 `research/baseline_grading_recheck_results_20260917.json`，解释见 `research/BASELINE_GRADING_RECHECK_2026-09-17.md`。

## 验收

2026-09-17 A800 定向 CPU 回归：156 passed，26 subtests passed；仅3条依赖弃用警告。Ruff 与新入口 `bash -n` 通过。包含真实 AdamW 数值一致性、原 CAPO Actor/Critic optimizer 方法、两 rank Gloo 非有限一致停止、默认/可靠模式 agent 协议、proxy、padding、GAE、全量数据计数、恢复控制逻辑。

checkpoint 控制测试以 mock 父级 worker 验证完成标记和恢复选择，**不等于真实4B FSDP权重/optimizer/RNG四卡恢复验收**。未运行 critic 诊断、minibatch128、零初始化、warmup、KL 加强或任何新在线训练。

## 磁盘与后续条件

当前约181 GiB可用；完整 Actor+Critic checkpoint 约91.39 GiB。保留两份并写第三份的预检为 `ceil(3*91.39+20)=295 GiB`，每次保存还要求112 GiB可用。当前会拒绝新训练，未自动删除旧实验。并行旧任务仍可能消耗空间，预检不能替代每次写入前检查。

后续需要先解决可用GPU/磁盘安排，再做四卡短程保存恢复验收；通过后才进入既定 P1/P2 实验。现有 P0 改变了基础设施中断/重试协议，旧数据只能作历史参照；未来 PI 必须使用同一套 P0 协议才能严格对照。

## 路径和已执行命令

- Windows 源：`D:\ai project\RL学习\agent-lightning-main\agent-lightning-baseline-v2-20260917`
- WSL 源：`/mnt/d/ai project/RL学习/agent-lightning-main/agent-lightning-baseline-v2-20260917`
- SSH：`A800`，用户 `ubuntu`
- 部署：`/media/ubuntu/D1/zsj/agent-lightning-baseline-v2-20260917`
- 分支：`experiment/swe-baseline-v2-reliability-20260917`，起点 `78ed866`，可靠性代码提交 `8441823`（本地提交，未推送）

每次部署先运行下列 `-anvz`，检查文件列表后执行相同参数的 `-az`；未使用 `--delete`：

```powershell
wsl rsync -anvz -e '/mnt/c/Windows/System32/OpenSSH/ssh.exe -l ubuntu' --exclude='.git' --exclude='__pycache__/' --exclude='.pytest_cache/' --exclude='.ruff_cache/' --exclude='data/' --exclude='datasets/' --exclude='checkpoints/' --exclude='logs/' --exclude='outputs/' --exclude='wandb/' --exclude='*.pt' --exclude='*.safetensors' '/mnt/d/ai project/RL学习/agent-lightning-main/agent-lightning-baseline-v2-20260917/' 'A800:/media/ubuntu/D1/zsj/agent-lightning-baseline-v2-20260917/'
```

通过 Windows OpenSSH 在上述部署目录运行以下验收；环境先 `source /media/ubuntu/D1/zsj/agent-lightning-runtime/admin/activate-agent-lightning-d1.sh`：

```bash
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -m pytest -p no:cacheprovider tests/verl/test_episode_contract.py tests/verl/test_reliability_control.py tests/verl/test_reliability_metrics.py tests/verl/test_reliability_optimizer.py tests/verl/test_reliability_checkpoint.py tests/verl/test_capo_padding.py tests/verl/test_multi_turn_ppo.py tests/verl/test_full_dataset.py tests/examples/test_reliable_launch.py tests/examples/test_swe_reliability.py tests/examples/test_smith_docker_loop.py tests/examples/test_full_python.py tests/server --basetemp=/tmp/agl-baseline-v2-p0-tests-20260917d -q
bash -n examples/multiturn_ppo/run_reliable_full_python_ppo.sh
```

最后增补的失败 attempt 原始事件落盘已包含在上述156项完整回归中；CAPO vendor hash校验通过（CAPO_VENDOR_UNCHANGED）。
