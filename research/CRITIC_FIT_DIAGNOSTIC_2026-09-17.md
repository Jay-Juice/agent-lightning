# P1 固定轨迹 Critic 诊断

## 当前状态

2026-09-17：按用户明确指示停止旧 baseline，保留 GPU 0-3 上的 PI。
旧 baseline 最后完成 step126，完整恢复点为 step120，121-126 未保存；未删除 checkpoint 或日志。
停止进程的身份核验及结果见工作树 `research/BASELINE_STOP_2026-09-17.md`。

22:04 已在 GPU 4-7 启动独立 Critic 诊断，两臂依次执行 default、zero。
真实 GPU 结果尚在生成；default 已完成初始评估和第 1 次真实更新：
fit token MSE=7.2573（训练均值常数基线 .07651），check MSE=7.4598（训练均值常数基线 .17883）。
第一次更新 decoder-layer 裁剪后梯度范数=.8684，36 个 decoder 参数梯度张量可见，head 输入/输出反向梯度非零。
目前确认默认初始化失准且梯度链路接通；本文件不代表拟合通过，也不代表可启动全量在线训练。

## 固定条件

- 原始 Qwen3-4B-Instruct-2507 backbone；不加载 Actor，不生成新 rollout，不保存模型 checkpoint。
- 使用旧 baseline 原始首批完整轨迹：32 个任务、7 个成功、380 calls、89,657 action tokens。
- 按完整 episode 分层划分，fit=24 任务/5 成功/270 calls，check=8 任务/2 成功/110 calls。
- 两臂相同 seed=1729、相同数据顺序；只改 value head 默认初始化或零初始化。
- 复用安装环境 veRL CriticWorker、vendored CAPO update/value loss、FSDP mixed precision 和 strict padding。
- LR=1e-5，weight_decay=.01，value clip=.5，call minibatch=32，microbatch/GPU=2，PPO epoch=1，warmup=0。
- 每臂固定数据训练 4 passes；每 pass 9 次真实 optimizer 更新，预计各 36 次。
- 每个 pass 开始前用本臂 Critic 重新计算 old_values；returns 始终固定为原 gamma=lambda=1 二值终局回报。
- 指标排除重复 padding，记录 token/call MSE、EV、常数预测基线、成功/失败及首末 action token。
- 使用 head backward hook 和 FSDP decoder-layer 参数梯度检查；后者不包括 embedding/finalnorm，不能误称完整 backbone 范数。
- 零 head 首次反传的 backbone 梯度为 0 属数学预期；head 更新后应恢复梯度传播。

这是固定数据接线与可拟合性诊断，随机 call 排序是两臂共有的诊断 schedule，不是在线 PPO 的新训练规则。
检查集只有 8 个任务；结果不能直接解释为 SWE 修复率变化，也不把诊断权重注入后续 baseline。

## 代码与验证

独立分支 `experiment/swe-baseline-v2-reliability-20260917`，本地源码在
`agent-lightning-baseline-v2-20260917/research/`：

- `critic_fit_data.py`：固定数据、episode split、padding 和统计。
- `critic_fit_diagnostic.py`：原 worker 两臂诊断及梯度证据。
- `run_critic_fit_diagnostic.sh`：四卡空闲预检、两臂顺序运行、每臂 2 小时硬上限。
- `test_critic_fit_diagnostic.py`：10 项 CPU 测试，在 A800 原环境全部通过。

Ruff check、bash -n 通过；安装环境配置 dataclass 转换成功。
Windows 为源码权威；每次 rsync 均先 dry-run，不使用 --delete。
旧 baseline、PI 的源目录未被此次诊断修改。

## 运行位置与命令

代码：`/media/ubuntu/D1/zsj/agent-lightning-baseline-v2-20260917`

输出：`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/audit-critic-fit-v2-20260917-02`

screen：`agl-critic-fit-v2-gpu47-20260917-02`

`...-01` 仅 CPU prepare，不曾启动 GPU；`...-02` 记录最终启动代码及输入 hash。

同步命令形式（逐次文件清单限定为上述 research 文件）：

```powershell
wsl rsync -anvz -e '/mnt/c/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes -l ubuntu' '<WSL本地文件>' 'A800:/media/ubuntu/D1/zsj/agent-lightning-baseline-v2-20260917/research/'
# 审核 dry-run 后，相同文件清单将 -anvz 改为 -avz。
```

实际 prepare 与启动：

```sh
source /media/ubuntu/D1/zsj/agent-lightning-runtime/admin/activate-agent-lightning-d1.sh
cd /media/ubuntu/D1/zsj/agent-lightning-baseline-v2-20260917
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python research/critic_fit_diagnostic.py prepare \
  --snapshot /media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-capo-swe-pythonfull-v7-4b-gpu47-20260914-01/ppo-audit/step-000001.pt \
  --config /media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-capo-swe-pythonfull-v7-4b-gpu47-20260914-01/resolved-config.json \
  --output /media/ubuntu/D1/zsj/agent-lightning-runtime/logs/audit-critic-fit-v2-20260917-02 --passes 4
screen -dmS agl-critic-fit-v2-gpu47-20260917-02 bash \
  /media/ubuntu/D1/zsj/agent-lightning-baseline-v2-20260917/research/run_critic_fit_diagnostic.sh \
  /media/ubuntu/D1/zsj/agent-lightning-runtime/logs/audit-critic-fit-v2-20260917-02
```

## 后续判据

核对两臂 backbone probe hash、head_before hash 相同，head_after 仅零初始化臂改变；probe 不是全权重 checksum。
核对两臂相同实际 optimizer 更新数、无非有限梯度和无跳步，再比较 fit/check MSE 与常数基线。
固定数据拟合通过之后，仍需 P0 四卡完整 Actor/Critic checkpoint 保存恢复验收及磁盘安排，才能开始有限在线候选。
当前没有启动新的全量 baseline，也没有将本轮代码 push 到 GitHub。
