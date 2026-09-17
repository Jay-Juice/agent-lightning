# Checkpoint 清理候选（只读盘点）

盘点时间：2026-09-17T23:37:49.748766 +08:00；进程引用复核：2026-09-17T23:39:03.401165 +08:00。此时为删除前盘点；随后已按主 agent 核准执行下方15项方案，见执行结果。

实际位置由 54 份 resolved-config 的 trainer.default_local_dir 和 logs/*/checkpoints 交叉发现。共 21 个 global_step 目录，占 835.19 GiB（du 分配占用）。

建议优先清理 A 组 14 个、207.56 GiB；B 组 3 个、262.07 GiB 为额外候选，由主 agent 决定。

按主 agent 最新空间预算，建议最终核准 A 组全部 + B 组失败 full step4：共 15 个目录，315.39 GiB。这样可保留 v6 step32、早期 4B SWE step1，以及全部 baseline/PI 对照点。机器可用空间按清理前 180 GiB 估算将升至约 495.39 GiB（实际以执行后 df 为准）。精确机器清单为 checkpoint_cleanup_candidates_20260917.json 的 recommended_300gib_plan 字段。

范围仅 /media/ubuntu/D1/zsj/agent-lightning-runtime/logs 下逐个列出的 global_step 目录。禁止删除 run 根目录、checkpoints 父目录、日志、数据、原始模型以及他人目录。所有盘点目录均无符号链接，resolved 路径与记录路径相同。

进程检查扫描了可读 /proc 的 cmdline、cwd、checkpoint fd，以及环境变量中的路径匹配（不保存环境值）。A/B 组未发现 checkpoint 或 run 路径引用；受权限限制无法证明其他用户隐藏进程完全无引用。PI 与 baseline run 在当前进程环境中仍有 run 级路径引用，不能把它解释为正在读取某个具体 checkpoint；这些目录均保留。

run.exit=0 仅表示包装脚本记载的退出状态，不等价于完整正式训练完成；未做所有 checkpoint 的加载验证。删除 smoke 的旧恢复点后，其历史配置不能原样重新 resume，但现有完成日志仍保留。

## A：优先候选（已完成的 0.6B 验收）

| 精确绝对路径 | GiB | 状态/价值 | 当前进程引用 |
|---|---:|---|---|
| `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-capo-padding-resume-20260912-02/checkpoints/global_step_3` | 16.73 | Qwen3-0.6B 的早期 smoke/resume 验收产物；run.exit=0，无当前进程引用 | 未发现 |
| `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-capo-padding-smoke-20260912-02/checkpoints/global_step_1` | 16.73 | Qwen3-0.6B 的早期 smoke/resume 验收产物；run.exit=0，无当前进程引用 | 未发现 |
| `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-capo-padding-smoke-20260912-02/checkpoints/global_step_2` | 16.73 | Qwen3-0.6B 的早期 smoke/resume 验收产物；run.exit=0，无当前进程引用 | 未发现 |
| `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-capo-resume-20260910-01/checkpoints/global_step_3` | 16.73 | Qwen3-0.6B 的早期 smoke/resume 验收产物；run.exit=0，无当前进程引用 | 未发现 |
| `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-capo-smoke-20260910-01/checkpoints/global_step_1` | 16.73 | Qwen3-0.6B 的早期 smoke/resume 验收产物；run.exit=0，无当前进程引用 | 未发现 |
| `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-capo-smoke-20260910-01/checkpoints/global_step_2` | 16.73 | Qwen3-0.6B 的早期 smoke/resume 验收产物；run.exit=0，无当前进程引用 | 未发现 |
| `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-ppo-four-smoke-20260910-01/checkpoints/global_step_1` | 16.73 | Qwen3-0.6B 的早期 smoke/resume 验收产物；run.exit=0，无当前进程引用 | 未发现 |
| `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-ppo-resume-20260910-01/checkpoints/global_step_3` | 16.73 | Qwen3-0.6B 的早期 smoke/resume 验收产物；run.exit=0，无当前进程引用 | 未发现 |
| `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-ppo-smoke-20260910-01/checkpoints/global_step_1` | 16.73 | Qwen3-0.6B 的早期 smoke/resume 验收产物；run.exit=0，无当前进程引用 | 未发现 |
| `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-ppo-smoke-20260910-01/checkpoints/global_step_2` | 16.73 | Qwen3-0.6B 的早期 smoke/resume 验收产物；run.exit=0，无当前进程引用 | 未发现 |
| `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-resume-20260910/checkpoints/global_step_3` | 10.06 | Qwen3-0.6B 的早期 smoke/resume 验收产物；run.exit=0，无当前进程引用 | 未发现 |
| `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-resume-20260910b/checkpoints/global_step_3` | 10.06 | Qwen3-0.6B 的早期 smoke/resume 验收产物；run.exit=0，无当前进程引用 | 未发现 |
| `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-smoke-20260910c/checkpoints/global_step_1` | 10.06 | Qwen3-0.6B 的早期 smoke/resume 验收产物；run.exit=0，无当前进程引用 | 未发现 |
| `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-smoke-20260910c/checkpoints/global_step_2` | 10.06 | Qwen3-0.6B 的早期 smoke/resume 验收产物；run.exit=0，无当前进程引用 | 未发现 |

## B：次优先候选（旧 4B 试验）

| 精确绝对路径 | GiB | 状态/价值 | 当前进程引用 |
|---|---:|---|---|
| `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-capo-swe-full-4b-20260912-01/checkpoints/global_step_4` | 107.82 | 旧 4B full 试验；run.exit=1，最后完整 step4，已被后续 v6/v7 替代；失败 run 不代表 checkpoint 损坏 | 未发现 |
| `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-capo-swe-pythonfull-v6-4b-20260913-03/checkpoints/global_step_32` | 91.39 | 旧 v6；末指标 step46，保存点32，已被 v7 替代；有旧版对照价值，次优先候选 | 未发现 |
| `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-swe-20260910/checkpoints/global_step_1` | 62.85 | 早期 4B SWE 单步试验；marker=1，run.exit=0，本盘点未找到 metrics.jsonl 完整指标；价值低但不能据此断言模型无效 | 未发现 |

## 必须/建议保留

| 精确绝对路径 | GiB | 状态/价值 | 当前进程引用 |
|---|---:|---|---|
| `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-capo-swe-pythonfull-v7-4b-gpu47-20260914-01/checkpoints/global_step_40` | 91.39 | step120 为明确保护恢复点；step40/80 暂留给历史策略/critic 对照及恢复链 | 未发现 |
| `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-capo-swe-pythonfull-v7-4b-gpu47-resume40-20260915-01/checkpoints/global_step_120` | 91.39 | step120 为明确保护恢复点；step40/80 暂留给历史策略/critic 对照及恢复链 | 63 个 PID 有 run 级环境路径引用（无明确 step 引用） |
| `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-capo-swe-pythonfull-v7-4b-gpu47-resume40-20260915-01/checkpoints/global_step_80` | 91.39 | step120 为明确保护恢复点；step40/80 暂留给历史策略/critic 对照及恢复链 | 63 个 PID 有 run 级环境路径引用（无明确 step 引用） |
| `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-capo-swe-pythonfull-v7-p2fixed-4b-gpu03-20260916-01/checkpoints/global_step_40` | 91.39 | 当前 PI 的全部恢复点必须保留 | 55 个 PID 有 run 级环境路径引用（无明确 step 引用） |

## 历史恢复依赖

以下是静态 resolved-config 引用，非活动进程引用。A 组依赖均是已完成 smoke/resume 验收；清理时一起清理这些旧验收权重，保留其日志即可。

- `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-capo-padding-smoke-20260912-02/checkpoints/global_step_2` ← `training-capo-padding-resume-20260912-02`
- `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-capo-smoke-20260910-01/checkpoints/global_step_2` ← `training-capo-resume-20260910-01`
- `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-ppo-smoke-20260910-01/checkpoints/global_step_2` ← `training-ppo-resume-20260910-01`
- `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-smoke-20260910c/checkpoints/global_step_2` ← `training-resume-20260910`, `training-resume-20260910b`
- `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-capo-swe-pythonfull-v7-4b-gpu47-20260914-01/checkpoints/global_step_40` ← `training-capo-swe-pythonfull-v7-4b-gpu47-resume40-20260915-01`

证据：checkpoint_inventory_20260917.json、checkpoint_config_inventory_20260917.json、checkpoint_run_metadata_20260917.json、checkpoint_process_refs_20260917.json。

执行前应再次核对路径位于明确的 runtime/logs 范围内、目录非符号链接、无新进程引用且不属于上述保护组。仅删除核准清单内的 step 目录；最初仅做只读盘点；随后收到主 agent 明确下发的15项删除授权后执行。

## 实际执行结果

时间：2026-09-17T23:48:10.009178 +08:00。经两阶段 rsync 部署、只读 dry-run、精确15项allowlist核对及当前进程引用复核，删除全部15个核准目录，退出码0。

- 删除的分配占用：338,645,102,592 bytes（315.3878 GiB）。
- 实际空闲增量：338,644,910,080 bytes（315.3876 GiB）；共享盘背景写入造成微小差异。
- 最终可用空间：495.5423 GiB。
- 15个目录全部不存在，15份 `global_step_N.REMOVED-20260917.json` 标记均存在。
- run 根目录、resolved-config、训练日志和数据全部保留；未修改现有 PI/baseline 文件。
- 其余6个checkpoint均在：v6 step32、baseline step40/80/120、当前PI step40、早期SWE step1。PI screen仍在。

进程扫描对同UID活动训练进程保持不可读即中止；只跳过已核实无文件句柄的僵尸、`(sd-pam)`/systemd 和 sshd/sshd 系统会话服务。未打印环境内容。

执行证据：checkpoint_cleanup_dryrun_20260917.json、checkpoint_cleanup_execution_20260917.jsonl、checkpoint_cleanup_verification_20260917.json。脚本：cleanup_authorized_checkpoints.py；精确15项：checkpoint_cleanup_allowlist_20260917.json。
