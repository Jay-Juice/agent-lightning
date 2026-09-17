# V2 环境审计与本轮临时权重清理

新源码对应的环境审计于 2026-09-18 00:11:54 +08:00 完成，
124/124 个 pinned image 均通过 empty patch reward=0、reference reward=1，
`summary.ready=true`，`run.exit=0`。使用 CPU/Docker 两并发、cached-only，
CUDA_VISIBLE_DEVICES 为空，未下载新镜像，未修改旧审计 signature。

新审计路径：
`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/swe-full-python-envs-reliable-v2-20260918-01`

`check_full_python_ready.py` 随后检查全部 124 个 image / 6718 个任务分支，
输出 `FULL_PYTHON_READY`：train=6248、val=470、4 epochs/784 steps，退出码 0。
当前 V2 的 grader/sandbox/checked_agent/data_preparation 哈希与 signature 一致。
00:17:35 再次采集时，共享 all-task-branches.json 仅含5 images/164 tasks，
属于后续 readiness 检查增量覆盖时的局部快照；本次独立完整检查以保存的成功 stdout 为证。
正式启动应设置 `AGL_FULL_ENV_AUDIT` 指向上述新路径。

证据：`v2_env_audit_summary_20260918.json`、`v2_env_ready_check_20260918.txt`。
本地启动 wrapper：`run_v2_env_audit.sh`，已按 rsync dry-run/同步部署后执行。

主 agent 确认恢复验收通过后，按明确授权仅删除以下四个本轮可再生权重目录：

- `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/audit-checkpoint-roundtrip-v2-20260917-01/actor/global_step_1`
- `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/audit-checkpoint-roundtrip-v2-20260917-02/actor/global_step_1`
- `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/audit-checkpoint-roundtrip-v2-20260917-02/critic/global_step_1`
- `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/audit-checkpoint-roundtrip-v2-20260918-03/critic/global_step_1`

执行前复核 01/02/03 退出码分别为 1/1/0、精确路径非符号链接，
当前进程无该 run/权重路径引用；dry-run 审阅后执行。
全部四个目录已不存在，每个 role 目录保留 REMOVED 标记。
31 份 rank/completed/config/inputs/run.log/run.exit 证据的 SHA256 前后完全一致；
role 根目录和在线验收需要读取的 JSON 未删除。

清理后可用空间 530,474,938,368 bytes（约 494.04 GiB）；
观测空闲增加 196,358,709,248 bytes（约 182.87 GiB，包含共享盘背景变化）。
本轮不修改任何旧 baseline 或现有 PI 的恢复点。

脚本：`cleanup_roundtrip_artifacts.py`。
证据：`roundtrip_cleanup_scope_20260918.json`、`roundtrip_cleanup_dryrun_20260918.json`、
`roundtrip_cleanup_execution_20260918.jsonl`。
