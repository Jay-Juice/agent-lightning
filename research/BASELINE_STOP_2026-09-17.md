# Authorized baseline stop

User authorization relayed by parent: stop baseline only; preserve PI.

At 2026-09-17 22:02–22:04 +08:00, stopped
`capo-swe-pythonfull-v7-4b-gpu47-resume40-20260915-01`.
The source was deployed using WSL rsync, after a reviewed dry-run limited to
`stop_authorized_baseline.py` and `baseline_stop_scope_20260917.json`.

The stop script verified the screen/session anchor by PID, start ticks, SID,
cwd, and exact screen name. It enumerated current descendants, including the
rollout worker's separate session, pinned identities with Linux pidfds,
and sent TERM to 54 processes. No KILL was required. It performed no Docker
actions and deleted no checkpoints or files.

Verification:

- Baseline screen PID 2471520 and all baseline GPU workers exited.
- A full readable `/proc` scan found no remaining process with baseline cwd.
- GPU4–7 each reported 14 MiB, 0% utilization, and no compute process.
- PI screen PID 3867191 and all eight GPU0–3 workers remained running;
  observed GPU utilization was 59–95%.
- Latest complete baseline metric: step 126, training reward 4/32 (0.125).
- Last validation: step 120, 60/470 (12.77%).
- Recovery checkpoint remains `global_step_120`; marker is 120.
  Previously checked actor/critic rank-file presence, sizes and ZIP index;
  this is not a full restoration/checksum test.
- Updates 121–126 were not saved; step 127 was incomplete at stopping.

Evidence files: `baseline_stop_scope_20260917.json`,
`baseline_stop_plan_20260917.json`, `baseline_stop_execution_20260917.jsonl`,
`baseline_stop_verification_20260917.txt`, and `baseline_stop_final_20260917.json`.

Remote execution:

```sh
python3 /media/ubuntu/D1/zsj/agent-lightning-baseline-v2-20260917/research/stop_authorized_baseline.py --scope /media/ubuntu/D1/zsj/agent-lightning-baseline-v2-20260917/research/baseline_stop_scope_20260917.json --execute
```
