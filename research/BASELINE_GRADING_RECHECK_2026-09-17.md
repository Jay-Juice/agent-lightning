# Frozen-Actor baseline grading recheck — 2026-09-17

This audit regrades only the nine saved candidates whose original baseline
validation subprocess exits were 3, 124, or 137 (steps 0 and 120). It does not
call an Actor/model, edit any original log, change the scoring protocol, or
restart training. Reference controls are visible only to the isolated grader.

## Source and deployment

- Authoritative Windows worktree: `D:/ai project/RL学习/agent-lightning-main/agent-lightning-baseline-v2-20260917`
- WSL source: `/mnt/d/ai project/RL学习/agent-lightning-main/agent-lightning-baseline-v2-20260917`
- SSH host/user: `ubuntu@A800`
- Deployment: `/media/ubuntu/D1/zsj/agent-lightning-baseline-v2-20260917/research`
- Frozen grader: `/media/ubuntu/D1/zsj/agent-lightning-speedtrial-20260913/examples/multiturn_ppo/full_python_agent.py`
- Grader source commit: `5fa18418331e0910ea2cd19ce5fb449282810ce4`
- Grader SHA256 (Windows and A800 matched): `0accf7e13fb922caf6a023de7afd66cf7e17d014d0bb107525a000c912b87556`
- Audit script SHA256 at launch: `fa2f190eb8f4c9965056dc24a0cf53c6627069bd13cc1f997e48f13f91d519c3`
- Case manifest SHA256: `2f49fc963170157efe00abccb2749be4bc036829075650efa3f01e9ced4b687c`

Deployment used `wsl rsync -anz --stats` first, then `-az --stats`, with
`-e '/mnt/c/Windows/System32/OpenSSH/ssh.exe -l ubuntu'`. Only
`research/regrade_baseline_exceptions.py` and
`research/baseline_regrade_cases_20260917.json` were supplied as source files.
The reviewed dry run reported two new regular files, 9,130 bytes, no deletion.
No production source or `.git` was deployed by this audit.

## Execution

Screen: `agl-grading-recheck-20260917-01`.

```bash
source /media/ubuntu/D1/zsj/agent-lightning-runtime/admin/activate-agent-lightning-d1.sh
CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=void PYTHONDONTWRITEBYTECODE=1 \
python -u /media/ubuntu/D1/zsj/agent-lightning-baseline-v2-20260917/research/regrade_baseline_exceptions.py \
  --baseline-source /media/ubuntu/D1/zsj/agent-lightning-speedtrial-20260913 \
  --logs-root /media/ubuntu/D1/zsj/agent-lightning-runtime/logs \
  --cases /media/ubuntu/D1/zsj/agent-lightning-baseline-v2-20260917/research/baseline_regrade_cases_20260917.json \
  --output /media/ubuntu/D1/zsj/agent-lightning-runtime/logs/audit-baseline-grading-recheck-20260917-01
```

Actual detached invocation redirects output to:
`/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/launch-audit-baseline-grading-recheck-20260917-01.log`.
The output directory must not already exist. A new tag is required for another
run; do not overwrite this run.

Single concurrency, original 600-second grading budget, reference first then
candidate. Stop after two failed references; skip a candidate when its reference
fails, since that pair is not interpretable. Resource guards before each grade:
64 GiB available RAM, 100 GiB free disk, load below 90% of CPU count. Initial
host: 64 CPUs, load about 17, available RAM 498 GiB, free D1 181.45 GiB.
The ordinary frozen grader containers retain their original 4 GiB/2 CPU limits.

`summary.json` records source/input hashes, resources, each leg's exit/status and
score; `progress.jsonl` records starts and finishes. Only compact results will be
copied back for review, never weights or complete rollout trajectories.

## Interpretation

A reference pass plus candidate failure supports a candidate/workspace-specific
failure under the unchanged grading constraints. It does not establish that the
candidate introduced the bug rather than left the original bug unfixed. A
reference failure invalidates that pair for causal attribution.

Exit 137 alone is not sufficient to claim OOM. Docker `oom` events with matching
`agl.run_id` and `agl.instance_id` supply stronger evidence. Grade-file presence
only establishes recorded execution, not that a run completed successfully.

## Completed result

Finished 2026-09-17T21:39:06.588542+08:00. Nine candidate/reference pairs (seven distinct tasks) completed; all references scored 1 and all candidates scored 0. No reference failure or resource guard interruption occurred.

| Case | Original step | Original exit | Reference exit / seconds | Candidate exit / seconds |
|---|---:|---:|---:|---:|
| 0 | 0 | 3 | 0 / 1.83 | 3 / 1.65 |
| 1 | 0 | 137 | 0 / 1.92 | 137 / 27.82 |
| 2 | 0 | 137 | 0 / 6.64 | 137 / 101.80 |
| 3 | 0 | 124 | 0 / 7.78 | 124 / 602.10 |
| 4 | 120 | 124 | 0 / 9.69 | 124 / 604.41 |
| 5 | 120 | 137 | 0 / 1.87 | 137 / 32.68 |
| 6 | 120 | 137 | 0 / 6.85 | 137 / 135.91 |
| 7 | 120 | 137 | 0 / 2.85 | 137 / 102.88 |
| 8 | 120 | 137 | 0 / 2.20 | 137 / 489.73 |

Every original exit code reproduced. All six 137 exits have Docker OOM events with the exact candidate `agl.run_id` and task ID. Two candidates reached the unchanged 600-second test timeout, while the corresponding references passed in under ten seconds. The remaining candidate reproduced pytest internal error 3 with a passing reference.

See `baseline_grading_recheck_results_20260917.json` for per-pair scores, timings, input hashes and matched OOM evidence, and `baseline_grading_recheck_oom_evidence.json` for the six captured events.

The audit screen ended, the read-only event watcher exited 0, and no audit containers remained. Both existing baseline and PI screen sessions were still present at verification. No original rewards were rewritten.

These results support candidate/workspace-specific failures under the original constraints. They do not establish whether every fault was newly introduced or merely left unfixed by its candidate. They do not justify broadly labeling exit 3/124/137 as infrastructure failures or replacing this audit with a new reward protocol.
