# SWE multi-turn PPO experiment index

Last updated: 2026-09-15 (Asia/Shanghai).

This file is the repository index for the SWE multi-turn PPO work. Runtime logs,
checkpoints, datasets, model weights, environment images, and machine-specific
handoff notes stay outside Git.

## Branches and immutable run tags

| Ref | Purpose | Status |
| --- | --- | --- |
| `main` | Verified SWE baseline implementation and reusable stability fixes | Canonical development base |
| `experiment/swe-speed-20260913` | Exact baseline development branch used for the active GPU 4-7 run | Retained for run provenance |
| `experiment/swe-sandbox-p1-20260914` | P1 sandbox-state privileged critic experiment | Experimental; do not merge on current results |
| `codex/multi-turn-ppo` | First complete-episode PPO implementation | Historical ancestor |
| `swe-baseline-v1-20260915` | Exact baseline code used to resume from global step 40 | Immutable tag |
| `swe-p1-v1-20260914` | Exact P1 code used by the active GPU 0-3 run | Immutable tag |

New algorithms should start from `main` on a new `experiment/<topic>-<date>`
branch. A run tag identifies the exact source revision; moving an existing run
tag is not allowed.

## Shared comparison configuration

- Model: Qwen3-4B-Instruct-2507.
- Dataset: 6,248 training tasks and 470 validation tasks from the retained full
  Python SWE-smith set.
- Four GPUs per run; 32 rollout workers.
- Train batch 32, actor/critic microbatch 2.
- Actor LR `1e-6`, critic LR `1e-5`, KL coefficient `0.001`, no warmup.
- `gamma=1`, `lambda=1`, terminal binary success reward.
- Maximum 32 turns, 4,096 response tokens, 65,536 context tokens, and 32,000
  observation characters.
- Full validation every 20 steps and checkpoint every 40 steps.

The baseline timeout fix preserves the 3,600-second agent interaction budget.
At that boundary the agent proceeds to patch export, grading, and reward
submission. The controller hard timeout is 5,400 seconds so cleanup can finish.
A post-training validation exception saves a recovery checkpoint before it is
re-raised.

## Results snapshot

Results below are from 2026-09-15 around 14:57 CST.

| Run | Validation step 0 | Validation step 20 | Validation step 40 | Mean train reward, steps 1-27 | Mean step time, steps 1-27 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 70/470 (14.89%) | 73/470 (15.53%) | 72/470 (15.32%) | 24.19% | 1,625 s |
| P1 | 70/470 (14.89%) | 48/470 (10.21%) | pending | 17.01% | 2,071 s |

Both runs have complete reward coverage in the recorded metrics and finite PPO
statistics. P1 is operationally healthy through step 27, but its step-20
validation and training reward are below the matched baseline. Its active v1 run
also predates the graceful timeout fix and still has a 3,600-second controller
hard timeout.

P1 should remain experimental until it:

1. completes a second full validation with all 470 rewards;
2. is rebased or ported onto the baseline timeout and recovery fixes;
3. passes the baseline parity and privileged-state tests; and
4. demonstrates validation performance that justifies its additional compute.

## Repository policy

- Commit source, launchers, tests, compact audit scripts, and concise result
  summaries.
- Do not commit checkpoints, model weights, raw trajectories, Docker state,
  datasets, secrets, host-specific environment files, or full runtime logs.
- Record the source commit/tag, resolved configuration, dataset fingerprint,
  model path identity, GPU assignment, and runtime log location in each local
  run handoff.
- Preserve failed experiments on their tagged commit rather than rewriting their
  branch history.

Detailed baseline implementation notes are in
[`SWE_FULL_GPU47_2026-09-14.md`](SWE_FULL_GPU47_2026-09-14.md) and
[`SWE_SPEED_TRIAL_2026-09-13.md`](SWE_SPEED_TRIAL_2026-09-13.md). The algorithm
and reference audits are in [`CAPO_PPO.md`](CAPO_PPO.md) and
[`PPO_REFERENCE_AUDIT_2026-09-10.md`](PPO_REFERENCE_AUDIT_2026-09-10.md).
