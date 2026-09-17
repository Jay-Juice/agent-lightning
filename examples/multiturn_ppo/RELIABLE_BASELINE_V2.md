# Reliable SWE baseline entrypoint

This P0 entrypoint is an implementation artifact, not a record of a started or
accepted training run. Existing runs are not changed by these files.

## Deployment mapping

- Windows source: `D:\ai project\RL学习\agent-lightning-main\agent-lightning-baseline-v2-20260917`
- WSL source: `/mnt/d/ai project/RL学习/agent-lightning-main/agent-lightning-baseline-v2-20260917`
- SSH host: `A800`, user `ubuntu`
- Deployment target: `/media/ubuntu/D1/zsj/agent-lightning-baseline-v2-20260917`
- Runtime: `/media/ubuntu/D1/zsj/agent-lightning-runtime`

Deploy via WSL rsync only after its dry run; no `--delete`. Source remains the
Windows worktree. Adding this entrypoint authorizes no process stop, checkpoint
deletion, or training launch.

## Recipe and launch boundaries

`run_reliable_full_python_ppo.sh` calls `run_full_python_ppo.sh` and retains its
Qwen3-4B-Instruct-2507 model, full v7 dataset, four epochs, task batch 32,
Actor/Critic call minibatch 32, microbatch 2, Actor/Critic LR 1e-6/1e-5,
reference KL .001, gamma=lambda=1, PPO epochs=1, default value head, and zero
critic warmup. The separate `run_reliable_ab_trial.sh A|B` entrypoint now runs
20-step minibatch-128 trials, differing only in default/zero head initialization.
`run_reliable_ab_pair.sh` gates their sequential execution on completed worker
recovery and environment audits. Warmup and KL changes remain unimplemented.

It inherits physical GPUs 4,5,6,7 from the full-data launcher. The existing
occupied-GPU check must pass before any launch. Default server port is 18521;
the default run tag starts `capo-swe-pythonfull-v2-reliable-4b-` and has a
timestamp. Existing run directories and occupied ports are rejected.

The reliability configuration, trainer/worker environment flag, and agent
environment flag must agree. Agent flags and deadline values are forwarded
explicitly with `local.env_map`, and `train.py` rejects mismatches before Ray
starts. Worker flag forwarding is implemented in the separate entrypoint
integration. Provenance records reliability settings, agent environment map,
runtime flags, and hashes of the reliability/proxy source files.

The interaction wall budget remains 3600 seconds; grading remains 600 seconds
per attempt. The 5400-second external hard deadline reserves two grading
attempts plus cleanup (minimum 3600 + 2 * (600 + 120) + 300 = 5340 seconds).
Only identified grading infrastructure exceptions may retry the same patch;
unknown grading outcomes, including unclassified exit 3/124/137, stop the batch
for review. Such outcomes are not recorded as ordinary reward zero.

Initial full validation remains enabled and repeats every 20 steps. Both Actor
and Critic retain two checkpoints. Initial free disk space must be at least
295 GiB: three measured 91.39 GiB checkpoint pairs (two retained plus one being
written), plus 20 GiB safety reserve, rounded up. A per-save 112 GiB reserve
protects each new write. The observed approximately 181 GiB therefore cannot
pass this launch preflight. No cleanup is automated. This estimate must be
revisited if model size or checkpoint contents change; concurrent jobs can
consume disk after preflight, so the trainer also checks at every save.

## Verification boundary

Thirteen CPU unit tests exercise disk peak requirements, flag/env-map consistency,
deadline reserve, retention/validation requirements, and legacy compatibility.
They can run with `PYTHONDONTWRITEBYTECODE=1 python tests/examples/test_reliable_launch.py`.
Passing them does not establish four-GPU training, checkpoint recovery, worker
flag propagation, or full-data grading correctness. Those require the parent's
combined integration checks and a separately authorized run with available
GPUs and sufficient disk space.


The combined CPU suite passed 156 tests plus 26 subtests on A800 on 2026-09-17.
Ruff and Bash syntax checks passed. This includes two-rank Gloo nonfinite
consensus, original CAPO optimizer methods, reliable/legacy agent loop parity,
logical-call proxy behavior, and checkpoint completion-marker/epoch restoration
with mocked parent workers. It does not constitute real FSDP checkpoint restore.

Agent SDK retries and gateway-internal retries are disabled for identified SWE
v2 logical calls. The trainer owns the one bounded whole-episode infrastructure
retry at unchanged weights. No real completion is removed by prompt deduplication.
Failed-attempt raw events and per-task outcome/reward manifests are retained.

Checkpoint auto-resume chooses the latest local actor/critic pair with a
reliability completion marker and dataloader state. A partially updated PPO
batch is never saved as complete. Worker optimizer count totals reset on worker
restart; they are process diagnostics, not persisted optimizer update counters.
