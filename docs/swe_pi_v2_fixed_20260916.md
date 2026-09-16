# Matched SWE PI V2 repair — 2026-09-16

The previous `capo-swe-pythonfull-v7-p2-4b-gpu03-20260916-03` launcher bypassed the checked baseline wrapper. Its Actor ran with default 8 turns / 768 response tokens / 12288 context / 4000 observation characters and without the checked editor controls. Its results are not a matched PI comparison.

## Changes

- New `examples/multiturn_ppo/run_full_python_p2.sh` inherits the complete full-Python checked baseline chain. The pilot entry delegates to it with an explicit eight-step limit. Full entry defaults to 784 steps from original model weights.
- Before starting services, `matched_recipe.py` compares resolved PPO/model/sampling configuration, all recorded baseline Agent environment variables, and canonical train/validation row hashes against the recovered baseline run. Only PI settings, runtime identities, resume/step fields and checkpoint retention can differ. The full entry explicitly sets 784 rather than inheriting nested two-step defaults.
- Port the baseline's 3600-second Agent wall budget, 5400-second rollout timeout and checkpoint-on-validation-error recovery behavior. PPO, GAE, loss, sampling, model and data settings are unchanged.
- Serialize frozen prepared S0-to-current-state cumulative semantic diffs. Preserve compact manifests, pack short lines together, give each file a chunk per round, split long lines explicitly, and reserve an omission notice. Count changed lines only when every fragment is present. Distinguish touched hunks from completely represented hunks; aggregated hunk coverage uses complete hunks. Zero-change rows are excluded from coverage denominators.
- Preserve the Actor's entire legal context domain. Within the reserved PI safety margin, fall back to the unmodified Actor prompt instead of rejecting the rollout.
- Add read-only value diagnostics for semantic/no-semantic, truncated/full PI, excluding padding and response holes. Empty groups are explicitly unavailable (NaN diagnostics, not NaN training tensors).
- Freeze the exact prepared commit's blob map, using NUL-delimited paths and rejecting ambiguous path components. Read only frozen blob IDs.

## Matched recipe

Qwen3-4B-Instruct-2507; physical GPUs 0–3; 6248 training / 470 validation tasks; four epochs, 196 batches/epoch including tail, 784 updates; batch/minibatch 32; microbatch 2; 32 local agents; 32 turns, 4096 output tokens, 65536 context, 32000 observation characters. Actor/Critic LR 1e-6/1e-5, gamma=lambda=1, KL=0.001, no warmup. Validate every 20, save every 40. PI budget 4096, margin 32. New PI retains two checkpoints as explicitly approved for disk capacity; the already-running baseline remains configured for three until its old checkpoints are separately cleaned.

Canonical JSON row hashes:
- train: `21e9172cb45b007cdf75c81c4918e0fec3c53a05a85bf9fee4ff2f7d5aca1e74`
- validation: `15a1effb0b71a9cc30bb692b0f27909bf78d5caa5c7ac31f5a985da1c6e2ad79`

## Verification

- 120 targeted CPU regression tests passed; only an upstream Ray deprecation warning. Includes all 33 context boundary cases and Actor input parity with PI disabled.
- Ruff check passed for the changed serializer, recipe verifier, audit and diagnostic modules/tests.
- Three real prepared tasks × three controlled edits (large-file single line, several files, new helper): all nine passed using the actual 4B tokenizer and 4096-token PI budget. V2 retained every controlled changed line; V1 dropped the large-file body. Frozen S0 blobs matched workspace contents; restoring each scenario produced zero filesystem delta. These are representation checks, not measured PPO gains.
- All 124 cached Docker images passed empty-patch=0 / reference-patch=1 grading controls. All 6718 task branches passed full-data readiness checks.
- Complete launch-chain config-only run passed the baseline recipe comparison and recorded 784 updates. No GPU/model service is started by `AGL_CONFIG_ONLY=1`.

## Deployment and launch

Authoritative Windows worktree: `agent-lightning-pi-v2-20260915`. Isolated deployment: `/media/ubuntu/D1/zsj/agent-lightning-pi-v2-fixed-20260916`. WSL rsync dry-run then sync, excluding git/cache/model/log/checkpoint/output state, without deletion. No remote source edits.

New tag: `capo-swe-pythonfull-v7-p2fixed-4b-gpu03-20260916-01`.
Screen: `agl-full-v7-p2fixed-4b-gpu03-20260916-01`.
Runtime logs: `/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-<tag>`.
Launch command from the isolated deployment:

```bash
AGL_GPUS=0,1,2,3 AGL_TRAIN_PORT=18701 AGL_TRAIN_TAG=capo-swe-pythonfull-v7-p2fixed-4b-gpu03-20260916-01 bash examples/multiturn_ppo/run_full_python_p2.sh
```

The old PI's verified process tree (screen 2564290, 48 processes) was stopped by explicit user request. Its `global_step_40` checkpoint (91.39 GiB) was deleted with explicit permission; all logs and a deletion manifest remain. Baseline screen 2471520 on GPUs 4–7 was preserved.

Audit artifacts under runtime/logs:
- `swe-full-python-envs-p2fixed-20260916-01`
- `pi-v2fixed-semantic-audit-20260916-01`
- `training-capo-swe-p2fixed-preflight-20260916-01/recipe-verification.json`

Disk caution: a complete checkpoint is about 91.39 GiB. Editing a launcher cannot change the running baseline's in-memory retention setting. Its directory needs explicit scoped cleanup of the oldest complete checkpoint once more than two have accumulated. The old baseline resume source must not be removed merely as part of the authorized old-PI deletion. Initial startup is not evidence that the full 784-step run has converged or that PI improves reward.
