#!/usr/bin/env bash
set -Eeuo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME=/media/ubuntu/D1/zsj/agent-lightning-runtime
DATA="${AGL_FULL_DATA:-$RUNTIME/data/swe-smith-training/python-full-v6}"
source "$RUNTIME/admin/activate-agent-lightning-d1.sh"
AUDIT="${AGL_FULL_ENV_AUDIT:-$RUNTIME/logs/swe-full-python-envs-20260913-08}"
CUDA_VISIBLE_DEVICES= python "$TOOLS/check_full_python_ready.py" --data "$DATA" --audit "$AUDIT"
SCHEDULE=$(CUDA_VISIBLE_DEVICES= python - "$DATA" <<'PY'
import json, sys
from pathlib import Path
manifest = json.loads((Path(sys.argv[1]) / 'manifest.json').read_text())
batches = (manifest['counts']['train'] + 31) // 32
assert batches * 4 == manifest['steps_for_four_complete_epochs']
print(batches, batches * 4)
PY
)
read -r EPOCH_STEPS TOTAL_STEPS <<< "$SCHEDULE"
export AGL_GPUS=0,1,2,3
export AGL_CAPO_PROGRESS=1
export AGL_MAX_LOCAL_AGENTS="${AGL_MAX_LOCAL_AGENTS:-16}"
export AGL_MIN_FREE_GIB="${AGL_MIN_FREE_GIB:-360}"
export AGL_TRAIN_TAG="${AGL_TRAIN_TAG:-capo-swe-pythonfull-4b-$(date +%Y%m%d-%H%M%S)}"
# Keep the checked PPO objective, model, precision and per-task budgets.
# Every retained task is used each epoch: ceil(training rows / 32) * 4.
# Approved exclusions and original row hashes are recorded in the manifest.
# Intermediate checkpoints retain optimizer/RNG and all FSDP weights. Avoid
# writing a duplicate HF export at every checkpoint; export the final actor once.
bash "$TOOLS/run_checked_swe_ppo.sh" \
  --train-file "$DATA/train.jsonl" --val-file "$DATA/val.jsonl" \
  trainer.total_epochs=4 trainer.total_training_steps=null \
  data.train_batch_size=32 data.val_batch_size=32 \
  agentlightning.full_dataset=true agentlightning.audit_every_n_steps=32 \
  agentlightning.local.agent_class=full_python_agent.FullPythonAgent \
  actor_rollout_ref.actor.fsdp_config.reshard_after_forward=false \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  critic.ppo_micro_batch_size_per_gpu=2 critic.forward_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.max_num_seqs=4 \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]' \
  trainer.save_freq=32 trainer.test_freq="$EPOCH_STEPS" \
  trainer.max_actor_ckpt_to_keep=1 trainer.max_critic_ckpt_to_keep=1 \
  trainer.val_before_train=true trainer.resume_mode=disable "$@"
RUN="$RUNTIME/logs/training-$AGL_TRAIN_TAG"
CUDA_VISIBLE_DEVICES= python -m verl.model_merger merge --backend fsdp \
  --local_dir "$RUN/checkpoints/global_step_$TOTAL_STEPS/actor" --target_dir "$RUN/final-actor" \
  >"$RUN/final-export.log" 2>&1
