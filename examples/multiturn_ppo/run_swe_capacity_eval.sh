#!/usr/bin/env bash
set -Eeuo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${AGL_TRAIN_MODEL:?Set AGL_TRAIN_MODEL to an existing local model}"
export AGL_GPUS="${AGL_GPUS:-0,1,2,3}"
export SMITH_MAX_TURNS="${SMITH_MAX_TURNS:-24}"
export SMITH_MAX_TOKENS="${SMITH_MAX_TOKENS:-4096}"
export SMITH_CONTEXT="${SMITH_CONTEXT:-32768}"
DATA=/media/ubuntu/D1/zsj/agent-lightning-runtime/data/swe-smith-training/exceptiongroup
# The original training pool is evaluated without any parameter update.
# Qwen3-8B has a 2.32 GiB fp32 embedding tensor; a 2 GiB transfer bucket cannot hold it.
bash "$TOOLS/run_capo_ppo.sh" --steps 1 \
  --train-file "$DATA/val.jsonl" --val-file "$DATA/train.jsonl" \
  data.train_batch_size=4 data.val_batch_size=32 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 critic.ppo_mini_batch_size=4 \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
  trainer.val_before_train=true trainer.val_only=true trainer.resume_mode=disable trainer.save_freq=-1 \
  agentlightning.rollout_timeout_seconds=3600 "$@"
