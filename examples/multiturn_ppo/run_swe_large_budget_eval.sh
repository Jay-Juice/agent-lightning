#!/usr/bin/env bash
set -Eeuo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA=/media/ubuntu/D1/zsj/agent-lightning-runtime/data/swe-smith-training/exceptiongroup
case "${1:-validation}" in
  validation) TRAIN_FILE="$DATA/train.jsonl"; VAL_FILE="$DATA/val.jsonl"; VAL_BATCH=6 ;;
  train-pool) TRAIN_FILE="$DATA/val.jsonl"; VAL_FILE="$DATA/train.jsonl"; VAL_BATCH=32 ;;
  *) echo 'Usage: run_swe_large_budget_eval.sh [validation|train-pool] [overrides...]' >&2; exit 2 ;;
esac
if (( $# > 0 )); then shift; fi
export AGL_TRAIN_MODEL="${AGL_TRAIN_MODEL:-/media/ubuntu/D1/zsj/GOPD/G-OPD-main/models/Qwen3-4B-Instruct-2507}"
export AGL_GPUS="${AGL_GPUS:-0,1,2,3}"
export SMITH_MAX_TURNS="${SMITH_MAX_TURNS:-100}"
export SMITH_MAX_TOKENS="${SMITH_MAX_TOKENS:-12288}"
export SMITH_CONTEXT="${SMITH_CONTEXT:-81920}"
export SMITH_OBS_CHAR_CAP="${SMITH_OBS_CHAR_CAP:-6000}"
export SMITH_MODEL_TIMEOUT="${SMITH_MODEL_TIMEOUT:-600}"
# Match the example's interaction budgets. Both splits are evaluation only.
# Extra KV cache capacity accommodates the longer histories without changing weights.
bash "$TOOLS/run_capo_ppo.sh" --steps 1 \
  --train-file "$TRAIN_FILE" --val-file "$VAL_FILE" \
  data.train_batch_size=4 data.val_batch_size="$VAL_BATCH" \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 critic.ppo_mini_batch_size=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
  agentlightning.rollout_timeout_seconds=7200 \
  trainer.val_before_train=true trainer.val_only=true trainer.resume_mode=disable trainer.save_freq=-1 "$@"
