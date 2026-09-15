#!/usr/bin/env bash
set -Eeuo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Evaluate the same complete 470-task validation file on GPUs 4-7 only.
# Queue more tasks together to avoid a barrier after every 32 completions.
export AGL_MAX_LOCAL_AGENTS="${AGL_MAX_LOCAL_AGENTS:-32}"
export AGL_SPEED_MAX_SEQS="${AGL_SPEED_MAX_SEQS:-8}"
export AGL_SPEED_BATCHED_TOKENS="${AGL_SPEED_BATCHED_TOKENS:-16384}"
bash "$TOOLS/run_speed_trial.sh" \
  trainer.val_before_train=true trainer.val_only=true \
  data.val_batch_size="${AGL_VAL_BATCH_SIZE:-470}" "$@"
