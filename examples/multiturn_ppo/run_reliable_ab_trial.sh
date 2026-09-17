#!/usr/bin/env bash
set -Eeuo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM="${1:?Specify A (default head) or B (zero head)}"
shift
case "$ARM" in
  A) HEAD_INIT=default ;;
  B) HEAD_INIT=zero ;;
  *) echo 'Expected A or B'; exit 2 ;;
esac
export AGL_TRAIN_TAG="${AGL_TRAIN_TAG:?Set a unique trial tag}"
export AGL_TRAIN_PORT="${AGL_TRAIN_PORT:-18521}"
export AGL_EXPORT_FINAL_ACTOR=0 AGL_GPU_MONITOR=1
# Keep the full dataset, sampling protocol and task batch32. The trial ends
# after 20 PPO task batches, with initial and final complete validation.
# A and B differ only in fresh value-head initialization.
bash "$TOOLS/run_reliable_full_python_ppo.sh" \
  trainer.total_training_steps=20 trainer.save_freq=20 \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 critic.ppo_mini_batch_size=128 \
  "agentlightning.multi_turn_ppo.critic_head_init=$HEAD_INIT" "$@"
