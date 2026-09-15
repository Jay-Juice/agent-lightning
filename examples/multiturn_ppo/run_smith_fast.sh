#!/usr/bin/env bash
set -Eeuo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# A800 80GB execution profile: same PPO objective, optimizer batches and precision.
# The no-warmup short run passed; retain the older warmup profile for comparison.
export AGL_GPU_MONITOR=1
export AGL_MAX_LOCAL_AGENTS="${AGL_MAX_LOCAL_AGENTS:-8}"
bash "$TOOLS/run_smith_stable.sh" \
  --steps 8 trainer.total_epochs=1 trainer.critic_warmup=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  critic.model.fsdp_config.param_offload=false critic.model.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.ref.fsdp_config.param_offload=false \
  actor_rollout_ref.model.enable_gradient_checkpointing=false \
  critic.model.enable_gradient_checkpointing=false \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  critic.forward_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.enforce_eager=false "$@"
