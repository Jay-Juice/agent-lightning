#!/usr/bin/env bash
set -Eeuo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AGL_GPUS="${AGL_GPUS:-0,1,2,3}"
bash "$TOOLS/run_smith_training.sh" \
  trainer.n_gpus_per_node=4 ray_kwargs.ray_init.num_gpus=4 ray_kwargs.ray_init.num_cpus=16 \
  data.train_batch_size=4 actor_rollout_ref.actor.ppo_mini_batch_size=4 critic.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean critic.loss_agg_mode=seq-mean-token-mean \
  agentlightning.multi_turn_ppo.distributed_padding=true "$@"
