#!/usr/bin/env bash
set -Eeuo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Conservative SWE PPO profile. Keep terminal binary rewards and native GAE.
# veRL starts actor updates at global_step >= critic_warmup: 5 means 4 warmup steps.
DATA=/media/ubuntu/D1/zsj/agent-lightning-runtime/data/swe-smith-training/exceptiongroup
bash "$TOOLS/run_four_gpu.sh" \
  --steps 12 --train-file "$DATA/train.jsonl" --val-file "$DATA/val.jsonl" \
  data.val_batch_size=6 trainer.total_epochs=2 trainer.test_freq=4 trainer.save_freq=4 \
  trainer.critic_warmup=5 \
  agentlightning.multi_turn_ppo.whiten_advantages=false \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=0.02 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.ref.fsdp_config.param_offload=true "$@"
