#!/usr/bin/env bash
set -Eeuo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# CAPO's token PPO recipe; batch 128 -> 32 only for the cached 32-task SWE pilot.
# Copied update loops, GAE, whitening, clipping, KL, and padding are selected together.
DATA=/media/ubuntu/D1/zsj/agent-lightning-runtime/data/swe-smith-training/exceptiongroup
export AGL_MAX_LOCAL_AGENTS="${AGL_MAX_LOCAL_AGENTS:-8}"
export AGL_GPU_MONITOR=1
bash "$TOOLS/run_four_gpu.sh" \
  --steps 2 --train-file "$DATA/train.jsonl" --val-file "$DATA/val.jsonl" \
  data.train_batch_size=32 data.val_batch_size=6 trainer.total_epochs=2 \
  agentlightning.multi_turn_ppo.backend=capo \
  agentlightning.multi_turn_ppo.distributed_padding=false \
  agentlightning.multi_turn_ppo.whiten_advantages=true \
  algorithm.adv_estimator=token_gae algorithm.gamma=0.99 algorithm.lam=1.0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 critic.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.optim.lr=1e-6 critic.optim.lr=1e-5 \
  actor_rollout_ref.actor.optim.weight_decay=0.01 critic.optim.weight_decay=0.01 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
  actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.2 \
  actor_rollout_ref.actor.clip_ratio_c=3 actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.ppo_epochs=1 critic.ppo_epochs=1 \
  trainer.use_legacy_worker_impl=enable trainer.critic_warmup=0 \
  trainer.save_freq=1 trainer.test_freq=1 "$@"
