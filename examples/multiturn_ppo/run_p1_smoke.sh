#!/usr/bin/env bash
set -Eeuo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME=/media/ubuntu/D1/zsj/agent-lightning-runtime
DATA="$RUNTIME/data/swe-smith-training/speedtrial-20260913"

export AGL_GPUS=0,1,2,3
export AGL_TRAIN_PORT="${AGL_TRAIN_PORT:-18601}"
export AGL_TRAIN_TAG="${AGL_TRAIN_TAG:-capo-swe-p1-smoke-4b-gpu03-$(date +%Y%m%d-%H%M%S)}"
export AGL_MAX_LOCAL_AGENTS=4
export AGL_MIN_FREE_GIB=150
export SMITH_PRIVILEGED_STATE=critic
export SMITH_PRIVILEGED_MAX_TOKENS=4096
export SMITH_PRIVILEGED_SAFETY_MARGIN=32

# One real four-task SWE batch. This validates the P1 data path, Critic update,
# Actor update, and actor+critic checkpoint without claiming an effect result.
bash "$TOOLS/run_checked_swe_ppo.sh" \
  --steps 1 --train-file "$DATA/train-seeded.jsonl" --val-file "$DATA/val.jsonl" \
  trainer.total_epochs=1 trainer.total_training_steps=1 \
  data.train_batch_size=4 data.val_batch_size=4 agentlightning.full_dataset=true \
  agentlightning.local.agent_class=full_python_agent.FullPythonAgent \
  agentlightning.privileged_critic.enabled=true \
  agentlightning.privileged_critic.state_schema_version=1 \
  agentlightning.privileged_critic.max_tokens="$SMITH_PRIVILEGED_MAX_TOKENS" \
  agentlightning.privileged_critic.safety_margin="$SMITH_PRIVILEGED_SAFETY_MARGIN" \
  actor_rollout_ref.actor.fsdp_config.reshard_after_forward=false \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 critic.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  critic.ppo_micro_batch_size_per_gpu=1 critic.forward_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.max_num_seqs=2 \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]' \
  trainer.val_before_train=false trainer.test_freq=-1 trainer.save_freq=1 \
  trainer.max_actor_ckpt_to_keep=1 trainer.max_critic_ckpt_to_keep=1 \
  trainer.resume_mode=disable "$@"
