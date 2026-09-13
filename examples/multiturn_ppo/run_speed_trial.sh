#!/usr/bin/env bash
set -Eeuo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME=/media/ubuntu/D1/zsj/agent-lightning-runtime
DATA="$RUNTIME/data/swe-smith-training/speedtrial-20260913"
export AGL_GPUS=4,5,6,7 AGL_TRAIN_PORT=18501
export AGL_TRAIN_TAG="${AGL_TRAIN_TAG:?Set a unique isolated trial tag}"
export AGL_MAX_LOCAL_AGENTS="${AGL_MAX_LOCAL_AGENTS:-32}"
export AGL_MIN_FREE_GIB=150
MICRO="${AGL_SPEED_TRAIN_MICRO:-2}"
FORWARD="${AGL_SPEED_FORWARD_MICRO:-2}"
SEQS="${AGL_SPEED_MAX_SEQS:-8}"
TOKENS="${AGL_SPEED_BATCHED_TOKENS:-16384}"
# These concurrency settings also apply to training rollouts. They were tested
# on full validation; end-to-end PPO step speed still needs separate measurement.
# Equal task batch and optimization minibatches. Only execution batching changes.
# This disposable one-step speed check does not run validation or save weights.
# The separate full entrypoint carries the requested 20/40/3 cadence.
bash "$TOOLS/run_checked_swe_ppo.sh" \
  --steps 1 --train-file "$DATA/train-seeded.jsonl" --val-file "$DATA/val.jsonl" \
  trainer.total_epochs=1 trainer.total_training_steps=1 \
  data.train_batch_size=32 data.val_batch_size=32 agentlightning.full_dataset=true \
  agentlightning.local.agent_class=full_python_agent.FullPythonAgent \
  actor_rollout_ref.actor.fsdp_config.reshard_after_forward=false \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO" \
  critic.ppo_micro_batch_size_per_gpu="$MICRO" \
  critic.forward_micro_batch_size_per_gpu="$FORWARD" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$FORWARD" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$FORWARD" \
  actor_rollout_ref.rollout.max_num_seqs="$SEQS" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$TOKENS" \
  trainer.val_before_train=false trainer.test_freq=-1 trainer.save_freq=-1 \
  trainer.resume_mode=disable "$@"
