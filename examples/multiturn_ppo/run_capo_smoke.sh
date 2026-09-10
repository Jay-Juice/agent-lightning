#!/usr/bin/env bash
set -Eeuo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AGL_TRAIN_MODEL="${AGL_TRAIN_MODEL:-/media/ubuntu/D1/zsj/GOPD/G-OPD-main/models/Qwen3-0.6B}"
bash "$TOOLS/run_capo_ppo.sh" --steps 2 \
  --train-file "$TOOLS/fixtures/uneven-train.jsonl" --val-file "$TOOLS/fixtures/uneven-val.jsonl" \
  data.train_batch_size=4 actor_rollout_ref.actor.ppo_mini_batch_size=4 critic.ppo_mini_batch_size=4 \
  data.val_batch_size=1 trainer.total_epochs=2 trainer.val_before_train=false trainer.test_freq=-1 \
  data.max_prompt_length=1024 data.max_response_length=128 \
  actor_rollout_ref.rollout.max_model_len=2048 actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
  agentlightning.local.agent_class=smoke_agent.SmokeAgent "$@"
