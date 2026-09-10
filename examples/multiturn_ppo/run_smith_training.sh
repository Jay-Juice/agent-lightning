#!/usr/bin/env bash
set -Eeuo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AGL_TRAIN_MODEL="${AGL_TRAIN_MODEL:-/media/ubuntu/D1/zsj/GOPD/G-OPD-main/models/Qwen3-1.7B}"
export SMITH_MAX_TURNS="${SMITH_MAX_TURNS:-8}" SMITH_MAX_TOKENS="${SMITH_MAX_TOKENS:-768}"
export SMITH_CONTEXT=12288
DATA=/media/ubuntu/D1/zsj/agent-lightning-runtime/data/swe-smith-training/exceptiongroup
bash "$TOOLS/run_training.sh" --steps 1 --train-file "$DATA/train-smoke.jsonl" --val-file "$DATA/val-smoke.jsonl" \
  data.train_batch_size=1 data.val_batch_size=1 data.max_prompt_length=11520 data.max_response_length=768 \
  actor_rollout_ref.rollout.max_model_len=12288 actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.max_num_seqs=2 \
  agentlightning.local.agent_class=smith_docker_agent.SmithDockerAgent \
  agentlightning.rollout_timeout_seconds=1200 "$@"
