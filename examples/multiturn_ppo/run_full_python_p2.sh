#!/usr/bin/env bash
set -Eeuo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AGL_GPUS="${AGL_GPUS:-0,1,2,3}"
[[ "$AGL_GPUS" == "0,1,2,3" ]] || { echo 'PI uses physical GPUs 0-3'; exit 2; }
export AGL_TRAIN_PORT="${AGL_TRAIN_PORT:-18701}"
export AGL_TRAIN_TAG="${AGL_TRAIN_TAG:-capo-swe-pythonfull-v7-p2fixed-4b-gpu03-$(date +%Y%m%d-%H%M%S)}"
export AGL_FULL_ENV_AUDIT="${AGL_FULL_ENV_AUDIT:-/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/swe-full-python-envs-p2fixed-20260916-01}"
export AGL_MATCH_BASELINE_RUN="${AGL_MATCH_BASELINE_RUN:-/media/ubuntu/D1/zsj/agent-lightning-runtime/logs/training-capo-swe-pythonfull-v7-4b-gpu47-resume40-20260915-01}"
export AGL_PI_TRAINING_STEPS="${AGL_PI_TRAINING_STEPS:-784}"
[[ "$AGL_PI_TRAINING_STEPS" =~ ^[1-9][0-9]*$ ]] && (( AGL_PI_TRAINING_STEPS <= 784 )) || exit 2
export SMITH_PRIVILEGED_STATE=critic SMITH_PRIVILEGED_ENCODING=semantic_hunks_v2
export SMITH_PRIVILEGED_MAX_TOKENS=4096 SMITH_PRIVILEGED_SAFETY_MARGIN=32
# Inherit the complete checked baseline recipe, including Agent environment.
# Explicit final step overrides prevent a nested smoke default from limiting the run.
bash "$TOOLS/run_full_python_ppo.sh" \
  --steps "$AGL_PI_TRAINING_STEPS" trainer.total_training_steps="$AGL_PI_TRAINING_STEPS" \
  trainer.max_actor_ckpt_to_keep=2 trainer.max_critic_ckpt_to_keep=2 \
  agentlightning.privileged_critic.enabled=true \
  agentlightning.privileged_critic.state_schema_version=1 \
  agentlightning.privileged_critic.encoding="$SMITH_PRIVILEGED_ENCODING" \
  agentlightning.privileged_critic.max_tokens=4096 \
  agentlightning.privileged_critic.safety_margin=32 "$@"
